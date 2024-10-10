# app/services/service_update.py

import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, List
import aiohttp
from app.models import get_db
from config import get_api_football_key
from pytz import utc
from pymongo import UpdateOne
from aiohttp import ClientResponseError, ClientConnectorError
from aiohttp_retry import RetryClient, ExponentialRetry

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
API_BASE_URL = "https://v3.football.api-sports.io/fixtures"
API_KEY = get_api_football_key()
HEADERS = {
    'x-apisports-key': API_KEY,
}

# Time interval for updating scores (in seconds)
UPDATE_INTERVAL = 20 * 60  # 20 minutes

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# ===========================
# Enhanced Utility Functions
# ===========================

async def fetch_fixtures_by_date(session: aiohttp.ClientSession, date: str) -> List[Dict]:
    """
    Fetch all fixtures for a specific date using the API with retry mechanism.
    """
    url = f"{API_BASE_URL}?date={date}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.get(url, headers=HEADERS) as response:
                response.raise_for_status()
                data = await response.json()
                logger.info(f"Successfully fetched fixtures for date {date} on attempt {attempt}.")
                return data.get('response', [])
        except (ClientResponseError, ClientConnectorError) as e:
            logger.error(f"Attempt {attempt}: Error fetching fixtures for date {date}: {e}")
            if attempt < MAX_RETRIES:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                await asyncio.sleep(RETRY_DELAY)
            else:
                logger.error(f"Max retries reached for fetching fixtures on {date}.")
        except Exception as e:
            logger.error(f"Unexpected error while fetching fixtures for date {date}: {e}")
            break  # For unexpected errors, don't retry
    return []

async def update_event_status(session: aiohttp.ClientSession, db, event: Dict, fixture_data: Dict):
    """
    Update the status and score of a single event.
    """
    fixture_id = event['fixture_id']
    logger.info(f"Updating status for fixture {fixture_id}")

    fixture = fixture_data.get('fixture', {})
    status = fixture.get('status', {}).get('short', '')
    current_score = fixture_data.get('score', {})

    # Handle database operations with exception handling
    try:
        await db[f"football-2daybet-{event['date']}"].update_one(
            {"fixture_id": fixture_id},
            {"$set": {"score": current_score, "status": status}},
            upsert=True
        )
        logger.info(f"Updated score and status for fixture {fixture_id} to {status}")
    except Exception as e:
        logger.error(f"Database update failed for fixture {fixture_id}: {e}")
        return  # Exit early if update fails

    # Check if match has finished
    if status in ["FT", "AET", "PEN"]:
        try:
            await db[f"football-2daybet-{event['date']}"].update_one(
                {"fixture_id": fixture_id},
                {"$set": {"starts_in": "Finished", "Prediction_Outcome": "Pending"}}  # Set default to Pending
            )
            logger.info(f"Match {fixture_id} has finished.")
            # After match finishes, update Prediction Outcome
            await update_prediction_outcome(db, fixture_id)
        except Exception as e:
            logger.error(f"Failed to update finished status for fixture {fixture_id}: {e}")

async def monitor_events():
    """
    Main function to monitor and update event statuses.
    Should be scheduled to run periodically (e.g., every 20 minutes).
    """
    db = await get_db()
    current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    collection_name = f"football-2daybet-{current_date}"
    collection = db[collection_name]

    # Set up RetryClient with ExponentialRetry
    retry_options = ExponentialRetry(attempts=MAX_RETRIES, start_timeout=RETRY_DELAY)
    async with RetryClient(retry_options=retry_options) as session:
        # Fetch all fixtures for the current date with retries
        fixtures = await fetch_fixtures_by_date(session, current_date)
        fixture_dict = {fixture['fixture']['id']: fixture for fixture in fixtures}

        # Fetch all events from the 2daybet collection
        try:
            events_cursor = collection.find({})
            events = await events_cursor.to_list(length=None)
        except Exception as e:
            logger.error(f"Failed to fetch events from collection {collection_name}: {e}")
            return

        # Prepare bulk operations
        bulk_updates = []
        finished_fixtures = []

        for event in events:
            fixture_id = event['fixture_id']
            starts_in = event.get('starts_in', '')
            fixture_data = fixture_dict.get(fixture_id)

            if not fixture_data:
                logger.warning(f"No data returned for fixture {fixture_id}")
                continue

            if starts_in == "Started":
                # Prepare update for ongoing matches
                fixture = fixture_data.get('fixture', {})
                status = fixture.get('status', {}).get('short', '')
                current_score = fixture_data.get('score', {})
                bulk_updates.append(
                    UpdateOne(
                        {"fixture_id": fixture_id},
                        {"$set": {"score": current_score, "status": status}},
                        upsert=True
                    )
                )

                if status in ["FT", "AET", "PEN"]:
                    finished_fixtures.append(fixture_id)

            elif starts_in in ["Not Started", "Scheduled"]:
                # Check if the match should be started
                await check_and_start_events(session, db, event)
            elif starts_in == "Finished":
                # Nothing to do for finished matches
                continue
            else:
                logger.warning(f"Unknown 'starts_in' status '{starts_in}' for fixture {fixture_id}")

        # Execute bulk updates
        if bulk_updates:
            try:
                result = await collection.bulk_write(bulk_updates)
                logger.info(f"Bulk update completed: {result.modified_count} documents modified.")
            except Exception as e:
                logger.error(f"Bulk update failed: {e}")

        # Handle finished fixtures with bulk operations
        if finished_fixtures:
            bulk_finished = [
                UpdateOne(
                    {"fixture_id": fixture_id},
                    {"$set": {"starts_in": "Finished", "Prediction_Outcome": "Pending"}}
                ) for fixture_id in finished_fixtures
            ]
            try:
                result = await collection.bulk_write(bulk_finished)
                logger.info(f"Finished fixtures updated: {result.modified_count} documents modified.")
                # Update Prediction Outcome for each finished fixture
                prediction_tasks = [update_prediction_outcome(db, fixture_id) for fixture_id in finished_fixtures]
                await asyncio.gather(*prediction_tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Failed to update finished fixtures: {e}")

async def check_and_start_events(session: aiohttp.ClientSession, db, event: Dict):
    """
    Check if the event should be started based on current time.
    If it's time to start, update 'starts_in' to 'Started'.
    """
    fixture_id = event['fixture_id']
    event_datetime_str = event.get('date')  # Assuming 'date' field holds the match start time in ISO format
    if not event_datetime_str:
        logger.warning(f"No start time found for fixture {fixture_id}")
        return

    try:
        event_datetime = datetime.fromisoformat(event_datetime_str.replace('Z', '+00:00')).astimezone(utc)
    except ValueError as e:
        logger.error(f"Invalid date format for fixture {fixture_id}: {e}")
        return

    now = datetime.now(timezone.utc)

    if now >= event_datetime and event.get('starts_in') != "Started":
        # Update status to Started with exception handling
        try:
            await db[f"football-2daybet-{event['date']}"].update_one(
                {"fixture_id": fixture_id},
                {"$set": {"starts_in": "Started"}}
            )
            logger.info(f"Match {fixture_id} has started.")
        except Exception as e:
            logger.error(f"Failed to update 'Started' status for fixture {fixture_id}: {e}")

def categorize_prediction(best_bet: str, actual_outcome: str, event: Dict) -> str:
    """
    Categorize the prediction outcome based on best_bet and actual outcome.
    """
    # If the match is not yet finished, the outcome is Pending
    if event.get('starts_in') != "Finished":
        return "Pending"

    score = event.get('score', {})
    full_time = score.get('fulltime', {})
    halftime = score.get('halftime', {})

    home_goals = full_time.get('home', 0)
    away_goals = full_time.get('away', 0)
    ht_home_goals = halftime.get('home', 0)
    ht_away_goals = halftime.get('away', 0)

    if best_bet == "Home Win":
        return "Won" if actual_outcome == "Home Win" else "Lost"
    elif best_bet == "Away Win":
        return "Won" if actual_outcome == "Away Win" else "Lost"
    elif best_bet == "Draw":
        return "Won" if actual_outcome == "Draw" else "Lost"
    elif best_bet == "Home Win or Draw":
        return "Won" if actual_outcome in ["Home Win", "Draw"] else "Lost"
    elif best_bet == "Away Win or Draw":
        return "Won" if actual_outcome in ["Away Win", "Draw"] else "Lost"
    elif best_bet == "Half Time Draw":
        return "Won" if ht_home_goals == ht_away_goals else "Lost"
    elif best_bet == "Both Teams to Score":
        return "Won" if home_goals > 0 and away_goals > 0 else "Lost"
    elif best_bet.startswith("Most Likely Score"):
        # Extract the predicted score
        try:
            predicted_score = best_bet.split(": ")[1]
            predicted_home, predicted_away = map(int, predicted_score.split("-"))
            return "Won" if (home_goals == predicted_home and away_goals == predicted_away) else "Lost"
        except (IndexError, ValueError):
            logger.error(f"Error parsing Most Likely Score from best_bet: {best_bet}")
            return "Voided"
    elif best_bet.startswith("Highest Scoring Half"):
        # Extract the predicted half
        try:
            predicted_half = best_bet.split(": ")[1]
            total_ht_goals = ht_home_goals + ht_away_goals
            total_sh_goals = home_goals + away_goals - total_ht_goals
            if predicted_half == "First Half":
                return "Won" if total_ht_goals > total_sh_goals else "Lost"
            elif predicted_half == "Second Half":
                return "Won" if total_sh_goals > total_ht_goals else "Lost"
            elif predicted_half == "Equal":
                return "Won" if total_ht_goals == total_sh_goals else "Lost"
            else:
                return "Voided"
        except IndexError:
            logger.error(f"Error parsing Highest Scoring Half from best_bet: {best_bet}")
            return "Voided"
    elif best_bet.startswith("Total Over"):
        # Extract the threshold
        try:
            threshold = float(best_bet.split("(")[1].split(")")[0])
            total_goals = home_goals + away_goals
            return "Won" if total_goals > threshold else "Lost"
        except (IndexError, ValueError):
            logger.error(f"Error parsing Total Over from best_bet: {best_bet}")
            return "Voided"
    elif best_bet.startswith("Total Under"):
        try:
            threshold = float(best_bet.split("(")[1].split(")")[0])
            total_goals = home_goals + away_goals
            return "Won" if total_goals < threshold else "Lost"
        except (IndexError, ValueError):
            logger.error(f"Error parsing Total Under from best_bet: {best_bet}")
            return "Voided"
    elif best_bet.startswith("Home Over"):
        try:
            threshold = float(best_bet.split("(")[1].split(")")[0])
            home_sh_goals = home_goals - ht_home_goals
            return "Won" if home_sh_goals > threshold else "Lost"
        except (IndexError, ValueError):
            logger.error(f"Error parsing Home Over from best_bet: {best_bet}")
            return "Voided"
    elif best_bet.startswith("Home Under"):
        try:
            threshold = float(best_bet.split("(")[1].split(")")[0])
            home_sh_goals = home_goals - ht_home_goals
            return "Won" if home_sh_goals < threshold else "Lost"
        except (IndexError, ValueError):
            logger.error(f"Error parsing Home Under from best_bet: {best_bet}")
            return "Voided"
    elif best_bet.startswith("Away Over"):
        try:
            threshold = float(best_bet.split("(")[1].split(")")[0])
            away_sh_goals = away_goals - ht_away_goals
            return "Won" if away_sh_goals > threshold else "Lost"
        except (IndexError, ValueError):
            logger.error(f"Error parsing Away Over from best_bet: {best_bet}")
            return "Voided"
    elif best_bet.startswith("Away Under"):
        try:
            threshold = float(best_bet.split("(")[1].split(")")[0])
            away_sh_goals = away_goals - ht_away_goals
            return "Won" if away_sh_goals < threshold else "Lost"
        except (IndexError, ValueError):
            logger.error(f"Error parsing Away Under from best_bet: {best_bet}")
            return "Voided"
    else:
        return "Voided"

async def update_prediction_outcome(db, fixture_id: int):
    """
    Update the Prediction_Outcome field based on the actual match result.
    """
    current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    collection_name = f"football-2daybet-{current_date}"
    collection = db[collection_name]
    try:
        event = await collection.find_one({"fixture_id": fixture_id})
    except Exception as e:
        logger.error(f"Failed to fetch event {fixture_id} from collection {collection_name}: {e}")
        return

    if not event:
        logger.error(f"Event {fixture_id} not found in collection {collection_name}")
        return

    best_bet = event.get('best_bet', "")
    score = event.get('score', {})
    full_time = score.get('fulltime', {})
    halftime = score.get('halftime', {})
    home_goals = full_time.get('home', 0)
    away_goals = full_time.get('away', 0)
    ht_home_goals = halftime.get('home', 0)
    ht_away_goals = halftime.get('away', 0)

    # Determine actual outcome
    if home_goals > away_goals:
        actual_outcome = "Home Win"
    elif away_goals > home_goals:
        actual_outcome = "Away Win"
    else:
        actual_outcome = "Draw"

    # Determine Prediction Outcome
    prediction_outcome = categorize_prediction(best_bet, actual_outcome, event)

    # Update the document with exception handling
    try:
        await collection.update_one(
            {"fixture_id": fixture_id},
            {"$set": {"Prediction_Outcome": prediction_outcome}}
        )
        logger.info(f"Updated Prediction_Outcome for fixture {fixture_id} to {prediction_outcome}")
    except Exception as e:
        logger.error(f"Failed to update Prediction_Outcome for fixture {fixture_id}: {e}")

async def update_all_prediction_outcomes():
    """
    Iterate through all finished matches and update their Prediction_Outcome if not already set.
    """
    db = await get_db()
    current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    collection_name = f"football-2daybet-{current_date}"
    collection = db[collection_name]

    try:
        # Find all finished matches without Prediction_Outcome
        finished_events_cursor = collection.find({
            "starts_in": "Finished",
            "Prediction_Outcome": {"$exists": False}
        })
        finished_events = await finished_events_cursor.to_list(length=None)
    except Exception as e:
        logger.error(f"Failed to fetch finished events from collection {collection_name}: {e}")
        return

    prediction_tasks = []
    for event in finished_events:
        fixture_id = event['fixture_id']
        prediction_tasks.append(update_prediction_outcome(db, fixture_id))

    if prediction_tasks:
        try:
            await asyncio.gather(*prediction_tasks, return_exceptions=True)
            logger.info(f"Updated Prediction_Outcome for {len(prediction_tasks)} fixtures.")
        except Exception as e:
            logger.error(f"Error updating Prediction_Outcome: {e}")

async def scheduled_score_updates():
    """
    Continuously run scheduled score updates every 20 minutes.
    """
    while True:
        try:
            logger.info("Starting scheduled event status update.")
            await monitor_events()
            logger.info("Completed scheduled event status update.")
        except Exception as e:
            logger.error(f"Error during scheduled event status update: {e}")
        await asyncio.sleep(UPDATE_INTERVAL)

def start_background_tasks():
    """
    Start background tasks when the server starts.
    """
    loop = asyncio.get_event_loop()
    loop.create_task(scheduled_score_updates())
    loop.create_task(update_all_prediction_outcomes())
    logger.info("Background tasks have been started.")
