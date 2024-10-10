# app/services/scraping_service.py

import aiohttp
import asyncio
import pytz
from datetime import datetime, timedelta
from app.models import insert_event
from config import get_db, get_api_football_key
from app.services.prediction_service import predict_football
from aiohttp_retry import RetryClient, ExponentialRetry

# Retry options for aiohttp retry
retry_options = ExponentialRetry(attempts=3)

async def fetch(session, url, headers):
    """Fetch data asynchronously with retry logic."""
    try:
        async with session.get(url, headers=headers) as response:
            response.raise_for_status()
            return await response.json()
    except aiohttp.ClientResponseError as e:
        print(f"HTTP Error: {e.status} - {e.message}")
        return None
    except aiohttp.ClientError as e:
        print(f"Error fetching data: {e}")
        return None

async def fetch_daily_fixtures(session, date):
    """Fetch football fixtures for a specific date that have not started."""
    url = f"https://v3.football.api-sports.io/fixtures?date={date}"
    headers = {
        'x-apisports-key': get_api_football_key(),
    }
    return await fetch(session, url, headers)

async def fetch_h2h_data(session, team_a_id, team_b_id):
    """Fetch head-to-head data for two teams."""
    url = f"https://v3.football.api-sports.io/fixtures/headtohead?h2h={team_a_id}-{team_b_id}"
    headers = {
        'x-apisports-key': get_api_football_key(),
    }
    return await fetch(session, url, headers)

async def fetch_last_10_matches(session, team_id):
    """Fetch the last 10 matches for a given team."""
    url = f"https://v3.football.api-sports.io/fixtures?team={team_id}&last=10"
    headers = {
        'x-apisports-key': get_api_football_key(),
    }
    return await fetch(session, url, headers)

async def create_statistics_collection(session, date):
    """Create or update a statistics collection for the given date."""
    try:
        db = await get_db()
        print("Connected to database.")
        collection_name = f"football-stats-{date}"
        main_collection_name = f"football-{date}"
        main_collection = db[main_collection_name]
        main_matches_cursor = main_collection.find()
        main_matches = {}
        async for match in main_matches_cursor:
            main_matches[match['fixture']['id']] = match
        print(f"Fetched {len(main_matches)} matches from main_collection for date {date}.")

        if main_matches:
            for fixture_id, match in main_matches.items():
                print(f"Processing fixture {fixture_id} for date {date}")
                home_team_id = match['teams']['home']['id']
                away_team_id = match['teams']['away']['id']

                # Fetch head-to-head data
                h2h_data = await fetch_h2h_data(session, home_team_id, away_team_id)

                if h2h_data is None or 'response' not in h2h_data:
                    print(f"Warning: No H2H data available for fixture {fixture_id}")
                    h2h_data_response = []
                else:
                    h2h_data_response = h2h_data.get('response', [])

                # Fetch last 10 matches for both teams
                last_10_home = await fetch_last_10_matches(session, home_team_id)
                last_10_away = await fetch_last_10_matches(session, away_team_id)

                # Handle NoneType issue in case of a failed fetch
                last_10_home_response = last_10_home.get('response', []) if last_10_home else []
                last_10_away_response = last_10_away.get('response', []) if last_10_away else []

                # Create a document for the statistics
                stats_document = {
                    'fixture_id': fixture_id,
                    'date': date,
                    'home_team': match['teams']['home'],
                    'away_team': match['teams']['away'],
                    'h2h': h2h_data_response,
                    'last_10_home': last_10_home_response,
                    'last_10_away': last_10_away_response
                }

                # Insert or update the statistic document in MongoDB
                collection = db[collection_name]
                await collection.replace_one(
                    {'fixture_id': fixture_id},  # Use fixture_id as the unique identifier
                    stats_document,
                    upsert=True  # Insert if it doesn't exist
                )
                print(f"Upserted statistics for fixture {fixture_id} into collection {collection_name}.")
        else:
            print(f"No matches found in main_collection for date {date}.")
    except Exception as e:
        print(f"Exception in create_statistics_collection for date {date}: {e}")

async def store_predictions(session, date):
    """Store or update predictions for matches on a specific date."""
    try:
        db = await get_db()
        print("Connected to database for storing predictions.")
        stats_collection_name = f"football-stats-{date}"
        bets_collection_name = f"football-2daybet-{date}"
        
        # Fetch match data from the stats collection
        stats_collection = db[stats_collection_name]
        stats_matches_cursor = stats_collection.find()
        stats_matches = []
        async for match in stats_matches_cursor:
            stats_matches.append(match)
        print(f"Fetched {len(stats_matches)} matches from stats_collection for date {date}.")

        # Fetch match data from the main collection
        main_collection_name = f"football-{date}"
        main_collection = db[main_collection_name]
        main_matches_cursor = main_collection.find()
        main_matches = {}
        async for match in main_matches_cursor:
            main_matches[match['fixture']['id']] = match
        print(f"Fetched {len(main_matches)} matches from main_collection for date {date}.")

        # Prepare to store or update predictions
        for stat_match in stats_matches:
            fixture_id = stat_match['fixture_id']
            home_team = stat_match['home_team']['name']
            away_team = stat_match['away_team']['name']
            event_date = stat_match['date']

            # Fetch additional match details
            main_match_data = main_matches.get(fixture_id)
            
            if main_match_data:
                home_logo = main_match_data['teams']['home']['logo']
                away_logo = main_match_data['teams']['away']['logo']
                match_time = datetime.fromisoformat(main_match_data['fixture']['date'].replace('Z', '+00:00'))
                match_time = match_time.astimezone(pytz.utc)
                
                league_name = main_match_data['league']['name']
                country = main_match_data['league']['country']

                if 'h2h' in stat_match and stat_match['h2h']:
                    predictions_result = predict_football(stat_match)
                    final_predictions = predictions_result.get('final_predictions', {})
                    best_bet = final_predictions.get('best_bet', "No prediction available")
                    confidence_score = final_predictions.get('confidence_score', "0%")
                    monte_carlo_simulation = final_predictions.get('monte_carlo_simulation', {})
                    best_bet_options = final_predictions.get('best_bet_options', [])
                    # Extract additional fields
                    avg_home_goals = final_predictions.get('avg_home_goals', 0.0)
                    avg_away_goals = final_predictions.get('avg_away_goals', 0.0)
                    home_win_streak = final_predictions.get('home_win_streak', 0)
                    away_win_streak = final_predictions.get('away_win_streak', 0)
                    total_goals_conceded_home = final_predictions.get('total_goals_conceded_home', 0)
                    total_goals_conceded_away = final_predictions.get('total_goals_conceded_away', 0)
                else:
                    best_bet = "No prediction available due to lack of H2H data"
                    confidence_score = "0%"
                    monte_carlo_simulation = {}
                    best_bet_options = []
                    avg_home_goals = 0.0
                    avg_away_goals = 0.0
                    home_win_streak = 0
                    away_win_streak = 0
                    total_goals_conceded_home = 0
                    total_goals_conceded_away = 0

                result = "Not available"  # Default value for result

                fixture_status_long = main_match_data.get('fixture', {}).get('status', {}).get('long', '')
                fixture_status_short = main_match_data.get('fixture', {}).get('status', '')

                if fixture_status_long == 'Match Finished':
                    result = f"{main_match_data['goals']['home']} - {main_match_data['goals']['away']}"
                    starts_in = "Finished"  # Store 'Finished' for completed matches
                else:
                    starts_in = match_time.isoformat()  # Store the match start time in ISO format for future matches

                # Prepare the prediction document
                prediction_document = {
                    "team_a": {"name": home_team, "logo": home_logo},
                    "team_b": {"name": away_team, "logo": away_logo},
                    "fixture_id": fixture_id,
                    "date": event_date,
                    "best_bet": best_bet,
                    "best_bet_options": best_bet_options,
                    "monte_carlo_simulation": monte_carlo_simulation,
                    "avg_home_goals": avg_home_goals,
                    "avg_away_goals": avg_away_goals,
                    "home_win_streak": home_win_streak,
                    "away_win_streak": away_win_streak,
                    "total_goals_conceded_home": total_goals_conceded_home,
                    "total_goals_conceded_away": total_goals_conceded_away,
                    "result": result,
                    "starts_in": starts_in,  # Store match start time in ISO format or 'Finished'
                    "league_name": league_name,
                    "country": country,
                    "confidence_score": confidence_score,
                    "Prediction_Outcome": "Pending"  # Default value when storing
                }

                # Upsert the prediction into the bets collection
                bets_collection = db[bets_collection_name]
                await bets_collection.replace_one(
                    {'fixture_id': fixture_id}, 
                    prediction_document,
                    upsert=True  
                )
                print(f"Upserted prediction for fixture {fixture_id} into collection {bets_collection_name}.")
            else:
                print(f"No main match data found for fixture {fixture_id} on date {date}")
    except Exception as e:
        print(f"Exception in store_predictions for date {date}: {e}")

async def scrape_daily_results_and_stats():
    """Fetch today's and tomorrow's results and statistics, saving/updating them in MongoDB."""
    async with RetryClient(raise_for_status=False, retry_options=retry_options) as session:
        # Determine current date and next date
        utc_now = datetime.utcnow()
        current_date = utc_now.strftime('%Y-%m-%d')
        next_date = (utc_now + timedelta(days=1)).strftime('%Y-%m-%d')
        dates_to_process = [current_date, next_date]

        for date in dates_to_process:
            print(f"\nProcessing data for date: {date}")
            
            # Fetch fixtures for the date
            print("Starting fetch_daily_fixtures")
            fixtures_data = await fetch_daily_fixtures(session, date)
            
            if fixtures_data and fixtures_data.get('response'):
                fixtures_to_process = fixtures_data['response'][:900]
                print(f"Fetched {len(fixtures_to_process)} fixtures to process for date {date}.")

                for match in fixtures_to_process:
                    await insert_event('football', match, date)  # Pass the date here
                    print(f"Inserted match {match['fixture']['id']} into MongoDB for date {date}.")
            else:
                print(f"No fixtures to process for date {date}.")
            
            # Create or update statistics collection for the date
            print("Starting create_statistics_collection()")
            try:
                await create_statistics_collection(session, date)
                print("Finished create_statistics_collection()")
            except Exception as e:
                print(f"Error while creating statistics collection for date {date}: {e}")
                continue  # Proceed to next date
            
            # Call store_predictions to store or update predictions after statistics are created
            print("Starting store_predictions()")
            try:
                await store_predictions(session, date)
                print("Finished store_predictions()")
            except Exception as e:
                print(f"Error while storing predictions for date {date}: {e}")