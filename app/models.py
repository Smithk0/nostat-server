# app/models.py

from config import get_db
from bson.objectid import ObjectId
from datetime import datetime
import pytz

# Make the insert_event function async and accept a date parameter
async def insert_event(sport, event_data, date):
    try:
        db = await get_db()  # Await the async database call
        collection_name = f"{sport.lower()}-{date}"
        collection = db[collection_name]

        event_data['sport'] = sport
        event_data['_id'] = event_data['fixture']['id']

        await collection.replace_one({"_id": event_data['_id']}, event_data, upsert=True)
        print(f"Successfully inserted event {event_data['_id']} into {collection_name}.")
    except Exception as e:
        print(f"Error inserting event {event_data.get('_id', 'Unknown ID')}: {e}")

# Updated get_event_count to accept 'date' as a second parameter
async def get_event_count(sport, date):
    """
    Fetch event counts for a specific sport and date.

    Args:
        sport (str): The sport name (e.g., 'football').
        date (str): The date in 'YYYY-MM-DD' format.

    Returns:
        tuple: (total_events, live_events, not_started_events, finished_events)
    """
    db = await get_db()
    collection_name = f"{sport.lower()}-{date}"
    collection = db[collection_name]
    
    total_events = await collection.count_documents({})
    live_events = 0
    not_started_events = 0
    finished_events = 0

    async for event in collection.find():
        status_long = event.get('fixture', {}).get('status', {}).get('long', '')
        if status_long in ["First Half", "Second Half", "In Progress", "Extra Time"]:
            live_events += 1
        elif status_long == "Not Started":
            not_started_events += 1
        elif status_long == "Match Finished":
            finished_events += 1

    return total_events, live_events, not_started_events, finished_events