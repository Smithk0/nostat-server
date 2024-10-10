# app/routes.py

import logging
from quart import Quart, jsonify, request
from quart_cors import cors  # Import quart-cors for handling CORS
from app.models import get_event_count
from datetime import datetime
from config import get_db
from bson.objectid import ObjectId
import re

# Initialize Quart app
app = Quart(__name__)
app = cors(app, allow_origin="*")  # Enable CORS and allow all origins for development

# Set up logging configuration
logging.basicConfig(level=logging.INFO)  # Can be changed to DEBUG to see detailed logs
logger = logging.getLogger(__name__)

# Utility function to validate date format
def validate_date(date_str):
    """Validate that the date string is in YYYY-MM-DD format."""
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False
    return False

# Define the route to get event count by sport with optional date
@app.route('/api/events/count/<sport>', methods=['GET'])
async def event_count(sport):
    """
    Get the count of events for a specific sport and date.

    Query Parameters:
        date (str): Optional. Date in YYYY-MM-DD format. Defaults to current date.
    """
    date = request.args.get('date')
    if date:
        if not validate_date(date):
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400
    else:
        date = datetime.utcnow().strftime('%Y-%m-%d')  # Use UTC to align with data storage

    # Optional: Validate if 'sport' is valid
    valid_sports = ['basketball', 'football', 'tennis', 'baseball', 'hockey', 'volleyball',
                   'boxing', 'soccer', 'american football', 'golf', 'darts', 'table tennis',
                   'handball', 'beach volleyball', 'cricket', 'rugby', 'snooker', 'badminton']
    
    if sport.lower() not in [s.lower() for s in valid_sports]:
        return jsonify({"error": f"Invalid sport '{sport}'. Available sports are: {', '.join(valid_sports)}."}), 400

    logger.info(f"Fetching event counts for sport: {sport}, date: {date}")

    try:
        total_events, live_events, not_started_events, finished_events = await get_event_count(sport, date)
    except Exception as e:
        logger.error(f"Error fetching event counts: {e}")
        return jsonify({"error": "Internal server error while fetching event counts."}), 500

    return jsonify({
        "date": date,
        "total_events": total_events,
        "live_now": live_events,
        "not_started": not_started_events,
        "finished": finished_events
    }), 200

# Define the route to get matches for a specific sport and date with pagination
@app.route('/api/matches/<sport>', methods=['GET'])
async def matches(sport):
    """
    Get matches for a specific sport and date with pagination.

    Query Parameters:
        date (str): Optional. Date in YYYY-MM-DD format. Defaults to current date.
        limit (int): Optional. Number of records to return. Defaults to 10.
        offset (int): Optional. Number of records to skip. Defaults to 0.
    """
    date = request.args.get('date')
    if date:
        if not validate_date(date):
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400
    else:
        date = datetime.utcnow().strftime('%Y-%m-%d')  # Use UTC to align with data storage

    # Validate limit and offset
    try:
        limit = int(request.args.get('limit', 10))
        offset = int(request.args.get('offset', 0))
        if limit < 1 or offset < 0:
            raise ValueError
    except ValueError:
        return jsonify({"error": "Invalid pagination parameters. 'limit' must be >=1 and 'offset' must be >=0."}), 400

    bets_collection_name = f"{sport}-2daybet-{date}"

    db = await get_db()  # Make sure to await get_db() since it's async
    bets_collection = db[bets_collection_name]
    
    # Check if the collection exists
    if bets_collection_name not in await db.list_collection_names():
        return jsonify({"error": f"No data found for date {date}."}), 404

    predictions_cursor = bets_collection.find().skip(offset).limit(limit)
    predictions = await predictions_cursor.to_list(length=limit)

    results = []
    for prediction in predictions:
        results.append({
            "team_a": prediction.get('team_a', {}),
            "team_b": prediction.get('team_b', {}),
            "fixture_id": prediction.get('fixture_id'),
            "date": prediction.get('date'),
            "best_bet": prediction.get('best_bet'),
            "result": prediction.get('result'),
            "starts_in": prediction.get('starts_in'),
            "league_name": prediction.get('league_name'),
            "country": prediction.get('country'),
            "confidence_score": prediction.get('confidence_score', "0%"),
            "prediction_outcome": prediction.get('Prediction_Outcome', "Pending"),
        })

    total_count = await bets_collection.count_documents({})
    response = {
        'date': date,
        'results': results,
        'total_count': total_count
    }

    return jsonify(response), 200

# Fetch full event details with fixture id and optional date
@app.route('/api/event/<sport>/<int:fixture_id>', methods=['GET'])
async def get_event_details(sport, fixture_id):
    """
    Fetch full event details based on sport, fixture_id, and optional date.

    Query Parameters:
        date (str): Optional. Date in YYYY-MM-DD format. Defaults to current date.
    """
    date = request.args.get('date')
    if date:
        if not validate_date(date):
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400
    else:
        date = datetime.utcnow().strftime('%Y-%m-%d')  # Use UTC to align with data storage

    bets_collection_name = f"{sport}-2daybet-{date}"

    db = await get_db()
    
    # Check if the collection exists
    if bets_collection_name not in await db.list_collection_names():
        return jsonify({"error": f"No data found for date {date}."}), 404

    bets_collection = db[bets_collection_name]

    # Query the document with the specified fixture_id
    event = await bets_collection.find_one({"fixture_id": fixture_id})

    if not event:
        logger.info(f"Event with fixture_id {fixture_id} not found in {bets_collection_name}.")
        return jsonify({"error": "Event not found."}), 404

    # Convert ObjectId to string if present
    if "_id" in event:
        event["_id"] = str(event["_id"])

    return jsonify(event), 200

# Define a new route to get matches for a specific sport and date
@app.route('/api/matches/<sport>/<date>', methods=['GET'])
async def matches_specific_date(sport, date):
    """
    Get matches for a specific sport and date with pagination.

    URL Parameters:
        sport (str): The sport to query (e.g., football).
        date (str): The date in YYYY-MM-DD format.

    Query Parameters:
        limit (int): Optional. Number of records to return. Defaults to 10.
        offset (int): Optional. Number of records to skip. Defaults to 0.
    """
    if not validate_date(date):
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

    # Validate limit and offset
    try:
        limit = int(request.args.get('limit', 10))
        offset = int(request.args.get('offset', 0))
        if limit < 1 or offset < 0:
            raise ValueError
    except ValueError:
        return jsonify({"error": "Invalid pagination parameters. 'limit' must be >=1 and 'offset' must be >=0."}), 400

    bets_collection_name = f"{sport}-2daybet-{date}"

    db = await get_db()  # Make sure to await get_db() since it's async
    bets_collection = db[bets_collection_name]
    
    # Check if the collection exists
    if bets_collection_name not in await db.list_collection_names():
        return jsonify({"error": f"No data found for date {date}."}), 404

    predictions_cursor = bets_collection.find().skip(offset).limit(limit)
    predictions = await predictions_cursor.to_list(length=limit)

    results = []
    for prediction in predictions:
        results.append({
            "team_a": prediction.get('team_a', {}),
            "team_b": prediction.get('team_b', {}),
            "fixture_id": prediction.get('fixture_id'),
            "date": prediction.get('date'),
            "best_bet": prediction.get('best_bet'),
            "result": prediction.get('result'),
            "starts_in": prediction.get('starts_in'),
            "league_name": prediction.get('league_name'),
            "country": prediction.get('country'),
            "confidence_score": prediction.get('confidence_score', "0%"),
            "prediction_outcome": prediction.get('Prediction_Outcome', "Pending"),
        })

    total_count = await bets_collection.count_documents({})
    response = {
        'date': date,
        'results': results,
        'total_count': total_count
    }

    return jsonify(response), 200

# New Route for Searching Matches
@app.route('/api/matches/search/<sport>', methods=['GET'])
async def search_matches(sport):
    """
    Search matches for a specific sport and date with pagination.

    Query Parameters:
        date (str): Optional. Date in YYYY-MM-DD format. Defaults to current date.
        query (str): Optional. Search term to match team names or country.
        limit (int): Optional. Number of records to return. Defaults to 10.
        offset (int): Optional. Number of records to skip. Defaults to 0.
    """
    date = request.args.get('date')
    search_query = request.args.get('query', '').strip().lower()
    if date:
        if not validate_date(date):
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400
    else:
        date = datetime.utcnow().strftime('%Y-%m-%d')  # Use UTC to align with data storage

    # Validate limit and offset
    try:
        limit = int(request.args.get('limit', 10))
        offset = int(request.args.get('offset', 0))
        if limit < 1 or offset < 0:
            raise ValueError
    except ValueError:
        return jsonify({"error": "Invalid pagination parameters. 'limit' must be >=1 and 'offset' must be >=0."}), 400

    bets_collection_name = f"{sport}-2daybet-{date}"

    db = await get_db()  # Make sure to await get_db() since it's async
    bets_collection = db[bets_collection_name]
    
    # Check if the collection exists
    if bets_collection_name not in await db.list_collection_names():
        return jsonify({"error": f"No data found for date {date}."}), 404

    # Build the search filter
    if search_query:
        filter_query = {
            "$or": [
                {"team_a.name": {"$regex": re.escape(search_query), "$options": "i"}},
                {"team_b.name": {"$regex": re.escape(search_query), "$options": "i"}},
                {"country": {"$regex": re.escape(search_query), "$options": "i"}},
            ]
        }
    else:
        filter_query = {}

    predictions_cursor = bets_collection.find(filter_query).skip(offset).limit(limit)
    predictions = await predictions_cursor.to_list(length=limit)

    results = []
    for prediction in predictions:
        results.append({
            "team_a": prediction.get('team_a', {}),
            "team_b": prediction.get('team_b', {}),
            "fixture_id": prediction.get('fixture_id'),
            "date": prediction.get('date'),
            "best_bet": prediction.get('best_bet'),
            "result": prediction.get('result'),
            "starts_in": prediction.get('starts_in'),
            "league_name": prediction.get('league_name'),
            "country": prediction.get('country'),
            "confidence_score": prediction.get('confidence_score', "0%"),
            "prediction_outcome": prediction.get('Prediction_Outcome', "Pending"),
        })

    total_count = await bets_collection.count_documents(filter_query)
    response = {
        'date': date,
        'results': results,
        'total_count': total_count
    }

    return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=False)
