#config.py

from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URI = "mongodb://admin:greystats@68.183.37.52:27017/admin"  # URI for your remote MongoDB
DATABASE_NAME = "sportsdb"

# List of API keys for rotation 
API_FOOTBALL_KEYS = [
    "292bdea5dda5c2507cb6c18331de9057",
]

# Initialize the current API key index and counter
current_key_index = 0
api_call_counter = 0

async def get_db():
    """Initialize and return the async MongoDB database connection."""
    client = AsyncIOMotorClient(MONGO_URI)  # Use the asynchronous MongoDB client
    db = client[DATABASE_NAME]
    return db

def get_api_football_key():
    global current_key_index, api_call_counter

    # Fetch the current API key
    api_key = API_FOOTBALL_KEYS[current_key_index]

    # Increment the API call counter
    api_call_counter += 1

    # Rotate the API key after 90 calls
    if api_call_counter >= 90:
        current_key_index = (current_key_index + 1) % len(API_FOOTBALL_KEYS)  # Rotate to the next key
        api_call_counter = 0  # Reset the counter after rotating the key

    return api_key


