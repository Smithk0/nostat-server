from flask import Flask
from pymongo import MongoClient
from config import MONGO_URI

# Initialize Flask app and MongoDB connection
app = Flask(__name__)

# MongoDB setup
client = MongoClient(MONGO_URI)
db = client['sports_database']  # Database name

from app.routes import *
