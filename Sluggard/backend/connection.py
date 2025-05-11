from pymongo import MongoClient

from os import load_dotenv
import os

load_dotenv()
ADDR = os.getenv('MONGO_ADDR')

client = MongoClient(ADDR)

db = client.moexMMD
candle_collection = db.candles


