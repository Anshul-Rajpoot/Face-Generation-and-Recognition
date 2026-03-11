from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get connection string from environment variable
CONNECTION_STRING = os.getenv('MONGO_CONNECTION_STRING')
if not CONNECTION_STRING:
    raise ValueError("MONGO_CONNECTION_STRING not found in environment variables. Please create a .env file.")

client = MongoClient(CONNECTION_STRING)
db = client["face_recognition_db"]
collection = db["face_data"]

def enroll_face(name, embedding, image_path):
    """Inserts a new face record into MongoDB."""
    try:
        document = {
            "name": name,
            "embedding": embedding,
            "image_path": image_path 
        }
        collection.insert_one(document)
        return True
    except Exception as e:
        print(f"Database insertion failed: {e}")
        return False

def search_faces(query_embedding, limit=3):
    """Performs Vector Search in MongoDB Atlas."""
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index", 
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 50,
                "limit": limit
            }
        },
        {
            "$project": {
                "_id": 0,
                "name": 1,
                "image_path": 1, 
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    return list(collection.aggregate(pipeline))