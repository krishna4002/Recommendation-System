# backend/vector_store.py
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key and environment from environment variables
api_key = os.getenv("PINECONE_API_KEY")
environment = os.getenv("PINECONE_ENV")  # No longer used in the same way, but you might use it for region info

# Create Pinecone client
pc = Pinecone(api_key=api_key)

# Define your index name
index_name = "recommendation-system"

# Check if index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Set the actual dimension based on your embeddings
        metric='cosine',  # or 'euclidean', etc.
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'  # Replace with your actual region
        )
    )

# Connect to the index
index = pc.Index(index_name)
