"""
Script to recreate the Milvus collection with the correct vector dimensions
"""
from app.database import recreate_collection

if __name__ == "__main__":
    print("Recreating Milvus collection with updated vector dimensions...")
    collection = recreate_collection()
    print(f"Collection recreated successfully with vector dimension: {collection.schema.fields[-1].params['dim']}")
    print("You can now run the application and upload documents.")
