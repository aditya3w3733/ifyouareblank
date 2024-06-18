import openai
import json
import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

pinecone = Pinecone(api_key=pinecone_api_key)
index_name = 'submission-index'

if index_name not in pinecone.list_indexes().names():
    spec = ServerlessSpec(cloud='aws', region='us-east-1')
    pinecone.create_index(name=index_name, dimension=1536, metric='cosine', spec=spec)

index = pinecone.Index(name=index_name)

with open('submissions_for_vectorization_without_year.txt', 'r') as file:
    text_data = file.readlines()

with open('updated_submissions_with_comments_with_year.json', 'r') as f:
    original_json_data = json.load(f)

def generate_embeddings(texts, retries=3):
    try:
        response = openai.Embedding.create(
            input=texts,
            model="text-embedding-3-small",
            timeout=120  # Set a longer timeout (in seconds)
        )
        return [item['embedding'] for item in response['data']]
    except openai.error.RateLimitError as e:
        print(f"Rate limit hit: {str(e)}")  # Log the error message
        if retries > 0:
            wait_time = 60  # Default wait time of 60 seconds
            print(f"Sleeping for {wait_time} seconds before retrying...")
            time.sleep(wait_time)
            return generate_embeddings(texts, retries - 1)
        else:
            raise Exception("Failed after multiple retries.") from e
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if retries > 0:
            print("Unexpected error, retrying in 30 seconds...")
            time.sleep(30)
            return generate_embeddings(texts, retries - 1)
        else:
            raise


def serialize_metadata(metadata):
    for key, value in metadata.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            metadata[key] = json.dumps(value)  # Serialize list of dicts to JSON string
        elif isinstance(value, list):
            metadata[key] = ', '.join(map(str, value))  # Convert list of strings to a single comma-separated string
        else:
            metadata[key] = str(value)  # Ensure all values are strings
    return metadata

# batch process text data to generate embeddings and store in Pinecone
batch_size = 32
for i in range(0, len(text_data), batch_size):
    batch_texts = text_data[i:i + batch_size]
    embeddings = generate_embeddings(batch_texts)
    ids = [str(j) for j in range(i, i + len(batch_texts))]
    upsert_data = [(ids[k], embeddings[k], serialize_metadata(original_json_data[k])) for k in range(len(ids))]
    # Correctly format the data for upsert
    for id, values, metadata in upsert_data:
        index.upsert(vectors=[(id, values, metadata)])  # make sure each upsert call has structured data

print("Data vectorized and stored in Pinecone successfully.")
