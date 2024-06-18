import openai
import json
import os
import time
import re
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

pinecone = Pinecone(api_key=pinecone_api_key)
index_name = 'submission-index1'

with open('submissions_for_vectorization_without_year.txt', 'r') as file:
    text_data = file.readlines()

with open('updated_submissions_with_comments_with_year.json', 'r') as f:
    original_json_data = json.load(f)

if index_name not in pinecone.list_indexes().names():
    spec = ServerlessSpec(cloud='aws', region='us-east-1')
    pinecone.create_index(name=index_name, dimension=1536, metric='cosine', spec=spec)

index = pinecone.Index(name=index_name)


# function with timeouts
def generate_embeddings(texts, retries=3):
    try:
        response = openai.Embedding.create(
            input=texts,
            model="text-embedding-3-small",
            timeout=120  
        )
        return [item['embedding'] for item in response['data']]
    except openai.error.RateLimitError as e:
        wait_message = str(e)
        wait_time = re.findall(r'Please try again in (\d+)m(\d+)s', wait_message)
        if wait_time:
            minutes, seconds = map(int, wait_time[0])
            wait_seconds = minutes * 60 + seconds
        else:
            wait_seconds = 300  # Default wait time if parsing fails

        print(f"Rate limit hit, sleeping for {wait_seconds} seconds...")
        time.sleep(wait_seconds)
        if retries > 0:
            return generate_embeddings(texts, retries - 1)
        else:
            raise Exception("Failed after multiple retries.") from e
    except Exception as e:
        if retries > 0:
            print(f"An error occurred: {str(e)}, retrying in 30 seconds...")
            time.sleep(30)
            return generate_embeddings(texts, retries - 1)
        else:
            raise

#load and cache embeddings
cache_file = 'embeddings_cache.json'
try:
    with open(cache_file, 'r') as f:
        embeddings_cache = json.load(f)
except FileNotFoundError:
    embeddings_cache = {}


#serialize complex metadata fields
def serialize_metadata(metadata):
    for key, value in metadata.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            metadata[key] = json.dumps(value)  
        elif isinstance(value, list):
            metadata[key] = ', '.join(map(str, value))  
        else:
            metadata[key] = str(value) 
    return metadata


# batch process text data to generate embeddings and store in Pinecone
batch_size = 32
for i in range(0, len(text_data), batch_size):
    batch_texts = text_data[i:i + batch_size]
    batch_keys = [str(j) for j in range(i, i + len(batch_texts))]

    # Check cache before processing
    new_texts = []
    new_keys = []
    for key, text in zip(batch_keys, batch_texts):
        if key not in embeddings_cache:
            new_texts.append(text)
            new_keys.append(key)

    if new_texts:
        embeddings = generate_embeddings(new_texts)
        for key, emb in zip(new_keys, embeddings):
            embeddings_cache[key] = emb  #update cache

    with open(cache_file, 'w') as f:
        json.dump(embeddings_cache, f)

    upsert_data = [(key, embeddings_cache[key], serialize_metadata(original_json_data[int(key)])) for key in batch_keys]
    for id, values, metadata in upsert_data:
        index.upsert(vectors=[(id, values, metadata)])  #make sure each upsert call has structured data

print("Data vectorized and stored in Pinecone successfully.")
