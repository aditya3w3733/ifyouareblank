import openai
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import json
import os

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

pinecone = Pinecone(api_key=pinecone_api_key)
index_name = 'submission-index'

if index_name not in pinecone.list_indexes().names():
    spec = ServerlessSpec(cloud='aws', region='us-east-1')
    pinecone.create_index(name=index_name, dimension=1536, metric='cosine', spec=spec)

index = pinecone.Index(name=index_name)

def load_system_message(file_path='system_message.txt'):
    with open(file_path, 'r') as file:
        return file.read()

def get_structured_query(system_message, user_input):
    conversation = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_input}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation
    )
    return response.choices[0].message['content'].strip()


#function to generate embeddings
def generate_embeddings(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response['data'][0]['embedding']

#function to search in Pinecone
def search_in_pinecone(embedding, top_k=5):
    return index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )

def extract_recommendations(matches):
    movie_frequency = {}
    for match in matches:
        metadata = match['metadata']  # assuming metadata is a dictionary
        comments = json.loads(metadata['comments'])  # assuming comments is stored as a serialized JSON string
        for comment in comments:
            movies = comment['movie_ids']
            for movie in movies:
                if movie in movie_frequency:
                    movie_frequency[movie] += 1
                else:
                    movie_frequency[movie] = 1

    # sorting movies by frequency
    sorted_movies = sorted(movie_frequency.items(), key=lambda x: x[1], reverse=True)

    #calculate frequency
    recommendations = []
    for movie, count in sorted_movies:
        recommendations.append(f"{movie} (recommended {count} times)")
    return recommendations


def main():
    system_message = load_system_message()
    user_input = input("Please describe the type of movies you're interested in: ")

    structured_query = get_structured_query(system_message, user_input)
    embedding = generate_embeddings(structured_query)
    results = search_in_pinecone(embedding)

    matches = results.get('matches', [])
    if matches:
        recommendations = extract_recommendations(matches)
        print("Movie Recommendations:")
        print(structured_query)
        for recommendation in recommendations:
            print(recommendation)
    else:
        print("No recommendations found.")

if __name__ == '__main__':
    main()
