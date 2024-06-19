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

def load_system_message(file_path='original_system_message.txt'):
    with open(file_path, 'r') as file:
        return file.read()

def get_structured_query(system_message, user_input):
        prompt_text = f"""{system_message}User has described their interest in movies as follows: "{user_input}" First, list all movie titles mentioned in the user query.Then, based on these titles and any additional descriptive content, classify the user's sentiments toward these movies and extract relevant genres, keywords, and actors."""
#User says: {user_input}\n\nIdentify any mentioned movie titles and classify the input into positive and negative preferences. Extract genres, actors, and specific themes if mentioned.")
    conversation = [
        {"role": "system", "content": prompt_text},
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
        movie_list = ', '.join([rec.split(' (')[0] for rec in recommendations])  #extract movie titles

        refinement_prompt = f"User expressed interest in: {user_input}.Based on given users preferences and the themes reflected in their query, please refine the following list of movie recommendations(don't include anything not on this list) and suggest the top 15 best fits with their respective year: {movie_list}"

        refined_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": refinement_prompt}
            ],
            max_tokens=350
        )

        refined_recommendations = refined_response.choices[0].message['content'].strip()
        print("Movie Recommendations:\n")
        #print(structured_query)
        print(refined_recommendations)
        #print(recommendations)
    else:
        print("No recommendations found.")

if __name__ == '__main__':
    main()
