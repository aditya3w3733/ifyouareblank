# ifyouareblank

# IfYouAreBlank: RAG-Based Movie Recommendation System
## Introduction
IfYouAreBlank is a sophisticated movie recommendation system that leverages a combination of OpenAI's GPT-3.5 Turbo, text-embedding-3-small model, and Pinecone's vector database to deliver narrative-driven movie suggestions. Inspired by queries from Reddit's r/MovieSuggestions, this system allows users to input detailed descriptions of their movie preferences, and utilizes a RAG system to generate tailored movie recommendations.


## Features
- Narrative-Driven Queries: Accepts user input in a freeform narrative style to capture detailed preferences.
- Advanced NLP Models: Utilizes Open-AI's GPT-3.5 Turbo for understanding and processing user queries.
- Vector Database: Employs Pinecone to manage and retrieve movie data efficiently through vector embeddings.
- Dynamic Interaction: Allows iterative query refinement to hone in on user preferences for more accurate recommendations.


## Results 

### User Query: 

I'm looking for movies that generally mess your mind and/or make you think who even thought of this?!? Movies like Primer, Enter the Void, Moon, Donnie Darko, Being John Malkovich, The Lobster, Cube, Midsommar, etc.

### Refined Movie Recommendations:

1. Solaris (2002)
2. Primer (2004)
3. Upstream Color (2013)
4. Waking Life (2001)
5. Eraserhead (1977)
6. Twelve Monkeys (1995)
7. Enter the Void (2009)
8. Donnie Darko (2001)
9. Memento (2000)
10. Predestination (2014)
11. Mulholland Dr. (2001)
12. Synecdoche, New York (2008)
13. eXistenZ (1999)
14. The Machinist (2004)
15. The Prestige (2006)


### Structured Query:
 
Positive Movies: Primer, Enter the Void, Moon, Donnie Darko, Being John Malkovich, The Lobster, Cube, Midsommar<br>

Negative Movies: None<br>

Positive Keywords: mess your mind, make you think, thought of this<br>

Negative Keywords: None<br>

Positive Genres: Psychological Thriller, Mind-Bending<br>

Negative Genres: None<br>

Positive Actors: None<br>

Negative Actors: None<br>

![image](https://github.com/aditya3w3733/ifyouareblank/assets/104208359/bcdec639-3f98-44b2-a221-b18df1c9b227)


## Future Enhancements
- This project can be made into a web app for better accessibility.
