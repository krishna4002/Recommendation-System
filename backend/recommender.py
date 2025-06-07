# backend/recommender.py

import os
from backend.embeddings import embed_text
from backend.vector_store import index
from backend.user_profile import build_user_vector
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# OpenRouter client setup (Mistral model)
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

import numpy as np

def personalized_search(query, user_id):
    query_vec = embed_text(query)
    user_vec = build_user_vector(user_id)

    if user_vec is not None:
        final_vector = np.mean([query_vec, user_vec], axis=0).tolist()
    else:
        final_vector = query_vec.tolist() if hasattr(query_vec, "tolist") else query_vec

    results = index.query(vector=final_vector, top_k=10, include_metadata=True)

    # Remove duplicate titles
    seen = set()
    unique_matches = []
    for item in results.matches:
        title = item.metadata['title']
        if title not in seen:
            seen.add(title)
            unique_matches.append(item)
    return unique_matches


def rerank_results(query, items):
    prompt = f"""
    The user is looking for books similar to this query: "{query}"

    Only include books that are clearly relevant.
    Avoid duplicates or near-identical self-help books.

    Options:
    {chr(10).join(f"- {item.metadata['title']}: {item.metadata['description']}" for item in items)}

    Return a ranked list of book titles only:
    """
    
    response = client.chat.completions.create(
        model="mistralai/devstral-small:free",
        messages=[
            {"role": "system", "content": "You rerank books for recommendations."},
            {"role": "user", "content": prompt}
        ]
    )

    # Parse clean list of titles
    raw_output = response.choices[0].message.content.strip()

    ranked_titles = []
    for line in raw_output.splitlines():
        if "-" in line:
            title = line.split("-", 1)[-1].strip()
        elif "." in line:
            title = line.split(".", 1)[-1].strip()
        else:
            title = line.strip()

        if title:
            ranked_titles.append(title)

    return ranked_titles