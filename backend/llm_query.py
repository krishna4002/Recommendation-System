# backend/llm_query.py

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Create OpenAI client for OpenRouter
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"  # OpenRouter endpoint
)

def parse_query_with_mistral(user_query: str) -> str:
    response = client.chat.completions.create(
        model="mistralai/devstral-small:free",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Rewrite the user query to improve book recommendations. Keep it short and focused."
            },
            {
                "role": "user",
                "content": user_query
            }
        ]
    )

    return response.choices[0].message.content.strip()