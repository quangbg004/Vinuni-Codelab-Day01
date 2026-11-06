from openai import OpenAI
import os
from dotenv import load_dotenv
import time
from google import genai
from google.genai import types
load_dotenv()
def streaming_model(
    model : str,
    formatted_history : str
) : 

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
    )
    response_stream = client.models.generate_content_stream(
        model=model,
        contents=formatted_history
    )
    for chunk in response_stream:
        print(chunk.text, end="", flush=True)
formatted_history = "Explain how UART communication works."
model="gemini-3.8-flash"
#model="gemini-3.1-pro"
#model="gemini-3.5-flash-lite"
streaming_model(
    model,
    formatted_history=formatted_history
)