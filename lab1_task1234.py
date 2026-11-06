from openai import OpenAI
import os
from dotenv import load_dotenv
import time
load_dotenv()
def call_openai(
    prompt : str,
    model : str,
    temperature : float = 0.7,
    top_p : float = 0.9,
    max_tokens : int = 256
)-> tuple[str, float, dict]:
   
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    start = time.time()
    res = client.chat.completions.create(
        model = model,
        messages=[{"role":"user","content":prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens = max_tokens
    )
    latency = time.time() - start
    answer = res.choices[0].message.content
    usage = {"input_tokens" : res.usage.prompt_tokens,
    "output_tokens" : res.usage.prompt_tokens}
    return answer, latency,usage
# answer, latency, input, output = call_openai(
#     prompt="Explain what an embedded system is",
#     model="gpt-4.1-mini",
#     temperature=0.7,
#     top_p=0.9,
#     max_tokens=256
# )
# print("Answer:", answer)
# print("Latency:", latency)
# print("input:", input)
# print("output:", output)
from google import genai
from google.genai import types
def call_gemini(
    prompt : str,
    model : str,
    temperature : float = 0.7,
    top_p : float = 0.9,
    max_tokens : int = 256
) -> tuple[str, float, dict] :
    start = time.time()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
    temperature=temperature,
    top_p=top_p,
    max_output_tokens=max_tokens
)
    res = client.models.generate_content(
        model = model,
        contents = prompt,
        config = config
    )
    latency = time.time() - start
    usage = {
        "input_tokens" : res.usage_metadata.prompt_token_count,
        "output_tokens" : res.usage_metadata.candidates_token_count
    }
    answer = res.text
    return answer, latency, usage
answer, latency, usage = call_gemini(
    prompt="Explain what an embedded system is",
    model="gemini-3.8-flash",
    temperature=0.7,
    top_p=0.9,
    max_tokens=256
)
print("Answer:", answer)
print("Latency:", latency)
print("usage", usage)

import anthropic
def call_anthropic(
    prompt: str,
    model: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256
) -> tuple[str, float, dict]:
    client = anthropic.Anthropic(
        api_key=os.getenv("CLAUDE_API_KEY")
    )

    start = time.time()

    res = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    latency = time.time() - start
    answer = "".join(
        block.text
        for block in res.content
        if block.type == "text"
    )
    usage = {
        "input_tokens": res.usage.input_tokens,
        "output_tokens": res.usage.output_tokens,
    
    }

    return answer, latency, usage
#bonus C : table comparison
import pandas as pd
def compare_models(
    prompt: str,
    openai_model: str,
    gemini_model: str,
    anthropic_model: str,
) -> pd.DataFrame:
    models = [
        ("OpenAI", openai_model, call_openai),
        ("Gemini", gemini_model, call_gemini),
        ("Anthropic", anthropic_model, call_anthropic)
    ]
    results = []
    for provider, model, api in models :
        try:
            answer, latency, usage = api(
                prompt=prompt,
                model=model,
                temperature=0.7,
                top_p=0.9,
                max_tokens=256
            )

            results.append({
                "Provider": provider,
                "Model": model,
                "Answer": answer,
                "Latency": round(latency, 3),
                "Input Tokens": usage["input_tokens"],
                "Output Tokens": usage["output_tokens"],
                "Total Tokens": usage.get("total_tokens", 0),
                "Status": "Success"
            })

        except Exception as e:
            results.append({
                "Provider": provider,
                "Model": model,
                "Answer": None,
                "Latency": None,
                "Input Tokens": None,
                "Output Tokens": None,
                "Total Tokens": None,
                "Status": f"Error: {str(e)}"
            })

    return pd.DataFrame(results)
    