"""Small app on a vLLM server: streaming chat + JSON-constrained output.

Start the server first:
  vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000
Then: python vllm_app.py        (pip install openai)
"""
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def stream_chat(question):
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
        max_tokens=100,
        stream=True,
    )
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
    print()


def structured(question):
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
        "required": ["city", "country"],
    }
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
        # vLLM structured outputs through the OpenAI response_format field
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "place", "schema": schema},
        },
    )
    return json.loads(resp.choices[0].message.content)


if __name__ == "__main__":
    stream_chat("Say hello in one sentence.")
    print(structured("Give me the capital of France."))
