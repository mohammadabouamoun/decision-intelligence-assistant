import os
import time
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = "llama-3.3-70b-versatile"

def call_llm(prompt: str, system_prompt: str = "You are a helpful assistant."):
    start = time.time()
    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            model=MODEL_NAME,
            temperature=0.5,
            max_tokens=300
        )
        latency_ms = (time.time() - start) * 1000
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost_usd = (input_tokens + output_tokens) * 0.000001
        return completion.choices[0].message.content, latency_ms, cost_usd
    except Exception as e:
        return f"Error: {e}", 0, 0

def rag_answer(question: str, context_chunks: list):
    context = "\n\n".join([chunk['text'] for chunk in context_chunks])
    prompt = f"""Use the following support ticket excerpts to answer the user's question. If the answer cannot be found, say so.

Context:
{context}

Question: {question}

Answer:"""
    return call_llm(prompt, system_prompt="You are a customer support AI.")

def non_rag_answer(question: str):
    prompt = f"Answer the following customer support question: {question}"
    return call_llm(prompt, system_prompt="You are a customer support AI.")

def zero_shot_priority(question: str):
    prompt = f"""Classify the urgency of this customer support message. Respond with ONLY a JSON object like {{"label": "urgent" or "normal", "confidence": 0.0-1.0}}.
Message: {question}"""
    response, latency, cost = call_llm(prompt, system_prompt="You are a classification assistant.")
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            label = data.get("label", "normal")
            confidence = float(data.get("confidence", 0.5))
        else:
            label = "normal"
            confidence = 0.5
    except Exception:
        label = "normal"
        confidence = 0.5
    return response, latency, cost, {"label": label, "confidence": confidence}