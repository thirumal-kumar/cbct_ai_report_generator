# llm_client.py
import os
import requests
import socket
from typing import List, Dict

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "gpt-4o-mini")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def can_resolve(hostname="openrouter.ai") -> bool:
    try:
        socket.gethostbyname(hostname)
        return True
    except Exception:
        return False

def call_openrouter(messages: List[Dict], timeout=60) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY missing")
    if not can_resolve("openrouter.ai"):
        raise RuntimeError("DNS resolution failed for openrouter.ai")
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 800
    }
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    r = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"]
