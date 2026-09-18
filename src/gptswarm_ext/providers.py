from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    model: str
    cached: bool = False


class SQLiteCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS responses "
            "(cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self.connection.commit()

    @staticmethod
    def key(payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def get(self, key: str) -> Generation | None:
        row = self.connection.execute(
            "SELECT payload FROM responses WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        generation = Generation(**json.loads(row[0]))
        generation.cached = True
        return generation

    def put(self, key: str, generation: Generation) -> None:
        payload = asdict(generation)
        payload["cached"] = False
        self.connection.execute(
            "INSERT OR REPLACE INTO responses(cache_key, payload) VALUES (?, ?)",
            (key, json.dumps(payload, sort_keys=True)),
        )
        self.connection.commit()


class BaseProvider:
    def generate(self, messages: list[dict[str, str]]) -> Generation:
        raise NotImplementedError


class MockProvider(BaseProvider):
    def __init__(self, model: str = "deterministic-mock-v1"):
        self.model = model

    def generate(self, messages: list[dict[str, str]]) -> Generation:
        prompt = "\n".join(message["content"] for message in messages)
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        answer_index = digest[0] % 4
        if "Answer a lie" in prompt:
            answer_index = (answer_index + 1) % 4
        text = "ABCD"[answer_index]
        return Generation(
            text=text,
            input_tokens=max(1, len(prompt.split())),
            output_tokens=1,
            latency_seconds=0.0,
            model=self.model,
        )


class OllamaProvider(BaseProvider):
    def __init__(
        self,
        model: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
        seed: int,
        timeout_seconds: int,
        cache: SQLiteCache | None = None,
    ):
        self.model = model
        self.url = f"{base_url.rstrip('/')}/api/chat"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.timeout_seconds = timeout_seconds
        self.cache = cache

    def generate(self, messages: list[dict[str, str]]) -> Generation:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "seed": self.seed,
            },
        }
        cache_key = SQLiteCache.key(payload)
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        started = time.perf_counter()
        response = requests.post(self.url, json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        body = response.json()
        generation = Generation(
            text=body.get("message", {}).get("content", ""),
            input_tokens=int(body.get("prompt_eval_count", 0)),
            output_tokens=int(body.get("eval_count", 0)),
            latency_seconds=time.perf_counter() - started,
            model=body.get("model", self.model),
        )
        if self.cache is not None:
            self.cache.put(cache_key, generation)
        return generation


def build_provider(config: dict[str, Any]) -> BaseProvider:
    provider = config["provider"].lower()
    if provider == "mock":
        return MockProvider(config["model"])
    if provider == "ollama":
        cache = None
        if config.get("cache_enabled", False):
            cache = SQLiteCache(Path(config["cache_path"]))
        return OllamaProvider(
            model=config["model"],
            base_url=config.get("base_url", "http://127.0.0.1:11434"),
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config.get("max_tokens", 4)),
            seed=int(config.get("seed", 0)),
            timeout_seconds=int(config.get("timeout_seconds", 120)),
            cache=cache,
        )
    raise ValueError(f"Unsupported provider: {provider}")
