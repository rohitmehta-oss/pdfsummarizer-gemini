#!/usr/bin/env python3
"""Phase 0 preflight: validate environment, dependencies, and Gemini access."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

REQUIRED_MODULES = (
    "streamlit",
    "langchain_core",
    "langchain_google_genai",
    "pypdf",
    "dotenv",
    "pandas",
    "requests",
    "tiktoken",
)


def fail(message: str) -> None:
    print(f"FAIL  {message}")


def ok(message: str) -> None:
    print(f"OK    {message}")


def clean(error: BaseException, secret: str | None = None) -> str:
    text = f"{type(error).__name__}: {error}"
    if secret:
        text = text.replace(secret, "***")
    return text


def check_python() -> bool:
    ok(f"Python {sys.version.split()[0]}")
    return True


def check_env() -> tuple[str, str] | None:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip()

    if not api_key:
        fail("GEMINI_API_KEY is missing (copy .env.example to .env)")
        return None
    if len(api_key) < 20:
        fail("GEMINI_API_KEY looks incomplete")
        return None

    ok(f"GEMINI_API_KEY is set ({len(api_key)} characters)")
    ok(f"GEMINI_MODEL={model}")
    return api_key, model


def check_imports() -> bool:
    missing = []
    for name in REQUIRED_MODULES:
        try:
            __import__(name)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{name} ({clean(exc)})")

    if missing:
        fail("Missing or broken dependencies: " + "; ".join(missing))
        return False

    ok("All runtime dependencies import successfully")
    return True


def list_models(api_key: str) -> list[str]:
    response = requests.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": api_key},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    return [
        str(item.get("name", "")).removeprefix("models/")
        for item in payload.get("models", [])
    ]


def check_model_access(api_key: str, model: str) -> bool:
    try:
        models = list_models(api_key)
    except Exception as exc:  # noqa: BLE001
        fail(f"Gemini model list failed: {clean(exc, api_key)}")
        return False

    if model not in models:
        fail(f"Model {model!r} is not available to this key")
        return False

    ok(f"Gemini model is available ({len(models)} models visible)")
    return True


def check_generation(api_key: str, model: str) -> bool:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            google_api_key=api_key,
        )
        message = llm.invoke("Reply with exactly: OK")
        text = getattr(message, "text", None) or str(message)
    except Exception as exc:  # noqa: BLE001
        fail(f"Gemini generation failed: {clean(exc, api_key)}")
        return False

    if "OK" not in text.upper():
        fail(f"Gemini replied unexpectedly: {text[:120]!r}")
        return False

    ok("Gemini generation works (received OK)")
    return True


def main() -> int:
    print(f"Project root: {ROOT}\n")

    checks = [check_python()]
    env = check_env()
    checks.append(env is not None)
    checks.append(check_imports())

    if env:
        api_key, model = env
        checks.append(check_model_access(api_key, model))
        checks.append(check_generation(api_key, model))

    if all(checks):
        print("\nPhase 0 validation passed.")
        return 0

    print("\nPhase 0 validation failed. Fix the items above and rerun.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
