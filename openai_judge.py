#!/usr/bin/env python3
"""Shared OpenAI judge helpers for EEDI-only semantic scoring."""

from __future__ import annotations

import os
import re
from pathlib import Path

from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The openai package is required. Install it with: pip install openai"
    ) from exc


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_API_KEY_FILE = ROOT_DIR / "eedi" / "openai_api.txt"
DEFAULT_OPENAI_EEDI_JUDGE_MODEL = os.environ.get(
    "OPENAI_EEDI_JUDGE_MODEL",
    "gpt-4.1-mini",
)

_OPENAI_JUDGE_CLIENT: OpenAI | None = None


def _load_api_key() -> str:
    if DEFAULT_API_KEY_FILE.exists():
        return DEFAULT_API_KEY_FILE.read_text(encoding="utf-8").strip()

    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    raise SystemExit(
        "No OpenAI API key found for EEDI judging. Put your key in "
        "eedi/openai_api.txt or set OPENAI_API_KEY."
    )


def get_openai_judge_client() -> OpenAI:
    global _OPENAI_JUDGE_CLIENT
    if _OPENAI_JUDGE_CLIENT is None:
        _OPENAI_JUDGE_CLIENT = OpenAI(api_key=_load_api_key())
    return _OPENAI_JUDGE_CLIENT


def parse_yes_no_judgment(text: str) -> int:
    stripped = text.strip()
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if lines:
        last_line = lines[-1].rstrip(".!")
        if last_line.lower() == "yes":
            return 1
        if last_line.lower() == "no":
            return 0

    verdicts = re.findall(r"\b(yes|no)\b", stripped, flags=re.IGNORECASE)
    if verdicts:
        return 1 if verdicts[-1].lower() == "yes" else 0

    lowered = stripped.lower()
    if "not semantically equivalent" in lowered or "are not equivalent" in lowered:
        return 0
    if "semantically equivalent" in lowered or "are equivalent" in lowered:
        return 1
    return 0


def judge_yes_no_openai(prompts, batch_size=8, model=DEFAULT_OPENAI_EEDI_JUDGE_MODEL):
    client = get_openai_judge_client()
    scores = []
    developer_message = (
        "You are a careful math evaluation judge. Follow the user instructions "
        "exactly. You may reason briefly, but the final line must be only Yes or No."
    )

    for start in tqdm(range(0, len(prompts), batch_size), desc="OpenAI judge"):
        batch = prompts[start : start + batch_size]
        for prompt in batch:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "developer", "content": developer_message},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_completion_tokens=256,
            )
            text = (response.choices[0].message.content or "").strip()
            scores.append(parse_yes_no_judgment(text))

    return scores


__all__ = [
    "DEFAULT_OPENAI_EEDI_JUDGE_MODEL",
    "judge_yes_no_openai",
    "parse_yes_no_judgment",
]