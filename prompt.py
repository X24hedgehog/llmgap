#!/usr/bin/env python3
"""Shared prompt builders used across the EEDI judge tooling and pipeline."""

from __future__ import annotations

import ast


def build_answer_equivalence_prompt(problem_context: str, answer_a: str, answer_b: str) -> str:
    return (
        "You are judging whether two answer choices for a middle-school math "
        "multiple-choice question are semantically equivalent. Do not solve the "
        "problem and do not evaluate factual correctness. Only compare whether "
        "the two answers express the same answer choice relative to the problem's "
        "formatting requirements. Think step by step, then end with a final line "
        "that is exactly 'Yes' or exactly 'No'. "
        "Treat harmless wrappers such as 'Answer:' and equivalent formatting such as "
        "LaTeX delimiters as the same answer when the meaning is unchanged.\n\n"
        "The 2 answers can be the correct or incorrect answer to the question, "
        "but as long as they are equivalent, you should say Yes; otherwise, say No.\n\n"
        f"Problem context:\n{problem_context}\n\n"
        f"Answer 1: {answer_a}\n"
        f"Answer 2: {answer_b}\n\n"
        "Reasoning:"
    )


def parse_first_distractor_answer(raw_value: str) -> str:
    parsed = ast.literal_eval(raw_value)
    if isinstance(parsed, list):
        return str(parsed[0])
    return str(parsed)


def build_eedi_correct_answer_trace_prompt(question: str, target_answer: str) -> str:
    return (
        "You are writing a gold reasoning trace for a math tutoring dataset. "
        "You will be given: a math question; the known correct answer. "
        "Your job: produce a concise but clear step-by-step reasoning trace that "
        "solves the question; use the known correct answer as ground truth; end "
        "with a final sentence that states the correct answer exactly; return only "
        "the reasoning trace.\n\n"
        f"Math question:\n{question}\n\n"
        f"Known correct answer: {target_answer}"
    )


def build_eedi_distractor_trace_prompt(
    question: str,
    correct_answer: str,
    target_incorrect_answer: str,
    misconception_name: str,
) -> str:
    return (
        "You are writing a gold reasoning trace for a student-misconception dataset. "
        "You will be given: a math question; the known correct answer; a target "
        "incorrect answer; the student's misconception. Your job: produce a concise "
        "but clear step-by-step incorrect reasoning trace that a student with this "
        "misconception could plausibly follow; the reasoning should be coherent with "
        "the misconception; it must end at the given target incorrect answer and not "
        "at the correct answer; return only the reasoning trace.\n\n"
        f"Math question:\n{question}\n\n"
        f"Known correct answer: {correct_answer}\n"
        f"Target incorrect answer: {target_incorrect_answer}\n"
        f"Student misconception: {misconception_name}"
    )


__all__ = [
    "build_answer_equivalence_prompt",
    "build_eedi_correct_answer_trace_prompt",
    "build_eedi_distractor_trace_prompt",
    "parse_first_distractor_answer",
]