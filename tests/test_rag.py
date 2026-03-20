"""Tests for app.rag module."""

from app.rag import build_prompt


def test_build_prompt_with_chunks():
    result = build_prompt("What is AI?", ["chunk one", "chunk two"])
    assert "What is AI?" in result
    assert "[1] chunk one" in result
    assert "[2] chunk two" in result


def test_build_prompt_empty_chunks():
    result = build_prompt("What is AI?", [])
    assert result == "What is AI?"
