from __future__ import annotations

import pytest

from idea2hypothesis.llm.parsing import extract_json_object, strip_thinking_tags


def test_baseline_regression_extra_data_between_objects() -> None:
    """Carried over from the baseline `test_parse_robust` case."""
    text = (
        '{\n  "shortlist": [{"id": "p1"}]\n}\n{"rejected": [{"id": "rx1"}]}\n'
        "Note: Extra data line 132 column 1 (char 7784)"
    )
    result = extract_json_object(text)
    assert set(result) == {"shortlist", "rejected"}
    assert result["shortlist"] == [{"id": "p1"}]


def test_plain_object() -> None:
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_fenced_object_with_language_tag() -> None:
    assert extract_json_object('```json\n{"a": {"b": [1, 2]}}\n```') == {"a": {"b": [1, 2]}}


def test_preamble_and_postscript() -> None:
    assert extract_json_object('Sure! Here you go:\n{"ok": true}\nHope that helps.') == {"ok": True}


def test_fence_with_commentary_after_it() -> None:
    text = '```json\n{"a": 1}\n```\nLet me know if you need changes.'
    assert extract_json_object(text) == {"a": 1}


def test_later_objects_override_earlier_keys() -> None:
    assert extract_json_object('{"a": 1, "b": 1} {"b": 2}') == {"a": 1, "b": 2}


def test_arrays_alone_are_not_objects() -> None:
    with pytest.raises(ValueError):
        extract_json_object("[1, 2, 3]")


@pytest.mark.parametrize("text", ["", "no json here", "{broken", "```json\n```"])
def test_invalid_output_raises_value_error(text: str) -> None:
    with pytest.raises(ValueError, match="Could not extract valid JSON"):
        extract_json_object(text)


def test_reasoning_tags_are_removed_before_parsing() -> None:
    text = '<think>I should output {"a": 99}</think>{"a": 1}'
    assert extract_json_object(text) == {"a": 1}


def test_strip_thinking_tags() -> None:
    assert strip_thinking_tags("<think>x</think>answer") == "answer"
    assert strip_thinking_tags("<think>never closed") == ""
    assert strip_thinking_tags("[thinking] hmm\n\nanswer") == "answer"


def test_clean_text_is_returned_unchanged() -> None:
    text = "def f():\n    pass\n\n\ndef g():\n    pass\n"
    assert strip_thinking_tags(text) == text
    assert strip_thinking_tags("") == ""
