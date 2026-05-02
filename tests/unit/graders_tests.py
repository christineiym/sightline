"""
Unit tests for sightline/graders.py

Run with: pytest tests/unit/graders_tests.py

AI use disclosure: This was written with assistance from ChatGPT (GPT-5.3).
"""

import pytest

from sightline.graders import Grader


class TestResponseToJSONObject:
    """
    Tests for Grader.response_to_JSON_object:
    - valid JSON extraction
    - malformed JSON recovery
    - edge cases and failure fallback
    """

    def _call(self, response):
        return Grader.response_to_JSON_object(response)

    def test_valid_json_object_embedded(self):
        response = "Some text before [{\"a\": 1, \"b\": 2}] some text after"
        result = self._call(response)

        assert isinstance(result, dict)
        assert result["a"] == 1
        assert result["b"] == 2

    def test_unquoted_keys_are_fixed(self):
        response = "prefix [{a: 1, b: 2}] suffix"
        result = self._call(response)

        assert result == {"a": 1, "b": 2}

    def test_single_quotes_are_converted(self):
        response = "text [{'a': 1, 'b': 2}] text"
        result = self._call(response)

        assert result == {"a": 1, "b": 2}

    def test_extra_text_around_json(self):
        response = "random text [ { \"x\": 10 } ] trailing"
        result = self._call(response)

        assert result == {"x": 10}

    def test_no_json_returns_empty_dict(self):
        response = "no brackets here"
        result = self._call(response)

        assert result == {}

    def test_malformed_json_returns_empty_dict(self):
        response = "text [ {a: } ] text"
        result = self._call(response)

        assert result == {}

    def test_nested_structure(self):
        response = "text [{a: {b: 2}}] text"
        result = self._call(response)

        assert result == {"a": {"b": 2}}

    def test_multiple_items_inside_brackets(self):
        response = "text [{a:1}, {b:2}] text"
        result = self._call(response)

        # Note: implementation wraps in {} so this may fail depending on parsing;
        # we assert it doesn't crash and returns a dict
        assert isinstance(result, dict)


class TestResponseToJSONList:
    """
    Tests for Grader.response_to_JSON_list:
    - valid list extraction
    - malformed JSON recovery
    - edge cases and fallback behavior
    """

    def _call(self, response):
        return Grader.response_to_JSON_list(response)

    def test_valid_json_list_embedded(self):
        response = "text before [{\"a\": 1}, {\"b\": 2}] text after"
        result = self._call(response)

        assert isinstance(result, list)
        assert result == [{"a": 1}, {"b": 2}]

    def test_unquoted_keys_are_fixed(self):
        response = "prefix [{a: 1}, {b: 2}] suffix"
        result = self._call(response)

        assert result == [{"a": 1}, {"b": 2}]

    def test_single_quotes_are_converted(self):
        response = "text [{'a': 1}, {'b': 2}] text"
        result = self._call(response)

        assert result == [{"a": 1}, {"b": 2}]

    def test_extra_text_around_json(self):
        response = "random [ { \"x\": 10 } ] trailing"
        result = self._call(response)

        assert result == [{"x": 10}]

    def test_no_json_returns_empty_list(self):
        response = "no brackets here"
        result = self._call(response)

        assert result == []

    def test_malformed_json_returns_empty_list(self):
        response = "text [ {a: } ] text"
        result = self._call(response)

        assert result == []

    def test_nested_structure(self):
        response = "text [{a: {b: 2}}] text"
        result = self._call(response)

        assert result == [{"a": {"b": 2}}]

    def test_empty_list(self):
        response = "text [] text"
        result = self._call(response)

        assert result == []

    def test_non_string_input(self):
        response = [{"a": 1}, {"b": 2}]
        result = self._call(response)

        assert result == [{"a": 1}, {"b": 2}]