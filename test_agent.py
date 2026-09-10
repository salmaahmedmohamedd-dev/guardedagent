"""
Tests for agent.py.

FIX: the previous version was a script of bare `assert`s with no `test_`
functions — pytest would collect zero tests from it and report success
even if nothing ran. Also, nothing here previously exercised run_agent()
or choose_tool()'s routing, which is exactly where the NameError bug and
the misrouting bug were hiding. Those paths are covered now.
"""

import pytest
from agent import (
    calculator,
    validate_tool_request,
    use_tool,
    choose_tool,
    run_agent,
)


# --- calculator ---

def test_calculator_valid_expression():
    assert calculator("25 * 17") == 425


def test_calculator_invalid_characters():
    assert calculator("25 + abc") == "Invalid expression"


def test_calculator_rejects_code_injection_attempt():
    # ast-based evaluator should reject anything that isn't pure arithmetic
    assert calculator("__import__('os').system('echo hi')") == "Invalid expression"


# --- validate_tool_request (guardrail) ---

def test_validate_allowed_tool():
    assert validate_tool_request({
        "tool": "calculator",
        "input": "25 * 17"
    }) is True


def test_validate_rejects_unauthorized_tool():
    assert validate_tool_request({
        "tool": "delete_files",
        "input": "important.txt"
    }) is False


def test_validate_rejects_missing_input():
    assert validate_tool_request({
        "tool": "calculator"
    }) is False


def test_validate_rejects_non_dict():
    assert validate_tool_request("not a dict") is False


def test_validate_rejects_empty_input():
    assert validate_tool_request({
        "tool": "calculator",
        "input": "   "
    }) is False


# --- choose_tool (routing heuristic) ---

def test_choose_tool_routes_pure_math_to_calculator():
    request = choose_tool("25 * 17")
    assert request["tool"] == "calculator"


def test_choose_tool_routes_question_with_hyphen_to_search():
    # Regression test: this used to route to the calculator because of the
    # hyphen in "state-of-the-art", fail there, and never fall back.
    request = choose_tool("What's the state-of-the-art model?")
    assert request["tool"] == "web_search"


def test_choose_tool_routes_plain_question_to_search():
    request = choose_tool("What is Docker?")
    assert request["tool"] == "web_search"


# --- run_agent (previously untested — this is where the NameError lived) ---

def test_run_agent_normal_path_does_not_crash():
    run_agent("What's the state-of-the-art model?")  # should not raise



    run_agent("   ")  # should not raise
    captured = capsys.readouterr()
    assert "rejected" in captured.out.lower()


def test_use_tool_rejects_disallowed_tool():
    assert use_tool("delete_files", "important.txt") == "Tool not allowed"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))