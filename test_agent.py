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
    parse_tool_call,
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


# --- parse_tool_call (guardrail) ---

def test_parse_allows_valid_tool():
    assert parse_tool_call('{"tool": "calculator", "input": "25 * 17"}') == {
        "tool": "calculator", "input": "25 * 17"
    }


def test_parse_rejects_unauthorized_tool():
    assert parse_tool_call('{"tool": "delete_files", "input": "important.txt"}') is None


def test_parse_rejects_missing_input():
    assert parse_tool_call('{"tool": "calculator"}') is None


def test_parse_rejects_malformed_json():
    assert parse_tool_call("not json at all") is None


def test_parse_rejects_empty_input():
    assert parse_tool_call('{"tool": "calculator", "input": "   "}') is None


def test_parse_rejects_injection_in_calculator_input():
    raw = '{"tool": "calculator", "input": "__import__(\'os\').system(\'ls\')"}'
    assert parse_tool_call(raw) is None


def test_parse_allows_none_decision():
    assert parse_tool_call('{"tool": "none", "input": ""}') == {"tool": "none", "input": ""}

# --- choose_tool (routing heuristic) ---
@pytest.mark.integration
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


def test_run_agent_handles_empty_input_without_crashing(capsys):
    run_agent("   ")
    captured = capsys.readouterr()
    assert captured.out.strip() != ""

def test_use_tool_rejects_disallowed_tool():
    assert use_tool("delete_files", "important.txt") == "Tool not allowed"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
