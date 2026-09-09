#checking parts from the agent we built still work
from agent import calculator, validate_tool_request, use_tool


# Calculator test
assert calculator("25 * 17") == 425


# Invalid calculator input
assert calculator("25 + abc") == "Invalid expression"


# Allowed tool test
assert validate_tool_request({
    "tool": "calculator",
    "input": "25 * 17"
}) is True


# Unauthorized tool test
assert validate_tool_request({
    "tool": "delete_files",
    "input": "important.txt"
}) is False


# Missing input test
assert validate_tool_request({
    "tool": "calculator"
}) is False


print("All tests passed!")