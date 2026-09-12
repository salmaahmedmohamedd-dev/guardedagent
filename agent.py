import ast   # abstract syntax tree - safer than eval(), parses text into a structure we can inspect
import operator   # gives python functions for math operations
import logging
import os
import json

from openai import OpenAI
from tavily import TavilyClient
from dotenv import load_dotenv   # reads variables from .env so keys are never hardcoded

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"   # ignored by ollama, but the library requires a value
)

MODEL = "qwen2.5:7b"


# ============================================================
# TOOLS
# ============================================================

def calculator(expression: str):
    # guardrail 1: character allowlist
    allowed = "0123456789+-*/(). "
    if not all(char in allowed for char in expression):
        return "Invalid expression"

    try:
        # mode="eval" means the input is treated as a single expression
        tree = ast.parse(expression, mode="eval")

        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
        }

        # guardrail 2: walk the tree and only allow known node types
        def evaluate(node):
            if isinstance(node, ast.Expression):
                return evaluate(node.body)

            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value

            if isinstance(node, ast.BinOp) and type(node.op) in operators:
                left = evaluate(node.left)
                right = evaluate(node.right)
                return operators[type(node.op)](left, right)

            # handles negative numbers
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                return -evaluate(node.operand)

            raise ValueError("Invalid expression")

        return evaluate(tree)

    except Exception:
        return "Invalid expression"


def web_search(query: str):
    if not query.strip():
        return "Invalid search query"

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        return "Search API key not configured"

    try:
        tavily = TavilyClient(api_key=api_key)
        response = tavily.search(query=query, max_results=3)
        return response["results"]
    except Exception as e:
        logger.error("web_search failed: %s", e)
        return "Search failed"


# ============================================================
# GUARDRAIL: TOOL ALLOWLIST + VALIDATION
# ============================================================

ALLOWED_TOOLS = {"calculator", "web_search"}


def parse_tool_call(raw):
    """Validate the model's tool decision. Returns the request dict, or None if rejected."""

    try:
        request = json.loads(raw.strip())
    except json.JSONDecodeError:
        logger.warning("REJECTED - malformed tool call: %r", raw)
        return None

    if not isinstance(request, dict):
        logger.warning("REJECTED - not a JSON object: %r", request)
        return None

    if "tool" not in request or "input" not in request:
        logger.warning("REJECTED - missing keys: %r", request)
        return None

    # "none" is a valid decision, not a violation
    if request["tool"] == "none":
        return request

    if request["tool"] not in ALLOWED_TOOLS:
        logger.warning("REJECTED - tool not in allowlist: %r", request)
        return None

    if not isinstance(request["input"], str):
        logger.warning("REJECTED - input is not a string: %r", request)
        return None

    if not request["input"].strip():
        logger.warning("REJECTED - input is empty: %r", request)
        return None

    if len(request["input"]) > 200:
        logger.warning("REJECTED - input too long: %d chars", len(request["input"]))
        return None

    if request["tool"] == "calculator":
        if not all(c in "0123456789+-*/(). " for c in request["input"]):
            logger.warning("REJECTED - calculator input has invalid characters: %r", request)
            return None

    return request


def use_tool(tool_name: str, tool_input: str):
    # second check, in case use_tool is ever called from elsewhere
    if tool_name not in ALLOWED_TOOLS:
        logger.warning("REJECTED at execution - tool not allowed: %s", tool_name)
        return "Tool not allowed"

    if tool_name == "calculator":
        return calculator(tool_input)

    if tool_name == "web_search":
        return web_search(tool_input)

    return "Unknown tool"


# ============================================================
# MODEL DECISION
# ============================================================

TOOL_PROMPT = """You have these tools:

- calculator: evaluates an arithmetic expression.
  input: the expression, e.g. "23 * 4"

- web_search: searches the web.
  input: the search query

Reply with ONLY a JSON object, no other text:
{"tool": "<tool name>", "input": "<input>"}

If no tool is needed, reply:
{"tool": "none", "input": ""}

User question: """


def choose_tool(question: str):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": TOOL_PROMPT + question}],
        temperature=0
    )

    raw = resp.choices[0].message.content
    logger.info("Model proposed: %s", raw.strip())

    return parse_tool_call(raw)


# ============================================================
# AGENT
# ============================================================

def run_agent(question: str):
    print(f"\n{'=' * 60}\nQUESTION: {question}\n{'=' * 60}")

    request = choose_tool(question)

    if request is None:
        print("BLOCKED: tool request rejected by guardrail")
        return

    if request["tool"] == "none":
        print("No tool needed")
        return

    logger.info("APPROVED: %s", request)

    try:
        result = use_tool(request["tool"], request["input"])
        print("Result:")
        print(result)
    except Exception as e:
        logger.error("Tool execution failed: %s", e)
        print("Tool execution failed:", e)


def test_guardrail():
    """Feed parse_tool_call the responses a misbehaving model would produce."""
    print(f"\n{'=' * 60}\nGUARDRAIL TESTS\n{'=' * 60}")

    cases = [
        '{"tool": "delete_file", "input": "/home/salma/important.txt"}',
        '{"tool": "calculator", "input": "__import__(\'os\').system(\'ls\')"}',
        'sure! here is the json: {"tool": "calculator", "input": "2+2"}',
        '{"tool": "calculator"}',
        '{"tool": "calculator", "input": ""}',
        '{"tool": "calculator", "input": "2+2"}',
    ]

    for raw in cases:
        print(f"\nRAW:    {raw}")
        print(f"PARSED: {parse_tool_call(raw)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_guardrail()
    else:
        run_agent("what is 23 times 4")