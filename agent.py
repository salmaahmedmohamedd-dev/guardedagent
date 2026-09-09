import ast #abstract syntax tree .safer than eval().as it parses string.parse is take text and analyze it to representation comp. understands
import operator # gives python funcs for math ops
import requests #python to comm with a webs service
import logging 
import os 
from tavily import TavilyClient

#basic agent skeleton
from dotenv import load_dotenv #reads variables from .env and make them here

load_dotenv() #performs loading
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def calculator(expression: str):
    #only allowed to receive those
    allowed = "0123456789+-*/(). "
    if not all(char in allowed for char in expression):#2 guardrails 1 allowed chars
        return "Invalid expression"
    try:
        tree = ast.parse(expression, mode="eval") #mode eval makes input treated as 1 expression.
#user input,parse into tree,inspect tree
        operators = {#explicitly sayin + = addition ...
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
        }

        def evaluate(node):#node is the current part of the tree
            if isinstance(node, ast.Expression):#if node is expression go to its body to evaluate it
                return evaluate(node.body)
#is the node number constant we allow int /float
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
#binary op. has left and right.calc left then right then perform operation
            if isinstance(node, ast.BinOp) and type(node.op) in operators:
                left = evaluate(node.left)
                right = evaluate(node.right)
                return operators[type(node.op)](left, right)
#handles negatice no.
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                return -evaluate(node.operand)

            raise ValueError("Invalid expression")

        return evaluate(tree)

    except Exception:
        return "Invalid expression"



def web_search(query: str):#query is the text we want to search for
    if not query.strip():#strip removes spaces.checks if query is empty
        return "Invalid search query"

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        return "Search API key not configured"


    tavily = TavilyClient(api_key=api_key)

    response = tavily.search(
        query=query,
        max_results=3
    )

    results = response["results"]

    return results

ALLOWED_TOOLS = {"calculator", "web_search"} # tool allowlist

def validate_tool_request(request):#guardrail f kol step
    if not isinstance(request, dict):#dictionary key-value pairs.msln tool w calc
        return False

    if "tool" not in request:
        return False

    if "input" not in request:
        return False

    if request["tool"] not in ALLOWED_TOOLS:
        return False

    if not isinstance(request["input"], str):
        return False

    if not request["input"].strip():
        return False

    return True

def use_tool(tool_name: str, tool_input: str):
    if tool_name not in ALLOWED_TOOLS:
        return "Tool not allowed"
    if tool_name == "calculator":
        return calculator(tool_input)
    if tool_name == "web_search":
        return web_search(tool_input)

def choose_tool(question: str):
    if any(op in question for op in ["+", "-", "*", "/"]):
        return {
            "tool": "calculator",
            "input": question
        }

    return {
        "tool": "web_search",
        "input": question
    }

def run_agent(question: str):
    request = choose_tool(question)

    logger.info("Tool request: %s", request)#lodder instead of print.to record what happened

    if not validate_tool_request(request):
        print("Tool execution failed:", e)
        return

    try:
        result = use_tool(request["tool"], request["input"])
        print(result)

    except Exception as e:
        print("Tool execution failed:", e)

#only run when file is exec directly
if __name__ == "__main__":
        run_agent("What is Docker?")

