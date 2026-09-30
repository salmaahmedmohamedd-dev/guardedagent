# LangChain + LangGraph version of the guarded agent.
# LangChain does two jobs: ChatOpenAI talks to the model and @tool wraps the tools. LangGraph controls the flow.
#
# NOT used on purpose: LangChain's own agent executor (create_agent). it runs whatever tool the model asks for,
# which makes the model's proposal and the execution the same thing, the exact failure this project prevents.
# here the model only proposes, and the graph runs every guardrail from agent.py before anything executes:
#   layer 1 check_user_input, layer 2 parse_tool_call, layer 4 check_tool_output + check_final_answer,
#   step limit, repeat detection. layer 3 is the docker container. nothing is re-implemented in this file.

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

import agent
from agent import ALLOWED_TOOLS, MODEL, OLLAMA_URL, logger, use_tool  # same logger, so the demo UI can trace it
from agent_langgraph import build_graph, run_graph

# LangChain's client for the same local ollama server. ollama ignores the key but the client requires one
llm = ChatOpenAI(base_url=OLLAMA_URL, api_key="ollama", model=MODEL, temperature=0)


# TOOLS: LangChain wrappers around the same tool functions. they only run after the guardrails approve

@tool("calculator")
def calculator_tool(text: str):
    """Evaluate a basic arithmetic expression, for example '23 * 4'. Only numbers and + - * / ( ) are allowed."""
    return agent.calculator(text)  # still the character allowlist + AST walker, not eval()


@tool("web_search")
def web_search_tool(text: str):
    """Search the web for a query and return up to 3 results."""
    return agent.web_search(text)


TOOLS = {t.name: t for t in (calculator_tool, web_search_tool)}
assert set(TOOLS) == ALLOWED_TOOLS  # the two lists must never drift apart


def lc_ask(prompt, label):
    # the model call through LangChain. same prompts as agent.py, same logging
    content = llm.invoke(prompt).content
    raw = content if isinstance(content, str) else ""
    logger.info("%s: %s", label, raw.strip())
    return raw


def lc_execute(tool_name, tool_input):
    # second allowlist check, independent of whoever called us (same reason as agent.use_tool)
    if tool_name not in TOOLS:
        return use_tool(tool_name, tool_input)  # rejects, logs "REJECTED at execution", returns "Tool not allowed"
    return TOOLS[tool_name].invoke({"text": tool_input})  # the raw result goes on to the layer 4 output check


app = build_graph(ask=lc_ask, execute=lc_execute)  # the same graph as agent_langgraph.py, LangChain plugged in


def run_agent(question: str):
    return run_graph(app, question)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "graph":  # python agent_langchain.py graph -> mermaid diagram
        print(app.get_graph().draw_mermaid())
    else:
        run_agent("what is 23 times 4")