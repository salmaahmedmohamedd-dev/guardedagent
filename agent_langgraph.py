# langgraph version of the guarded agent: same guardrails, model and tools as agent.py.
# only HOW the flow is controlled changes: agent.py uses a while loop, here every step is a node and every
# decision (blocked? done? repeat? step limit?) is an edge, so the flow is explicit, drawable and testable.
# nothing is re-implemented: every guardrail comes from agent.py, so a fix there fixes every version.
# build_graph() takes `ask` (how the model is called) and `execute` (how a tool runs), that is the only thing
# agent_langchain.py changes

import logging
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agent import (MAX_STEPS, ask_model, check_user_input, choose_next_tool, choose_tool,
                   finish, is_repeat, repeat_warning, run_step, use_tool)

logger = logging.getLogger(__name__)


# STATE: shared memory every node reads and writes. a node returns only the keys it changed

class AgentState(TypedDict):
    question: str            # the user's question, cleaned by the input guardrail
    request: Optional[dict]  # latest tool request after guardrails, None if rejected
    history: list            # every checked tool result: {"tool", "input", "output"}
    nudged: bool             # the model gets one warning if it repeats a call
    note: str                # that warning, sent with the next step (empty if none)
    answer: Optional[str]    # final answer after layer 4, None if blocked
    stop_reason: str         # why the run ended, for logs and tests


# NODES that never touch the model or a tool: each does one job, the edges decide where to go next

def check_input(state: AgentState):
    if not state["question"].strip():  # don't waste a model call on an empty question
        print("Please ask a question")
        return {"stop_reason": "empty_question"}
    question = check_user_input(state["question"])  # layer 1, before the model sees anything
    if question is None:
        print("BLOCKED: question rejected by input guardrail")
        return {"stop_reason": "input_rejected"}
    return {"question": question}


def warn_repeat(state: AgentState):
    return {"nudged": True, "note": repeat_warning(state["request"])}  # warn once, then choose again


def blocked(state: AgentState):
    print("BLOCKED: tool request rejected by guardrail")  # the first request was rejected, nothing to answer from
    return {"stop_reason": "blocked"}


# EDGES: only read the state and return a route name. names are the guardrail decision, they become diagram labels

# three ways a later step ends lead to final_answer. the diagram draws one edge per node pair, so one label lists all
FINISH = "finish: done, rejected, or repeated again"


def after_input(state: AgentState):
    if state.get("stop_reason") in ("empty_question", "input_rejected"):
        return "empty or rejected input"
    return "input ok"


def after_first_step(state: AgentState):
    request = state["request"]  # no results yet, so a rejection means nothing to answer from
    if request is None:
        return "rejected by guardrail"
    return "no tool needed" if request["tool"] == "none" else "approved"


def after_next_step(state: AgentState):
    request = state["request"]  # there is always at least one result to answer from
    if request is None:
        print("BLOCKED: tool request rejected by guardrail")
        return FINISH  # a bad later step must not throw away the good results we already have
    if request["tool"] == "none":  # model says it has enough
        return FINISH
    if is_repeat(request, state["history"]):  # same call again means the model is stuck
        if state["nudged"]:
            logger.info("DONE - model repeated a tool call again, treating it as finished: %r", request)
            return FINISH
        return "repeated call"
    return "approved"


def after_tool(state: AgentState):
    if len(state["history"]) >= MAX_STEPS:  # step limit: the agent can never loop forever
        logger.warning("STOPPED - reached max steps (%d)", MAX_STEPS)
        return "step limit reached"
    return "ask model for next step"


# GRAPH
# START -> check_input -> first_step -> (decision) -> run_tool -> next_step -> (decision) -> ... -> final_answer -> END

def build_graph(ask=ask_model, execute=use_tool):
    # the four nodes that call the model or run a tool are closures over `ask` / `execute`
    def first_step(state: AgentState):
        return {"request": choose_tool(state["question"], ask)}  # layer 2 runs inside choose_tool

    def run_tool(state: AgentState):
        entry = run_step(state["request"], state["history"], execute)  # execute + layer 4 on the output
        return {"history": state["history"] + [entry], "note": ""}  # a real step clears any old warning

    def next_step(state: AgentState):
        return {"request": choose_next_tool(state["question"], state["history"], state["note"], ask)}

    def final_answer(state: AgentState):
        if not state["history"] and not state.get("stop_reason"):
            print("No tool needed")
        return {"answer": finish(state["question"], state["history"], ask)}  # always checked by layer 4

    g = StateGraph(AgentState)
    for name, fn in [("check_input", check_input), ("first_step", first_step), ("run_tool", run_tool),
                     ("warn_repeat", warn_repeat), ("next_step", next_step),
                     ("final_answer", final_answer), ("blocked", blocked)]:
        g.add_node(name, fn)

    g.add_edge(START, "check_input")
    g.add_conditional_edges("check_input", after_input, {"input ok": "first_step", "empty or rejected input": END})
    # first decision can only be rejected, no tool, or approved (a repeat is impossible with no history)
    g.add_conditional_edges("first_step", after_first_step, {
        "rejected by guardrail": "blocked", "no tool needed": "final_answer", "approved": "run_tool"})
    # later decisions never go to "blocked": there are always results to answer from
    g.add_conditional_edges("next_step", after_next_step, {
        "approved": "run_tool", FINISH: "final_answer", "repeated call": "warn_repeat"})
    g.add_conditional_edges("run_tool", after_tool, {
        "ask model for next step": "next_step", "step limit reached": "final_answer"})
    g.add_edge("warn_repeat", "next_step")
    g.add_edge("final_answer", END)
    g.add_edge("blocked", END)
    return g.compile()


app = build_graph()  # compiled once, reused for every question

# backstop: even if the routing had a bug, langgraph stops after this many node runs.
# worst real path: check_input + first_step + 3x(run_tool + next_step) + warn_repeat + next_step + final_answer = 12
RECURSION_LIMIT = 20


def run_graph(graph, question: str):
    initial = {"question": question, "request": None, "history": [], "nudged": False,
               "note": "", "answer": None, "stop_reason": ""}
    try:
        return graph.invoke(initial, config={"recursion_limit": RECURSION_LIMIT})
    except Exception as e:
        logger.error("Agent run failed: %s", e)
        print("Agent run failed:", e)
        return None


def run_agent(question: str):
    return run_graph(app, question)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "graph":  # python agent_langgraph.py graph -> mermaid diagram
        print(app.get_graph().draw_mermaid())
    else:
        run_agent("what is 23 times 4")