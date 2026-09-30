# backend for the visual demo. run with:  python server.py   then open http://127.0.0.1:8000
#
# it runs either agent (agent.py or agent_langgraph.py) and streams every step to the browser live.
# the agents are NOT changed for this: they already log and print every decision (model proposed,
# APPROVED, REJECTED ..., Result, Answer). this file listens to those logs and prints while the agent
# runs and turns each one into a structured event the frontend can draw. so the ui shows exactly what
# the real guardrails did, not a separate copy of the logic

import io
import json
import logging
import os
import queue
import re
import threading
import time
from contextlib import redirect_stdout

import httpx
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

import agent as plain_agent            # the while-loop version
import agent_langgraph as graph_agent  # the langgraph version, same guardrails

HERE = os.path.dirname(os.path.abspath(__file__))
AGENTS = {"langgraph": graph_agent.run_agent, "plain": plain_agent.run_agent}

app = FastAPI(title="GuardedAgent demo")

# the trace is built from INFO logs (Model proposed, APPROVED, ...). agent.py sets INFO with basicConfig,
# but that does nothing if logging was already configured (uvicorn, pytest), so set it explicitly here
for name in ("agent", "agent_langgraph"):
    logging.getLogger(name).setLevel(logging.INFO)

# only one question at a time: the local model can only work on one anyway, and stdout capture
# is process wide, so two runs at once would mix their output
run_lock = threading.Lock()


# EVENT TRANSLATION
# each rule: (pattern on the log message, what it means). layers follow the project brief:
# layer 1 input, layer 2 tool-call validation, layer 4 output. "loop" = step limit / repeat handling

LOG_RULES = [
    (r"^REJECTED input - (.*)",            dict(layer="input",  status="blocked", title="Input guardrail blocked the question")),
    (r"^Model proposed next step: (.*)",   dict(layer="model",  status="info",    title="Model chose the next step", node="next_step")),
    (r"^Model proposed: (.*)",             dict(layer="model",  status="info",    title="Model chose a tool", node="first_step")),
    (r"^REJECTED at execution - (.*)",     dict(layer="tool",   status="blocked", title="Execution check blocked the tool")),
    (r"^REJECTED - (.*)",                  dict(layer="tool",   status="blocked", title="Tool-call guardrail rejected the request")),
    (r"^APPROVED: (.*)",                   dict(layer="tool",   status="ok",      title="Tool call approved", node="run_tool")),
    (r"^REJECTED output - (.*)",           dict(layer="output", status="blocked", title="Output guardrail rejected the tool result")),
    (r"^REJECTED answer - (.*)",           dict(layer="output", status="blocked", title="Output guardrail rejected the final answer")),
    (r"^TRUNCATED answer - (.*)",          dict(layer="output", status="warn",    title="Output guardrail shortened the answer")),
    (r"^Model repeated a tool call, asking once more: (.*)",
                                           dict(layer="loop",   status="warn",    title="Model repeated itself, warned once", node="warn_repeat")),
    (r"^DONE - model repeated a tool call again.*?: (.*)",
                                           dict(layer="loop",   status="warn",    title="Repeated again, treated as finished")),
    (r"^STOPPED - reached max steps \((.*)\)",
                                           dict(layer="loop",   status="warn",    title="Step limit reached")),
    (r"^Model answered: (.*)",             dict(layer="model",  status="info",    title="Model wrote the final answer", node="final_answer")),
]

PRINT_RULES = [
    (r"^Please ask a question$",                        dict(layer="input",  status="blocked", title="Empty question, nothing sent to the model")),
    (r"^BLOCKED: question rejected by input guardrail", None),  # already shown by the REJECTED input log
    (r"^BLOCKED: tool request rejected by guardrail",   "tool_blocked"),
    (r"^BLOCKED: step (\d+) output rejected",           None),  # already shown by the REJECTED output log
    (r"^BLOCKED: final answer rejected",                None),  # already shown by the REJECTED answer log
    (r"^No tool needed$",                               dict(layer="model",  status="info",    title="No tool needed, answering directly")),
    (r"^Result \(step (\d+), (\w+)\):$",                "result"),
    (r"^Answer:$",                                      "answer"),
    (r"^(Tool execution failed|Answer generation failed|Next step failed|Agent run failed):? ?(.*)",
                                                        "error"),
]


class EventCollector:
    # turns log records and printed lines from one agent run into events on a queue

    def __init__(self, q):
        self.q = q
        self.mode = None        # "result" or "answer": which block the next printed lines belong to
        self.has_result = False
        self.answer_lines = []
        self.blocked = False

    def emit(self, **event):
        self.q.put(event)

    def on_log(self, record):
        if record.name not in ("agent", "agent_langgraph"):
            return
        msg = record.getMessage()
        for pattern, meaning in LOG_RULES:
            m = re.match(pattern, msg, re.S)
            if m:
                ev = dict(type="step", detail=m.group(1), **meaning)
                if meaning["status"] == "blocked":
                    self.blocked = True
                node = ev.pop("node", None)
                if node:
                    self.emit(type="node", node=node)
                self.mode = None
                self.emit(**ev)
                return
        if record.levelno >= logging.ERROR:
            self.emit(type="step", layer="error", status="blocked", title="Error", detail=msg)

    def on_line(self, line):
        for pattern, action in PRINT_RULES:
            m = re.match(pattern, line)
            if not m:
                continue
            if action is None:
                return
            if action == "tool_blocked":
                self.blocked = True
                if not self.has_result:
                    self.emit(type="node", node="blocked")  # rejected before any tool ran
                return
            if action == "result":
                self.has_result = True
                self.mode = "result"
                self.emit(type="step", layer="result", status="ok",
                          title=f"Step {m.group(1)}: {m.group(2)} returned (after output check)", detail="")
                return
            if action == "answer":
                self.mode = "answer"
                return
            if action == "error":
                self.emit(type="step", layer="error", status="blocked", title=m.group(1), detail=m.group(2))
                return
            self.mode = None
            if action["status"] == "blocked":
                self.blocked = True
            self.emit(type="step", detail="", **action)
            return

        # plain line: belongs to the block above it
        if self.mode == "result":
            self.emit(type="append", text=line)
        elif self.mode == "answer":
            self.answer_lines.append(line)
            self.emit(type="answer", text="\n".join(self.answer_lines))


class LogToCollector(logging.Handler):
    def __init__(self, collector):
        super().__init__(level=logging.INFO)
        self.collector = collector

    def emit(self, record):
        try:
            self.collector.on_log(record)
        except Exception:
            pass  # the ui must never break the agent run


class LinesToCollector(io.TextIOBase):
    # stands in for stdout during a run: every complete printed line goes to the collector
    def __init__(self, collector):
        self.collector = collector
        self.buffer = ""

    def write(self, text):
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self.collector.on_line(line)
        return len(text)

    def flush(self):
        if self.buffer:
            self.collector.on_line(self.buffer)
            self.buffer = ""


def run_in_background(question, agent_name, q):
    collector = EventCollector(q)
    handler = LogToCollector(collector)
    root = logging.getLogger()
    root.addHandler(handler)
    started = time.time()
    try:
        collector.emit(type="node", node="check_input")
        writer = LinesToCollector(collector)
        with redirect_stdout(writer):
            AGENTS[agent_name](question)
            writer.flush()
        outcome = "answered" if collector.answer_lines else ("blocked" if collector.blocked else "no_answer")
    except Exception as e:
        collector.emit(type="step", layer="error", status="blocked", title="Server error", detail=str(e))
        outcome = "error"
    finally:
        root.removeHandler(handler)
        run_lock.release()
    collector.emit(type="done", outcome=outcome, elapsed=round(time.time() - started, 1))


# ROUTES

@app.get("/")
def page():
    return FileResponse(os.path.join(HERE, "index.html"))


@app.get("/api/run")
def run(question: str = Query("", max_length=2000), agent: str = Query("langgraph")):
    if agent not in AGENTS:
        return JSONResponse({"error": "agent must be 'langgraph' or 'plain'"}, status_code=400)

    def stream():
        if not run_lock.acquire(blocking=False):
            yield sse(dict(type="busy"))
            return
        q = queue.Queue()
        yield sse(dict(type="start", agent=agent, question=question))
        threading.Thread(target=run_in_background, args=(question, agent, q), daemon=True).start()
        while True:
            try:
                event = q.get(timeout=15)
            except queue.Empty:
                yield ": still working\n\n"  # keeps the connection open during slow model calls
                continue
            yield sse(event)
            if event["type"] == "done":
                return

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/health")
def health():
    # lets the page show whether ollama is running and the model is pulled, the most common setup problem
    status = {"ollama": False, "model": plain_agent.MODEL, "model_ready": False,
              "search_key": bool(os.getenv("TAVILY_API_KEY")), "busy": run_lock.locked()}
    try:
        ollama_root = str(plain_agent.client.base_url).rstrip("/").removesuffix("/v1")  # same address the agent uses
        tags = httpx.get(ollama_root + "/api/tags", timeout=2).json()
        status["ollama"] = True
        status["model_ready"] = any(m.get("name") == plain_agent.MODEL for m in tags.get("models", []))
    except Exception:
        pass
    return status


@app.get("/api/graph")
def graph():
    # the real langgraph structure as mermaid text, for the report
    return {"mermaid": graph_agent.app.get_graph().draw_mermaid()}


def sse(event):
    return f"data: {json.dumps(event)}\n\n"


#run code below only if runned directly
if __name__ == "__main__":
    # 127.0.0.1 = only reachable from this computer, not from the network.
    # docker sets HOST=0.0.0.0 so the port mapping can reach it; compose still only publishes it on 127.0.0.1
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=8000)