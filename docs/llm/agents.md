# AI Agents & Tool Use

AI Agents are autonomous systems that use LLMs to perceive environments, reason about goals, plan multi-step actions, use tools, and iteratively work toward completing complex tasks. They go far beyond single-turn Q&A.

---

## 📖 **Sections**

- [Agent Architecture](#agent-architecture)
- [Tool Use / Function Calling](#tool-use--function-calling)
- [ReAct Loop Pattern](#react-loop-pattern)
- [Multi-Agent Systems](#multi-agent-systems)
- [Memory Systems](#memory-systems)
- [Agent Frameworks](#agent-frameworks)
- [Building a Production Agent](#building-a-production-agent)

---

## 🏗️ **Agent Architecture**

```
┌─────────────────────────────────────────────┐
│                AI AGENT                     │
│                                             │
│  Perception    Reasoning     Action         │
│  ┌────────┐   ┌──────────┐  ┌──────────┐   │
│  │ Input  │──►│  LLM     │──►│  Tools   │   │
│  │ (text, │   │ (plan,   │  │ (search, │   │
│  │  img,  │   │  decide) │  │  code,   │   │
│  │  tools)│   └────┬─────┘  │  APIs)   │   │
│  └────────┘        │        └────┬─────┘   │
│                    │             │          │
│       ┌────────────▼─────────────┘          │
│       │         Memory                      │
│       │  (conversation, facts, plans)       │
│       └─────────────────────────────────────┘
└─────────────────────────────────────────────┘
```

**Core components:**
- **LLM**: The reasoning engine
- **Tools**: Functions the agent can call (search, code execution, APIs)
- **Memory**: Short-term (context window) + long-term (vector DB / files)
- **Planning**: Breaking tasks into subtasks
- **Observation**: Processing tool results to inform next steps

---

## 🔧 **Tool Use / Function Calling**

### Basic Tool Use with Claude

```python
import anthropic
import json
from datetime import datetime
import subprocess

client = anthropic.Anthropic()

# Define tools
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "execute_python",
        "description": "Execute Python code and return the output. Use for calculations, data processing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"]
        }
    },
    {
        "name": "search_web",
        "description": "Search the web for current information",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
]

# Tool implementations
def get_weather(city: str, unit: str = "celsius") -> dict:
    # In production, call a real weather API
    return {
        "city": city,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "partly cloudy",
        "humidity": 65
    }

def execute_python(code: str) -> dict:
    """Safely execute Python code in a subprocess."""
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=10
        )
        return {"output": result.stdout, "error": result.stderr}
    except subprocess.TimeoutExpired:
        return {"error": "Code execution timed out"}

def search_web(query: str) -> dict:
    # In production, call a real search API (Brave, SerpAPI, etc.)
    return {"results": [f"Search result for: {query}"]}

TOOL_MAP = {
    "get_weather": get_weather,
    "execute_python": execute_python,
    "search_web": search_web,
}

def process_tool_call(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return its result as a string."""
    if tool_name not in TOOL_MAP:
        return f"Error: Unknown tool '{tool_name}'"

    result = TOOL_MAP[tool_name](**tool_input)
    return json.dumps(result)

# Agentic loop
def run_agent(user_message: str, max_iterations: int = 10) -> str:
    """Run agent until it produces a final answer or hits max iterations."""
    messages = [{"role": "user", "content": user_message}]

    for iteration in range(max_iterations):
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            tools=tools,
            messages=messages
        )

        # If model wants to use tools
        if response.stop_reason == "tool_use":
            # Add assistant's response (includes tool use blocks)
            messages.append({"role": "assistant", "content": response.content})

            # Execute all tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  Tool: {block.name}({json.dumps(block.input)[:100]})")
                    result = process_tool_call(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            # Add tool results to conversation
            messages.append({"role": "user", "content": tool_results})

        else:
            # Model produced final answer
            final_text = next(
                (block.text for block in response.content if hasattr(block, "text")), ""
            )
            print(f"Completed in {iteration + 1} iteration(s)")
            return final_text

    return "Max iterations reached without final answer"

# Test the agent
answer = run_agent("What's the weather in Tokyo? Then calculate 15% tip on a $87.50 restaurant bill.")
print(f"\nAnswer: {answer}")
```

---

## 🔄 **ReAct Loop Pattern**

Reason → Act → Observe, repeat until done.

```python
class ReActAgent:
    """
    ReAct (Reasoning + Acting) agent that explicitly
    thinks before taking each action.
    """

    SYSTEM_PROMPT = """You are a helpful AI agent. For each step:
1. THOUGHT: Think about what to do next
2. ACTION: Choose a tool to use
3. OBSERVATION: Review the tool result
4. Repeat until you have a final answer, then output: FINAL ANSWER: [your answer]

Available tools: {tools}"""

    def __init__(self, tools: list[dict]):
        self.client = anthropic.Anthropic()
        self.tools = tools
        self.tool_map = {}

    def register_tool(self, name: str, func):
        self.tool_map[name] = func

    def run(self, task: str, max_steps: int = 10) -> str:
        tool_names = ", ".join(t["name"] for t in self.tools)
        system = self.SYSTEM_PROMPT.format(tools=tool_names)

        messages = [{"role": "user", "content": f"Task: {task}"}]
        steps = []

        for step in range(max_steps):
            response = self.client.messages.create(
                model="claude-opus-4-6",
                max_tokens=2048,
                system=system,
                tools=self.tools,
                messages=messages
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                # Check for final answer
                text = next((b.text for b in response.content if hasattr(b, "text")), "")
                if "FINAL ANSWER:" in text:
                    return text.split("FINAL ANSWER:")[-1].strip()

            elif response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        steps.append(f"Step {step+1}: {block.name}({block.input})")
                        result = process_tool_call(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })

                messages.append({"role": "user", "content": tool_results})

        return "Could not complete task within step limit"
```

---

## 🤝 **Multi-Agent Systems**

Multiple specialized agents collaborating on complex tasks.

```python
class AgentOrchestrator:
    """
    Orchestrator that delegates to specialized sub-agents.
    Pattern: one planner + multiple specialists.
    """

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.agents = {}

    def register_agent(self, name: str, description: str, system_prompt: str):
        self.agents[name] = {
            "description": description,
            "system_prompt": system_prompt
        }

    def call_agent(self, agent_name: str, task: str, context: str = "") -> str:
        """Delegate a task to a specific sub-agent."""
        agent = self.agents[agent_name]

        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            system=agent["system_prompt"],
            messages=[{
                "role": "user",
                "content": f"Context: {context}\n\nTask: {task}" if context else task
            }]
        )
        return response.content[0].text

    def run(self, user_request: str) -> str:
        """Planner decides which agents to call and in what order."""
        agents_desc = "\n".join(
            f"- {name}: {info['description']}"
            for name, info in self.agents.items()
        )

        # Planner creates execution plan
        plan_response = self.client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"""Available agents:
{agents_desc}

Create a step-by-step execution plan for this request.
Format: JSON list of {{"agent": "name", "task": "specific task", "depends_on": [step indices]}}

Request: {user_request}"""
            }]
        )

        import json
        plan = json.loads(plan_response.content[0].text)

        # Execute plan
        results = {}
        for i, step in enumerate(plan):
            context = "\n".join(
                f"Step {dep} result: {results[dep]}"
                for dep in step.get("depends_on", [])
                if dep in results
            )

            print(f"Step {i}: Agent '{step['agent']}' - {step['task'][:60]}...")
            results[i] = self.call_agent(step["agent"], step["task"], context)

        # Synthesize final answer
        all_results = "\n\n".join(
            f"Step {i} ({plan[i]['agent']}): {result}"
            for i, result in results.items()
        )

        synthesis = self.client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": f"Synthesize these agent outputs into a final answer:\n\n{all_results}\n\nOriginal request: {user_request}"
            }]
        )
        return synthesis.content[0].text

# Setup multi-agent system
orchestrator = AgentOrchestrator()

orchestrator.register_agent(
    "researcher",
    "Finds and summarizes information",
    "You are a research specialist. Find relevant information, summarize key facts, and cite sources."
)
orchestrator.register_agent(
    "analyst",
    "Analyzes data and draws insights",
    "You are a data analyst. Analyze information, identify patterns, and provide quantitative insights."
)
orchestrator.register_agent(
    "writer",
    "Creates clear, well-structured documents",
    "You are a professional writer. Create clear, concise, well-structured content based on provided information."
)

result = orchestrator.run(
    "Analyze the state of the AI industry in 2025 and write an executive summary for non-technical stakeholders."
)
```

---

## 🧠 **Memory Systems**

### Hierarchical Memory

```python
from collections import deque
import json
from pathlib import Path

class AgentMemory:
    """
    Three-tier memory system:
    1. Working memory: Current conversation (context window)
    2. Episodic memory: Recent interactions (in-memory)
    3. Long-term memory: Persistent facts (file/vector DB)
    """

    def __init__(self, agent_id: str, working_memory_size: int = 10):
        self.agent_id = agent_id
        self.working_memory = deque(maxlen=working_memory_size)
        self.episodic_memory = []
        self.memory_file = Path(f"./agent_memory_{agent_id}.json")
        self.long_term = self._load_long_term()

    def _load_long_term(self) -> dict:
        if self.memory_file.exists():
            return json.loads(self.memory_file.read_text())
        return {"facts": {}, "preferences": {}, "task_history": []}

    def _save_long_term(self):
        self.memory_file.write_text(json.dumps(self.long_term, indent=2))

    def add_to_working(self, role: str, content: str):
        self.working_memory.append({"role": role, "content": content})

    def remember_fact(self, key: str, value: str):
        """Store a fact in long-term memory."""
        self.long_term["facts"][key] = {"value": value, "timestamp": str(datetime.now())}
        self._save_long_term()

    def recall_fact(self, key: str) -> str | None:
        fact = self.long_term["facts"].get(key)
        return fact["value"] if fact else None

    def get_context(self) -> list[dict]:
        """Get conversation context for LLM."""
        return list(self.working_memory)

    def summarize_and_compress(self, client: anthropic.Anthropic):
        """When working memory fills up, summarize and store key facts."""
        if len(self.working_memory) < 5:
            return

        history = "\n".join(f"{m['role']}: {m['content']}" for m in self.working_memory)

        summary_response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": f"Extract 3-5 key facts from this conversation as bullet points:\n\n{history}"
            }]
        )

        # Move to episodic memory
        self.episodic_memory.append({
            "summary": summary_response.content[0].text,
            "timestamp": str(datetime.now())
        })

        # Clear working memory, keep last 2 messages
        last_two = list(self.working_memory)[-2:]
        self.working_memory.clear()
        self.working_memory.extend(last_two)
```

---

## 🛠️ **Agent Frameworks**

### LangGraph (State Machine Agents)

```python
# pip install langgraph
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: list
    tool_calls: list
    final_answer: str
    iteration: int

def should_continue(state: AgentState) -> str:
    """Decide whether to continue or end."""
    if state.get("final_answer"):
        return END
    if state["iteration"] >= 10:
        return END
    return "agent"

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_llm)
workflow.add_node("tools", execute_tools)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"agent": "tools", END: END})
workflow.add_edge("tools", "agent")

app = workflow.compile()

# Run
result = app.invoke({
    "messages": [{"role": "user", "content": "Research and summarize recent AI breakthroughs"}],
    "tool_calls": [],
    "final_answer": "",
    "iteration": 0
})
```

---

## 🏭 **Building a Production Agent**

```python
import anthropic
import logging
from dataclasses import dataclass, field
from typing import Any
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    model: str = "claude-opus-4-6"
    max_iterations: int = 20
    max_tokens: int = 4096
    timeout_seconds: int = 300
    retry_attempts: int = 3

class ProductionAgent:
    def __init__(self, config: AgentConfig, tools: list[dict], system_prompt: str):
        self.client = anthropic.Anthropic()
        self.config = config
        self.tools = tools
        self.system = system_prompt

    def run(self, task: str, session_id: str = None) -> dict:
        start_time = time.time()
        messages = [{"role": "user", "content": task}]
        iterations = 0
        tool_call_count = 0

        logger.info(f"Session {session_id}: Starting task: {task[:100]}")

        while iterations < self.config.max_iterations:
            # Timeout check
            if time.time() - start_time > self.config.timeout_seconds:
                logger.warning(f"Session {session_id}: Timeout after {iterations} iterations")
                return {"status": "timeout", "iterations": iterations}

            # Call LLM with retry logic
            response = self._call_with_retry(messages)

            if response.stop_reason == "end_turn":
                text = next((b.text for b in response.content if hasattr(b, "text")), "")
                duration = time.time() - start_time
                logger.info(f"Session {session_id}: Completed in {iterations} iter, {duration:.1f}s, {tool_call_count} tool calls")
                return {
                    "status": "success",
                    "answer": text,
                    "iterations": iterations,
                    "tool_calls": tool_call_count,
                    "duration_seconds": duration
                }

            elif response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []

                for block in response.content:
                    if block.type == "tool_use":
                        tool_call_count += 1
                        logger.info(f"Session {session_id}: Tool call {tool_call_count}: {block.name}")
                        result = process_tool_call(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })

                messages.append({"role": "user", "content": tool_results})

            iterations += 1

        return {"status": "max_iterations_reached", "iterations": iterations}

    def _call_with_retry(self, messages: list) -> Any:
        for attempt in range(self.config.retry_attempts):
            try:
                return self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    system=self.system,
                    tools=self.tools,
                    messages=messages
                )
            except anthropic.RateLimitError:
                wait = 2 ** attempt
                logger.warning(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            except anthropic.APIError as e:
                if attempt == self.config.retry_attempts - 1:
                    raise
                time.sleep(1)

        raise RuntimeError("All retry attempts failed")
```

---

## 💡 **Tips & Tricks**

1. **Limit tool surface area**: Give agents only the tools they need — fewer tools = better decisions
2. **Tool descriptions are critical**: The LLM decides what to call based on descriptions; be precise
3. **Always have a timeout**: Agents can get stuck in loops — enforce max iterations + wall clock timeout
4. **Log every tool call**: For debugging and audit trails, log all tool invocations with inputs/outputs
5. **Structured tool outputs**: Return consistent JSON from tools; avoid free-form text
6. **Human-in-the-loop for dangerous actions**: For irreversible actions (delete, send email, pay), add approval step
7. **Use smaller models for simple steps**: Route to Haiku for retrieval/formatting, Opus for complex reasoning

---

## 🔗 **Related Topics**

- [LLM Agents](../LLM/Agents.md)
- [Prompt Engineering](../Prompt-Engineering/README.md)
- [RAG - Retrieval Augmented Generation](../RAG/README.md)
- [MLOps & Deployment](../MLOps/README.md)
