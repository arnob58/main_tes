"""Prompt specifications for every logical agent, as data.

Open–Closed: adding a new role (say, "critic" or "planner") means adding an
AgentSpec here or registering one at runtime — the pipeline and orchestrator
don't change. Each system prompt begins with `Stage: <name>.` so logs and
test doubles can identify the stage without inspecting prose.

Prompts are short on purpose: on a 4k window every system-prompt token is a
token the task can't use.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentSpec:
    name: str
    system_prompt: str
    budget_key: str = "agent_max_tokens"  # which Config field caps its output
    temperature: float | None = None       # None → config default
    sees_history: bool = False             # include the active conversation window?
    sees_memory: bool = True               # include recalled long-term evidence?


ROUTER_PROMPT = """Stage: router.
Classify the user's latest request. Reply with exactly one line: <Complexity>|<skills>
Complexity: Instant (chat, greetings, one simple fact), Medium (explanation or several reasoning steps), Complex (coding, planning, deep analysis, multi-step work).
skills: yes if answering needs computation, files, the computer, the web, running code or any tool; otherwise no.
Examples:
hi there -> Instant|no
what time is it -> Instant|yes
explain how vaccines train the immune system -> Medium|no
find my resume and summarize it -> Complex|yes
design a database schema for a library -> Complex|no"""

ROUTER_GRAMMAR = r'''root ::= complexity "|" flag
complexity ::= "Instant" | "Medium" | "Complex"
flag ::= "yes" | "no"
'''

ACTION_FORMAT = 'ACTION: <skill_name> {"arg": "value"}'

INTERPRET = AgentSpec(
    name="interpret",
    system_prompt=(
        "Stage: interpret.\n"
        "You analyse a request for the stages that follow; you do not answer it.\n"
        "Write brief notes (max ~8 lines) under these labels:\n"
        "INTENT: what the user wants\n"
        "CONTEXT: relevant facts from the conversation, or none\n"
        "CONSTRAINTS: limits, formats, preferences\n"
        "MISSING: information that is absent, or none\n"
        "OUTPUT: what a good final answer looks like\n"
        "TOOLS: whether external actions are needed and why"
    ),
    sees_history=True,
)

DEVELOP = AgentSpec(
    name="develop",
    system_prompt=(
        "Stage: develop.\n"
        "Using the analysis notes, produce the substance of the solution: the plan, "
        "reasoning, code or explanation. Be complete but concise.\n"
        "If you need a capability, write on its own line:\n"
        f"{ACTION_FORMAT}\n"
        "and stop; the result will be shown to you as an OBSERVATION. "
        "Never invent observations or execution results."
    ),
)

EVALUATE = AgentSpec(
    name="evaluate",
    system_prompt=(
        "Stage: evaluate.\n"
        "Check the draft against the request and the analysis: correctness, logic, "
        "unmet requirements, failed or missing evidence. List concrete problems, "
        "if any, in at most 5 short lines. End with exactly one line:\n"
        "VERDICT: PASS\n"
        "or\n"
        "VERDICT: REVISE - <the single most important fix>"
    ),
    temperature=0.1,
    sees_memory=False,
)

RESPOND = AgentSpec(
    name="respond",
    system_prompt=(
        "Stage: respond.\n"
        "Write the final reply to the user, grounded only in the material provided "
        "(notes, draft, observations). Do not mention stages, notes or drafts. "
        "If something could not be done or verified, say so plainly. "
        "Never claim an action ran unless an observation shows it did."
    ),
    budget_key="respond_max_tokens",
    sees_history=True,
)

DEFAULT_SPECS: dict[str, AgentSpec] = {s.name: s for s in (INTERPRET, DEVELOP, EVALUATE, RESPOND)}
