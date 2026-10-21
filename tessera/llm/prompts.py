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
    user_facing: bool = False              # its text is shown to the user verbatim


ROUTER_PROMPT = """Stage: router.
Classify the user's latest request. Reply with exactly one line: <Complexity>|<skills>
Complexity: Instant (chat, greetings, one simple fact), Medium (explanation or several reasoning steps), Complex (coding, planning, deep analysis, multi-step work).
skills: yes if answering needs computation, files, the computer, the web, running code or any tool; otherwise no.
Examples:
hi there -> Instant|no
what time is it -> Instant|yes
explain how vaccines train the immune system -> Medium|no
find my resume and summarize it -> Complex|yes
create me a website tracker -> Complex|yes
build a to-do app -> Complex|yes
write a python script that renames my photos -> Complex|yes
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
        "If the user asks to create, build, write or save a file, page, script or project, you MUST save it. "
        "Code in your reply is NOT a saved file. For anything longer than ~40 lines, describe it and let the "
        "create_file tool write it in parts, e.g.\n"
        'ACTION: create_file {"path": "notes_app/index.html", "description": "A notes page: ... (every feature, '
        'every design detail, every API URL the user gave)"}\n'
        "For a short file, write it directly in this plain format (no JSON, no escaping):\n"
        "FILE: folder/name.txt\n<exact file content>\nEND FILE\n"
        "If no listed capability fits, request one anyway with a descriptive snake_case name and the arguments it "
        "needs; it will be written, tested and kept for next time.\n"
        "If an OBSERVATION shows an error or bugs, fix the cause and try again with the corrected input — do not "
        "give up after one failure and do not repeat the same failing call.\n"
        "Never invent observations or execution results."
    ),
)

EVALUATE = AgentSpec(
    name="evaluate",
    system_prompt=(
        "Stage: evaluate.\n"
        "Check the draft against the request and the analysis: correctness, logic, "
        "unmet requirements, failed or missing evidence. List concrete problems, "
        "if any, in at most 5 short lines. If SAVED FILES are present, they are the deliverable: judge the file "
        "content, not the draft text — a saved file that does what was asked means PASS. "
        "End with exactly one line:\n"
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
        "Never claim an action ran unless an observation shows it did. "
        "Give the answer itself: never narrate your own process or say what you 'will' do. "
        "When an observation shows a file was saved, give its full path and a 1-2 sentence summary — do not "
        "paste the file's contents. "
        "If the request is too vague to do well, ask ONE short clarifying question with 2-3 example options instead."
    ),
    budget_key="respond_max_tokens",
    sees_history=True,
    user_facing=True,
)

DEFAULT_SPECS: dict[str, AgentSpec] = {s.name: s for s in (INTERPRET, DEVELOP, EVALUATE, RESPOND)}
