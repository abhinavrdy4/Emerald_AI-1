# main.py
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai  # pip install google-genai

# -------------------- Startup --------------------

load_dotenv()

app = FastAPI(title="Task Updater API (Gemini)", version="2.0.0")

# CORS (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: set to your frontend domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
client = genai.Client() if GEMINI_API_KEY else None

# Choose a fast model for extraction. Change if needed.
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"

# -------------------- API Contracts --------------------

class UpdateReq(BaseModel):
    user_input: str
    tasks: list  # existing tasks array


# -------------------- JSON Schema (strict) --------------------

TASKS_JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "Title": {"type": "string", "minLength": 1},
            "deadline": {"type": ["string", "null"]},
            "effort": {"type": ["integer", "null"], "minimum": 1},
            "priority": {"type": "string", "enum": ["High", "Medium", "Low"]},
            "additional details": {"type": "string"},
        },
        "required": ["Title", "deadline", "effort", "priority", "additional details"],
    },
}

SYSTEM_PROMPT = """
You are a task state manager.

Input:
- existing_tasks: JSON array of tasks (subset of tasks that might be relevant)
- user_input: one directive in natural language (may add OR edit tasks and may imply dependencies)

Output:
Return the UPDATED tasks as a JSON array of task objects.

Rules:
1) PRESERVE STATE INSIDE THIS SUBSET:
   - Return ALL tasks you received in existing_tasks unless the directive explicitly deletes them.
   - Apply edits to matching tasks in this subset.
   - Add new tasks if the directive adds them.
2) Split multiple tasks in the directive into separate entries.
3) Editing:
   - If the directive changes an existing task, match by Title (case-insensitive).
   - If ambiguous, pick the closest match and mention ambiguity in "additional details".
4) Dependencies:
   - Write dependencies inside "additional details" using:
     "Depends on: <Title1>, <Title2>"
   - Example:
     directive: "Email mentor after Finish PPT"
     => Email mentor.additional details MUST include "Depends on: Finish PPT"
5) deadline:
   - If only a date is mentioned, use YYYY-MM-DD.
   - If time is mentioned, use ISO 8601 string (include timezone if available).
   - If missing, null.
6) effort:
   - Convert hours/minutes to integer minutes (e.g., 1.5h -> 90).
   - If missing, null.
7) priority:
   - If not given, infer: urgent deadlines -> High, otherwise Medium; trivial -> Low.
8) Be aware of today's date and time. So, if only time is mentioned, the event should be scheduled for anytime after right now.
CRITICAL OUTPUT RULES:
- Output MUST be a single JSON array of task objects. No nesting. No extra wrapper keys.
- Never output empty objects {}.
- Keys MUST be exactly: "Title", "deadline", "effort", "priority", "additional details".
- Deduplicate tasks by Title (case-insensitive). If duplicates occur, merge them into ONE task:
  - Prefer values that are not null/empty.
  - If priorities differ, choose the higher urgency: High > Medium > Low.
""".strip()


# -------------------- Helpers: cleaning / dedupe / merge --------------------

PRIORITY_RANK = {"High": 3, "Medium": 2, "Low": 1}

def _norm_title(t: str) -> str:
    return " ".join(t.strip().lower().split())

def _flatten_list(x: Any) -> List[Any]:
    out: List[Any] = []
    if isinstance(x, list):
        for item in x:
            out.extend(_flatten_list(item))
    else:
        out.append(x)
    return out

def _coerce_task_keys(t: Dict[str, Any]) -> Dict[str, Any]:
    key_map = {
        "title": "Title",
        "Title": "Title",
        "deadline": "deadline",
        "effort": "effort",
        "priority": "priority",
        "additional_details": "additional details",
        "additional detail": "additional details",
        "additional details": "additional details",
    }
    out: Dict[str, Any] = {}
    for k, v in t.items():
        kk = key_map.get(k)
        if kk:
            out[kk] = v
    return out

def _merge_task(a: dict, b: dict) -> dict:
    out = dict(a)

    # Prefer non-empty values
    out["deadline"] = out.get("deadline") or b.get("deadline")
    out["effort"] = out.get("effort") or b.get("effort")

    # Priority: take higher
    pa = out.get("priority") or "Medium"
    pb = b.get("priority") or "Medium"
    out["priority"] = pb if PRIORITY_RANK.get(pb, 0) > PRIORITY_RANK.get(pa, 0) else pa

    # Merge additional details (avoid duplicates)
    ad = (out.get("additional details") or "").strip()
    bd = (b.get("additional details") or "").strip()
    if bd and bd not in ad:
        out["additional details"] = (ad + ("; " if ad else "") + bd)
    else:
        out["additional details"] = ad

    out["Title"] = (out.get("Title") or b.get("Title") or "").strip()
    return out

def clean_and_dedupe(tasks: Any) -> List[Dict[str, Any]]:
    flat = _flatten_list(tasks)
    cleaned: List[Dict[str, Any]] = []

    for item in flat:
        if not isinstance(item, dict):
            continue
        if "Title" not in item and "title" not in item:
            continue

        t = _coerce_task_keys(item)
        title = t.get("Title")
        if not isinstance(title, str) or not title.strip():
            continue

        t["Title"] = title.strip()
        t["deadline"] = t.get("deadline", None)

        effort = t.get("effort", None)
        t["effort"] = effort if isinstance(effort, int) and effort > 0 else None

        pr = t.get("priority", "Medium")
        t["priority"] = pr if pr in PRIORITY_RANK else "Medium"

        ad = t.get("additional details", "")
        t["additional details"] = ad if isinstance(ad, str) else ""

        cleaned.append(t)

    by_title: Dict[str, Dict[str, Any]] = {}
    for t in cleaned:
        k = _norm_title(t["Title"])
        by_title[k] = _merge_task(by_title[k], t) if k in by_title else t

    return list(by_title.values())

def merge_into_global(global_tasks: List[Dict[str, Any]], partial: Any) -> List[Dict[str, Any]]:
    partial_clean = clean_and_dedupe(partial)
    combined = global_tasks + partial_clean
    return clean_and_dedupe(combined)


# -------------------- Helpers: directive split + candidate selection --------------------

def split_directives(user_input: str) -> List[str]:
    """
    Splits a big user input into smaller directives.
    This keeps each Gemini call small & consistent.
    """
    user_input = user_input.strip()
    if not user_input:
        return []

    # Split by lines/bullets first
    lines: List[str] = []
    for line in user_input.splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove common bullet markers
        line = re.sub(r"^\s*[-*•]\s*", "", line).strip()
        if line:
            lines.append(line)

    if not lines:
        lines = [user_input]

    # Split each line by sentence-ish separators
    parts: List[str] = []
    for line in lines:
        chunks = re.split(r"\s*(?:\.\s+|;\s+)\s*", line)
        for c in chunks:
            c = c.strip(" ,")
            if c:
                parts.append(c)

    # Second pass for long parts: split by connectors
    final: List[str] = []
    for p in parts:
        if len(p) > 140:
            more = re.split(r"\b(?:also|then|and then|next)\b", p, flags=re.IGNORECASE)
            final.extend([m.strip(" ,") for m in more if m.strip(" ,")])
        else:
            final.append(p)

    return [f for f in final if f]

def select_relevant_tasks(existing_tasks: List[Dict[str, Any]], directive: str, k: int = 8) -> List[Dict[str, Any]]:
    """
    Select top-k tasks most relevant to this directive (Title/details overlap).
    This prevents prompt growth as task list becomes large.
    """
    if not existing_tasks:
        return []

    words = set(re.findall(r"[a-z0-9]+", directive.lower()))
    if not words:
        return existing_tasks[:k]

    scored = []
    directive_l = directive.lower()
    for t in existing_tasks:
        hay = f"{t.get('Title','')} {t.get('additional details','')}".lower()
        hay_words = set(re.findall(r"[a-z0-9]+", hay))
        score = len(words & hay_words)
        if t.get("Title") and t["Title"].lower() in directive_l:
            score += 5
        scored.append((score, t))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [t for s, t in scored[:k]]

    # If all scores are 0, still return a small slice (avoid sending everything)
    if top and all(s == 0 for s, _ in scored[:k]):
        return existing_tasks[:k]

    return top


# -------------------- Helpers: dependency patch safety net --------------------

def _best_title_match(fragment: str, titles: List[str]) -> Optional[str]:
    frag = fragment.strip().lower()
    if not frag:
        return None
    # exact
    for tt in titles:
        if frag == tt.lower():
            return tt
    # substring
    for tt in titles:
        if frag in tt.lower():
            return tt
    return None

def patch_dependencies(tasks: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    """
    Safety net: if user writes "A after B" or "A depends on B", ensure additional details includes it.
    Runs on each directive.
    """
    titles = [t["Title"] for t in tasks if t.get("Title")]
    if not titles:
        return tasks

    segments = re.split(r"[.\n;]+", text)
    pairs = []

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        m = re.search(r"(.+?)\s+after\s+(.+?)(?:,|$)", seg, flags=re.IGNORECASE)
        if m:
            pairs.append((m.group(1).strip(), m.group(2).strip()))

        m2 = re.search(r"(.+?)\s+depends on\s+(.+?)(?:,|$)", seg, flags=re.IGNORECASE)
        if m2:
            pairs.append((m2.group(1).strip(), m2.group(2).strip()))

    for a_frag, b_frag in pairs:
        a_title = _best_title_match(a_frag, titles) or a_frag
        b_title = _best_title_match(b_frag, titles) or b_frag

        for t in tasks:
            if t["Title"] == a_title:
                dep_str = f"Depends on: {b_title}"
                ad = t.get("additional details", "")
                if dep_str not in ad:
                    t["additional details"] = (ad + ("; " if ad else "") + dep_str)

    return tasks


# -------------------- Gemini call (structured outputs) --------------------

def gemini_update_subset(existing_tasks_subset: List[Dict[str, Any]], directive: str) -> str:
    """
    Gemini call for ONE directive + small subset of tasks.
    Returns JSON text (array of task objects).
    """
    if client is None:
        raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY not configured")

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "existing_tasks:\n"
        f"{json.dumps(existing_tasks_subset, ensure_ascii=False)}\n\n"
        "user_input:\n"
        f"{directive}"
    )

    # Simple retry for transient errors (rate limits etc.)
    last_err = None
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={
                    "temperature": 0.2,
                    "response_mime_type": "application/json",
                    "response_json_schema": TASKS_JSON_SCHEMA,
                },
            )
            return resp.text
        except Exception as e:
            last_err = e
            time.sleep(0.6 * (attempt + 1))

    raise last_err


# -------------------- Routes --------------------

@app.get("/health")
def health():
    return {
        "ok": True,
        "has_key": bool(GEMINI_API_KEY),
        "model": GEMINI_MODEL,
    }

@app.post("/tasks/update")
def update_tasks(req: UpdateReq):
    """
    POST /tasks/update
    Body: { "user_input": "...", "tasks": [...] }
    Returns: { "tasks": [...] }

    Scalable approach:
    - Split input into directives
    - For each directive: pick top-k relevant tasks and call Gemini on only that subset
    - Merge back into global state deterministically
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured: GEMINI_API_KEY/GOOGLE_API_KEY missing")

    directives = split_directives(req.user_input)
    if not directives:
        return {"tasks": clean_and_dedupe(req.tasks)}

    global_tasks = clean_and_dedupe(req.tasks)

    for directive in directives:
        subset = select_relevant_tasks(global_tasks, directive, k=8)

        # If subset is empty and directive looks like "add", still send empty subset.
        raw = gemini_update_subset(subset, directive)
        try:
            parsed = json.loads(raw)
        except Exception:
            # If something went wrong, fail loudly (better than corrupting state)
            raise HTTPException(status_code=400, detail=f"Gemini returned invalid JSON for directive: {directive}")

        # Merge model output into global list deterministically
        global_tasks = merge_into_global(global_tasks, parsed)

        # Patch dependencies (safety net) and clean again
        global_tasks = patch_dependencies(global_tasks, directive)
        global_tasks = clean_and_dedupe(global_tasks)

    return {"tasks": global_tasks}
