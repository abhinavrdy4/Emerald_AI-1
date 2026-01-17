import json
import os
import re
from typing import Any, Dict, List, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel

# Load .env once at startup (local dev)
load_dotenv()

app = FastAPI(title="Task Updater API", version="1.0.0")

# CORS (tighten allow_origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq client (server-side only)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    # Don't crash here; we'll raise a clear error on request.
    client = None
else:
    client = Groq(api_key=GROQ_API_KEY)


# ----------- Request schema -----------

class UpdateReq(BaseModel):
    user_input: str
    tasks: list


# ----------- Structured output schema -----------

TASKS_SCHEMA = {
    "name": "tasks_array",
    "strict": True,
    "schema": {
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
    },
}

SYSTEM_PROMPT = """
You are a task state manager.

Input:
- existing_tasks: JSON array of tasks (may be empty)
- user_input: natural language that may add multiple tasks, specify dependencies, or edit existing tasks.

Output:
Return the COMPLETE, UPDATED JSON array of tasks.

Rules:
1) PRESERVE STATE: The output array MUST include ALL tasks from 'existing_tasks' that were not modified, PLUS any new or updated tasks. Do not drop existing tasks unless the user explicitly asks to delete them.
2) Split multiple tasks into separate entries.
3) If the user edits an existing task, update it by matching Title (case-insensitive).
   If ambiguous, choose the closest match and mention ambiguity in "additional details".
4) Put dependencies in "additional details" using: "Depends on: <Title1>, <Title2>".
   Example:
   user_input: "Email mentor after Finish PPT"
   => Email mentor.additional details MUST include "Depends on: Finish PPT"
5) deadline:
   - If only a date is mentioned, use YYYY-MM-DD.
   - If time is mentioned, use ISO 8601 string (include timezone if available).
   - If missing, null.
6) effort:
   - Convert hours/minutes to integer minutes if possible (e.g., 1.5h -> 90).
   - If missing, null.
7) priority:
   - If not given, infer: urgent deadlines -> High, otherwise Medium; trivial -> Low.

CRITICAL OUTPUT RULES:
- Output MUST be a single JSON array of task objects. No nesting. No extra wrapper keys.
- Never output empty objects {}.
- Keys MUST be exactly: "Title", "deadline", "effort", "priority", "additional details".
- Deduplicate tasks by Title (case-insensitive). If duplicates occur, merge them into ONE task:
  - Prefer values that are not null/empty.
  - If priorities differ, choose the higher urgency: High > Medium > Low.
""".strip()


# ----------- Post-processing helpers -----------

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
    # Helps JSON-mode fallback where keys sometimes differ
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

    out["deadline"] = out.get("deadline") or b.get("deadline")
    out["effort"] = out.get("effort") or b.get("effort")

    pa = out.get("priority") or "Medium"
    pb = b.get("priority") or "Medium"
    if PRIORITY_RANK.get(pb, 0) > PRIORITY_RANK.get(pa, 0):
        out["priority"] = pb
    else:
        out["priority"] = pa

    ad = (out.get("additional details") or "").strip()
    bd = (b.get("additional details") or "").strip()
    if bd and bd not in ad:
        out["additional details"] = (ad + ("; " if ad else "") + bd)
    else:
        out["additional details"] = ad

    out["Title"] = out.get("Title") or b.get("Title") or ""
    return out


def clean_and_dedupe(tasks: Any) -> List[Dict[str, Any]]:
    flat = _flatten_list(tasks)

    cleaned: List[Dict[str, Any]] = []
    for item in flat:
        if not isinstance(item, dict):
            continue

        if "Title" in item or "title" in item:
            t = _coerce_task_keys(item)
            title = t.get("Title")
            if not isinstance(title, str) or not title.strip():
                continue

            t["Title"] = title.strip()
            t["deadline"] = t.get("deadline", None)

            effort = t.get("effort", None)
            if isinstance(effort, int) and effort > 0:
                t["effort"] = effort
            else:
                t["effort"] = None

            pr = t.get("priority", "Medium")
            if pr not in PRIORITY_RANK:
                pr = "Medium"
            t["priority"] = pr

            ad = t.get("additional details", "")
            t["additional details"] = ad if isinstance(ad, str) else ""

            cleaned.append(t)

    by_title: Dict[str, Dict[str, Any]] = {}
    for t in cleaned:
        k = _norm_title(t["Title"])
        if k in by_title:
            by_title[k] = _merge_task(by_title[k], t)
        else:
            by_title[k] = t

    return list(by_title.values())


def _best_title_match(fragment: str, titles: List[str]) -> Union[str, None]:
    frag = fragment.strip().lower()
    if not frag:
        return None
    for tt in titles:
        if frag == tt.lower():
            return tt
    for tt in titles:
        if frag in tt.lower():
            return tt
    return None


def patch_dependencies(tasks: List[Dict[str, Any]], user_input: str) -> List[Dict[str, Any]]:
    """
    Safety-net dependency patcher for 'A after B' / 'depends on B' phrasing.
    """
    titles = [t["Title"] for t in tasks if t.get("Title")]

    segments = re.split(r"[.\n;]+", user_input)
    pairs = []

    for seg in segments:
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


# ----------- Groq calls -----------

def groq_update(existing_tasks, user_input) -> str:
    if client is None:
        raise RuntimeError("GROQ_API_KEY not configured")

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "existing_tasks:\n"
                    f"{json.dumps(existing_tasks, ensure_ascii=False)}\n\n"
                    "user_input:\n"
                    f"{user_input}"
                ),
            },
        ],
        response_format={"type": "json_schema", "json_schema": TASKS_SCHEMA},
        temperature=0.2,
        max_tokens=1200,
    )
    return resp.choices[0].message.content


def groq_update_fallback_json_mode(existing_tasks, user_input) -> str:
    if client is None:
        raise RuntimeError("GROQ_API_KEY not configured")

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + "\nReturn valid JSON only."},
            {
                "role": "user",
                "content": (
                    "existing_tasks:\n"
                    f"{json.dumps(existing_tasks, ensure_ascii=False)}\n\n"
                    "user_input:\n"
                    f"{user_input}"
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=1200,
    )
    return resp.choices[0].message.content


# ----------- Routes -----------

@app.get("/health")
def health():
    return {"ok": True, "has_key": bool(GROQ_API_KEY)}


@app.post("/tasks/update")
def update_tasks(req: UpdateReq):
    """
    POST /tasks/update
    Body: { "user_input": "...", "tasks": [...] }
    Returns: { "tasks": [...] }
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured: GROQ_API_KEY missing")

    # 1) Structured outputs path
    try:
        raw = groq_update(req.tasks, req.user_input)
        parsed = json.loads(raw)
        updated = clean_and_dedupe(parsed)
        updated = patch_dependencies(updated, req.user_input)
        updated = clean_and_dedupe(updated)
        return {"tasks": updated}
    except Exception:
        pass

    # 2) Fallback: JSON mode (less strict)
    try:
        raw = groq_update_fallback_json_mode(req.tasks, req.user_input)
        parsed = json.loads(raw)

        # Some models may wrap: {"tasks":[...]}
        if isinstance(parsed, dict) and "tasks" in parsed:
            parsed = parsed["tasks"]

        updated = clean_and_dedupe(parsed)
        updated = patch_dependencies(updated, req.user_input)
        updated = clean_and_dedupe(updated)
        return {"tasks": updated}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to produce valid tasks JSON: {str(e)}")
