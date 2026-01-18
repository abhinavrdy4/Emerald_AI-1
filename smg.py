# smg.py
import os
import json
import re
from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import HTTPException, APIRouter
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo
from dateutil import parser as dtparser

from google import genai  # pip install google-genai

load_dotenv()

TZ_NAME_DEFAULT = "Asia/Kolkata"
TZ_DEFAULT = ZoneInfo(TZ_NAME_DEFAULT)

# -------------------- Models --------------------

class TaskIn(BaseModel):
    Title: str
    dependency: Optional[str] = None
    deadline: Optional[str] = None
    effort: Optional[int] = None
    priority: str = "Medium"
    additional_details: str = Field(default="", alias="additional details")

class ScheduleRequest(BaseModel):
    tasks: List[TaskIn]
    existing_events: List[Dict[str, Any]] = Field(default_factory=list)

    timezone: str = TZ_NAME_DEFAULT
    work_start: str = "07:00"
    work_end: str = "23:00"
    lunch_start: str = "13:00"
    lunch_end: str = "14:00"
    break_minutes: int = 10
    schedule_from: Optional[str] = None

class EventOut(BaseModel):
    summary: str
    description: str
    start: Dict[str, str]
    end: Dict[str, str]

class ScheduleResponse(BaseModel):
    events_to_create: List[EventOut]
    reasoning: str

# -------------------- Router --------------------

router = APIRouter()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# -------------------- Helpers: parsing --------------------

def parse_dt(s: str, tz: ZoneInfo) -> datetime:
    dt = dtparser.parse(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(tz)

def parse_date_only(s: str) -> date:
    return dtparser.parse(s).date()

def parse_hhmm(hhmm: str) -> time:
    hh, mm = hhmm.split(":")
    return time(int(hh), int(mm))

def dt_to_iso(dt: datetime) -> str:
    return dt.isoformat()

# -------------------- Helpers: intervals --------------------

@dataclass(frozen=True)
class Interval:
    start: datetime
    end: datetime

def merge_intervals(intervals: List[Interval]) -> List[Interval]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x.start)
    out = [intervals[0]]
    for cur in intervals[1:]:
        last = out[-1]
        if cur.start <= last.end:
            out[-1] = Interval(last.start, max(last.end, cur.end))
        else:
            out.append(cur)
    return out

def intersects(a: Interval, b: Interval) -> bool:
    return a.start < b.end and b.start < a.end

def subtract_busy(free: Interval, busy: Interval) -> List[Interval]:
    if not intersects(free, busy):
        return [free]
    pieces = []
    if busy.start > free.start:
        pieces.append(Interval(free.start, busy.start))
    if busy.end < free.end:
        pieces.append(Interval(busy.end, free.end))
    return pieces

# -------------------- Existing events -> busy intervals --------------------

def events_to_busy(existing_events: List[Dict[str, Any]], tz: ZoneInfo) -> List[Interval]:
    busy = []
    for ev in existing_events:
        start = ev.get("start", {})
        end = ev.get("end", {})

        # Timed event
        if "dateTime" in start and "dateTime" in end:
            s = parse_dt(start["dateTime"], tz)
            e = parse_dt(end["dateTime"], tz)
            if e > s:
                busy.append(Interval(s, e))
            continue

        # All-day event (date -> date)
        if "date" in start and "date" in end:
            d0 = parse_date_only(start["date"])
            d1 = parse_date_only(end["date"])  # end is exclusive
            cur = d0
            while cur < d1:
                s = datetime.combine(cur, time(0, 0), tzinfo=tz)
                e = datetime.combine(cur, time(23, 59, 59), tzinfo=tz)
                busy.append(Interval(s, e))
                cur = cur + timedelta(days=1)
            continue

    return merge_intervals(busy)

# -------------------- Workday windows (with lunch) --------------------

def day_windows(day: date, tz: ZoneInfo, work_start: time, work_end: time,
                lunch_start: time, lunch_end: time) -> List[Interval]:
    ws = datetime.combine(day, work_start, tzinfo=tz)
    we = datetime.combine(day, work_end, tzinfo=tz)
    ls = datetime.combine(day, lunch_start, tzinfo=tz)
    le = datetime.combine(day, lunch_end, tzinfo=tz)

    windows = []
    if ls > ws:
        windows.append(Interval(ws, min(ls, we)))
    if we > le:
        windows.append(Interval(max(le, ws), we))

    return [w for w in windows if w.end > w.start]

# -------------------- Task ordering (dependencies + urgency) --------------------

PRIORITY_RANK = {"High": 3, "Medium": 2, "Low": 1}

def infer_dependencies(task: TaskIn) -> List[str]:
    deps = []
    if task.dependency:
        deps.append(task.dependency.strip())

    ad = task.additional_details or ""
    if "Depends on:" in ad:
        part = ad.split("Depends on:", 1)[1]
        for d in part.split(","):
            d = d.strip()
            if d:
                deps.append(d)
    return deps

def topo_order(tasks: List[TaskIn]) -> List[TaskIn]:
    def norm(s: str) -> str:
        return " ".join(s.lower().split())

    by_key = {norm(t.Title): t for t in tasks}
    deps_map = {norm(t.Title): [norm(d) for d in infer_dependencies(t)] for t in tasks}

    temp = set()
    perm = set()
    out_keys = []

    def visit(n: str):
        if n in perm:
            return
        if n in temp:
            return  # cycle -> ignore
        temp.add(n)
        for d in deps_map.get(n, []):
            if d in by_key:
                visit(d)
        temp.remove(n)
        perm.add(n)
        out_keys.append(n)

    for k in by_key:
        visit(k)

    ordered = [by_key[k] for k in out_keys if k in by_key]

    def deadline_key(t: TaskIn):
        if t.deadline:
            try:
                return parse_dt(t.deadline, TZ_DEFAULT)
            except Exception:
                return datetime.max.replace(tzinfo=TZ_DEFAULT)
        return datetime.max.replace(tzinfo=TZ_DEFAULT)

    ordered.sort(key=lambda t: (deadline_key(t), -PRIORITY_RANK.get(t.priority, 2)))
    return ordered

# -------------------- Preferred time extraction --------------------

def extract_preferred_time_hint(text: str) -> Optional[Dict[str, Any]]:
    """
    Supported patterns inside additional details:
    - Preferred time: YYYY-MM-DD HH:MM
    - Preferred time: HH:MM
    - Preferred window: HH:MM-HH:MM
    - casual: "at 6am", "at 6 pm", "at 18:00"
    Returns dict or None.
    """
    if not text:
        return None
    s = text.strip()

    m = re.search(r"preferred\s*time\s*:\s*(\d{4}-\d{2}-\d{2})\s+(\d{1,2}:\d{2})", s, re.IGNORECASE)
    if m:
        d = dtparser.parse(m.group(1)).date()
        tod = parse_hhmm(m.group(2))
        return {"type": "exact_date_time", "date": d, "time": tod}

    m = re.search(r"preferred\s*time\s*:\s*(\d{1,2}:\d{2})", s, re.IGNORECASE)
    if m:
        return {"type": "time_of_day", "time": parse_hhmm(m.group(1))}

    m = re.search(r"preferred\s*window\s*:\s*(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})", s, re.IGNORECASE)
    if m:
        return {"type": "window", "start": parse_hhmm(m.group(1)), "end": parse_hhmm(m.group(2))}

    m = re.search(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", s, re.IGNORECASE)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2) or "0")
        ap = (m.group(3) or "").lower()
        if ap == "pm" and hh != 12:
            hh += 12
        if ap == "am" and hh == 12:
            hh = 0
        hh = hh % 24
        return {"type": "time_of_day", "time": time(hh, mm)}

    return None

def build_preferred_windows(
    hint: Dict[str, Any],
    start_from: datetime,
    deadline: Optional[datetime],
    tz: ZoneInfo,
    work_start_t: time,
    work_end_t: time,
    lunch_start_t: time,
    lunch_end_t: time,
) -> List[Interval]:
    windows: List[Interval] = []

    def clamp_to_work(day: date, raw_start: datetime, raw_end: datetime) -> List[Interval]:
        allowed = day_windows(day, tz, work_start_t, work_end_t, lunch_start_t, lunch_end_t)
        out = []
        for a in allowed:
            s = max(raw_start, a.start)
            e = min(raw_end, a.end)
            if e > s:
                out.append(Interval(s, e))
        return out

    def apply_deadline(ws: List[Interval]) -> List[Interval]:
        if not deadline:
            return ws
        out = []
        for w in ws:
            if w.start >= deadline:
                continue
            e = min(w.end, deadline)
            if e > w.start:
                out.append(Interval(w.start, e))
        return out

    if hint["type"] == "exact_date_time":
        d = hint["date"]
        tod = hint["time"]
        s = datetime.combine(d, tod, tzinfo=tz)
        e = s + timedelta(hours=4)
        if e <= start_from:
            return []
        return apply_deadline(clamp_to_work(d, s, e))

    if hint["type"] == "time_of_day":
        tod = hint["time"]
        horizon_days = 7
        if deadline:
            horizon_days = min(horizon_days, max(1, (deadline.date() - start_from.date()).days + 1))
        for i in range(horizon_days):
            d = start_from.date() + timedelta(days=i)
            s = datetime.combine(d, tod, tzinfo=tz)
            e = s + timedelta(hours=4)
            if e <= start_from:
                continue
            windows.extend(clamp_to_work(d, s, e))
        return apply_deadline(windows)

    if hint["type"] == "window":
        wstart = hint["start"]
        wend = hint["end"]
        if wend <= wstart:
            return []
        horizon_days = 7
        if deadline:
            horizon_days = min(horizon_days, max(1, (deadline.date() - start_from.date()).days + 1))
        for i in range(horizon_days):
            d = start_from.date() + timedelta(days=i)
            s = datetime.combine(d, wstart, tzinfo=tz)
            e = datetime.combine(d, wend, tzinfo=tz)
            if e <= start_from:
                continue
            windows.extend(clamp_to_work(d, s, e))
        return apply_deadline(windows)

    return []

def find_slot_in_windows(
    duration_min: int,
    windows: List[Interval],
    start_from: datetime,
    deadline: Optional[datetime],
    busy: List[Interval],
) -> Optional[Interval]:
    dur = timedelta(minutes=duration_min)

    for w in windows:
        w0 = w
        if w0.end <= start_from:
            continue
        if w0.start < start_from:
            w0 = Interval(start_from, w0.end)

        if deadline:
            if w0.start >= deadline:
                continue
            w0 = Interval(w0.start, min(w0.end, deadline))

        if w0.end - w0.start < dur:
            continue

        free_parts = [w0]
        for b in busy:
            new_parts = []
            for fp in free_parts:
                new_parts.extend(subtract_busy(fp, b))
            free_parts = new_parts
            if not free_parts:
                break

        for fp in free_parts:
            if fp.end - fp.start >= dur:
                return Interval(fp.start, fp.start + dur)

    return None

# -------------------- Slot finding (fallback) --------------------

def find_earliest_slot(
    duration_min: int,
    start_from: datetime,
    deadline: Optional[datetime],
    busy: List[Interval],
    tz: ZoneInfo,
    work_start_t: time,
    work_end_t: time,
    lunch_start_t: time,
    lunch_end_t: time,
) -> Optional[Interval]:
    dur = timedelta(minutes=duration_min)

    if deadline:
        last_day = deadline.date()
        horizon_days = max(1, min(14, (last_day - start_from.date()).days + 1))
    else:
        horizon_days = 14

    cur_day = start_from.date()

    for i in range(horizon_days):
        day = cur_day + timedelta(days=i)
        windows = day_windows(day, tz, work_start_t, work_end_t, lunch_start_t, lunch_end_t)

        for w in windows:
            w0 = w
            if w0.end <= start_from:
                continue
            if w0.start < start_from:
                w0 = Interval(start_from, w0.end)

            if deadline and w0.start >= deadline:
                return None
            if deadline and w0.end > deadline:
                w0 = Interval(w0.start, deadline)

            if w0.end - w0.start < dur:
                continue

            free_parts = [w0]
            for b in busy:
                new_parts = []
                for fp in free_parts:
                    new_parts.extend(subtract_busy(fp, b))
                free_parts = new_parts
                if not free_parts:
                    break

            for fp in free_parts:
                if fp.end - fp.start >= dur:
                    return Interval(fp.start, fp.start + dur)

    return None

# -------------------- Main scheduler --------------------

def schedule_tasks(
    tasks: List[TaskIn],
    existing_events: List[Dict[str, Any]],
    tz_name: str,
    work_start: str,
    work_end: str,
    lunch_start: str,
    lunch_end: str,
    break_minutes: int,
    schedule_from: Optional[str],
) -> Tuple[List[EventOut], List[Dict[str, Any]]]:
    tz = ZoneInfo(tz_name)

    busy = events_to_busy(existing_events, tz)

    now = datetime.now(tz)
    start_from = parse_dt(schedule_from, tz) if schedule_from else now

    work_start_t = parse_hhmm(work_start)
    work_end_t = parse_hhmm(work_end)
    lunch_start_t = parse_hhmm(lunch_start)
    lunch_end_t = parse_hhmm(lunch_end)

    ordered = topo_order(tasks)

    planned_events: List[EventOut] = []
    decisions: List[Dict[str, Any]] = []
    current_pointer = start_from

    for t in ordered:
        effort = t.effort if (isinstance(t.effort, int) and t.effort > 0) else 30
        deadline_dt = parse_dt(t.deadline, tz) if t.deadline else None

        # break before each task except first
        if planned_events:
            current_pointer = current_pointer + timedelta(minutes=break_minutes)

        # --- Try preferred time/window first ---
        slot = None
        hint = extract_preferred_time_hint(t.additional_details or "")
        preferred_windows: List[Interval] = []
        if hint:
            preferred_windows = build_preferred_windows(
                hint=hint,
                start_from=current_pointer,
                deadline=deadline_dt,
                tz=tz,
                work_start_t=work_start_t,
                work_end_t=work_end_t,
                lunch_start_t=lunch_start_t,
                lunch_end_t=lunch_end_t,
            )

        if preferred_windows:
            slot = find_slot_in_windows(
                duration_min=effort,
                windows=preferred_windows,
                start_from=current_pointer,
                deadline=deadline_dt,
                busy=busy,
            )
            if slot:
                decisions.append({
                    "task": t.Title,
                    "note": "Scheduled near preferred time from additional details (no conflicts).",
                    "scheduled_start": slot.start.isoformat(),
                    "scheduled_end": slot.end.isoformat(),
                })

        # --- Fallback to earliest available (respect deadline if present) ---
        if slot is None:
            slot = find_earliest_slot(
                duration_min=effort,
                start_from=current_pointer,
                deadline=deadline_dt,
                busy=busy,
                tz=tz,
                work_start_t=work_start_t,
                work_end_t=work_end_t,
                lunch_start_t=lunch_start_t,
                lunch_end_t=lunch_end_t,
            )

            if slot is None:
                slot = find_earliest_slot(
                    duration_min=effort,
                    start_from=current_pointer,
                    deadline=None,
                    busy=busy,
                    tz=tz,
                    work_start_t=work_start_t,
                    work_end_t=work_end_t,
                    lunch_start_t=lunch_start_t,
                    lunch_end_t=lunch_end_t,
                )
                if slot is None:
                    raise RuntimeError(f"Could not find any free slot for task: {t.Title}")

                decisions.append({
                    "task": t.Title,
                    "note": "Deadline could not be satisfied; scheduled at next available slot.",
                    "scheduled_start": slot.start.isoformat(),
                    "scheduled_end": slot.end.isoformat(),
                })
            else:
                decisions.append({
                    "task": t.Title,
                    "note": "Scheduled at earliest available slot respecting work hours, lunch, and conflicts.",
                    "scheduled_start": slot.start.isoformat(),
                    "scheduled_end": slot.end.isoformat(),
                })

        # reserve the slot
        busy = merge_intervals(busy + [slot])

        description = f"Priority: {t.priority}\n{t.additional_details or ''}".strip()

        planned_events.append(EventOut(
            summary=t.Title,
            description=description,
            start={"dateTime": dt_to_iso(slot.start), "timeZone": tz_name},
            end={"dateTime": dt_to_iso(slot.end), "timeZone": tz_name},
        ))

        current_pointer = slot.end

    return planned_events, decisions

# -------------------- Gemini reasoning (explanation only) --------------------

def gemini_explain(tasks: List[TaskIn], decisions: List[Dict[str, Any]], tz_name: str) -> str:
    if gemini_client is None:
        lines = ["Scheduled tasks without conflicts, respecting work hours, lunch, breaks, and deadlines when possible. If a task is asked for a preferred time, schedule at that time if there isn't a conflict with existing events."]
        for d in decisions:
            lines.append(f"- {d['task']}: {d['note']} ({d['scheduled_start']} → {d['scheduled_end']})")
        return "\n".join(lines)

    payload = {
        "timezone": tz_name,
        "tasks": [t.model_dump(by_alias=True) for t in tasks],
        "decisions": decisions,
    }

    resp = gemini_client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=(
            "Explain scheduling decisions clearly for the user.\n"
            "Mention when you honored preferred times from additional details.\n"
            "Constraints: no overlaps with existing events, breaks between tasks, respect deadlines and priorities when possible.\n"
            "If any deadline could not be met, explain why and what was done.\n"
            "Write a concise but clear explanation.\n\n"
            f"DATA:\n{json.dumps(payload, indent=2)}"
        ),
    )
    return (resp.text or "").strip()

# -------------------- Endpoints --------------------

@router.post("/schedule/plan", response_model=ScheduleResponse)
def plan_schedule(req: ScheduleRequest):
    try:
        planned_events, decisions = schedule_tasks(
            tasks=req.tasks,
            existing_events=req.existing_events,
            tz_name=req.timezone,
            work_start=req.work_start,
            work_end=req.work_end,
            lunch_start=req.lunch_start,
            lunch_end=req.lunch_end,
            break_minutes=req.break_minutes,
            schedule_from=req.schedule_from,
        )
        reasoning = gemini_explain(req.tasks, decisions, req.timezone)
        return ScheduleResponse(events_to_create=planned_events, reasoning=reasoning)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/health")
def health():
    return {"ok": True, "has_gemini_key": bool(GEMINI_API_KEY)}
