"""
llm_handler.py
--------------
Responsibility: Send messages to the OpenAI Chat Completions API and
                return the parsed response.

KEY FIX: Accumulates attributes across conversation turns so previously
collected values (fuel_type, transmission) are never lost when the user
provides budget in a later message.
"""

import json
import re
from openai import OpenAI

from prompt_builder import build_attribute_extraction_prompt

CHAT_MODEL = "gpt-4o-mini"

# Canonical allowed values — used for validation after LLM response
ALLOWED_FUEL       = {"petrol", "diesel", "cng", "electric", "hybrid"}
ALLOWED_TRANS      = {"manual", "automatic"}
ALLOWED_USAGE      = {"city", "highway", "off-road", "family"}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_budget(raw: str) -> str:
    """
    Convert any budget string → canonical "X lakh" form.
    Examples:
        "25 lakhs"  → "25 lakh"
        "25L"       → "25 lakh"
        "around 25" → "25 lakh"
        "23"        → "23 lakh"
        ""          → ""
    """
    if not raw:
        return ""
    raw = str(raw).strip().lower()

    # Extract first number (int or float)
    match = re.search(r"(\d+(?:\.\d+)?)", raw)
    if not match:
        return ""

    amount = match.group(1)
    # Remove trailing .0
    if amount.endswith(".0"):
        amount = amount[:-2]

    return f"{amount} lakh"


def _merge_attributes(existing: dict, new_attrs: dict) -> dict:
    """
    Merge newly extracted attributes into the existing accumulated dict.
    Rule: NEVER overwrite a valid value with an empty string.
    This ensures attributes collected in earlier turns are preserved.
    """
    merged = dict(existing)
    for key, new_val in new_attrs.items():
        new_val = str(new_val).strip()
        if new_val:  # only update if LLM actually returned something
            merged[key] = new_val
    return merged


def _validate_attributes(attrs: dict) -> dict:
    """
    Validate and clean extracted attributes against allowed value sets.
    Invalid values are set back to "" so the bot re-asks for them.
    """
    fuel = str(attrs.get("fuel_type", "")).strip().lower()
    trans = str(attrs.get("transmission", "")).strip().lower()
    usage = str(attrs.get("usage", "")).strip().lower()
    budget = _normalize_budget(attrs.get("budget", ""))

    return {
        "fuel_type":    fuel    if fuel  in ALLOWED_FUEL  else "",
        "transmission": trans   if trans in ALLOWED_TRANS  else "",
        "budget":       budget,
        "usage":        usage   if usage in ALLOWED_USAGE  else "",
    }


def _is_ready(attrs: dict) -> bool:
    """All three required attributes must be non-empty."""
    return bool(
        attrs.get("fuel_type")
        and attrs.get("transmission")
        and attrs.get("budget")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_attributes_via_llm(
    user_message: str,
    conversation_history: list[dict],
    client: OpenAI,
    accumulated_attributes: dict | None = None,  # ← NEW: pass in existing state
) -> dict:
    """
    Send the full conversation history to the LLM and extract car preference
    attributes, merging them with any previously collected attributes.

    Args:
        user_message            — latest message from the user
        conversation_history    — full [{role, content}] history including new msg
        client                  — authenticated OpenAI client
        accumulated_attributes  — attributes collected in previous turns (or None)

    Returns:
        dict with keys:
            'message'              (str)  — conversational reply for the user
            'attributes'           (dict) — MERGED attributes (all turns combined)
            'ready_for_retrieval'  (bool) — True when fuel+transmission+budget all set
    """
    if accumulated_attributes is None:
        accumulated_attributes = {
            "fuel_type": "", "transmission": "", "budget": "", "usage": ""
        }

    system_prompt = build_attribute_extraction_prompt()

    # Inject current known attributes into the system prompt so the LLM
    # doesn't re-ask for things already collected
    known_summary = "\n".join(
        f"  - {k}: {v}" for k, v in accumulated_attributes.items() if v
    )
    if known_summary:
        system_prompt += f"""

══════════════════════════════════════════════════════
ALREADY COLLECTED (do NOT ask for these again):
══════════════════════════════════════════════════════
{known_summary}

Only ask for attributes that are still missing from the above list.
"""

    messages = [{"role": "system", "content": system_prompt}] + conversation_history

    # ── LLM call ────────────────────────────────────────────────────────────
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        raw_content = response.choices[0].message.content
        llm_result = json.loads(raw_content)

    except json.JSONDecodeError as e:
        print(f"[LLM HANDLER] JSON parse error: {e}")
        # Graceful fallback: keep existing attributes, ask user to continue
        return {
            "message": "Sorry, I didn't catch that. Could you repeat your preference?",
            "attributes": accumulated_attributes,
            "ready_for_retrieval": False,
        }
    except Exception as e:
        print(f"[LLM HANDLER] API error: {e}")
        return {
            "message": "Something went wrong. Please try again.",
            "attributes": accumulated_attributes,
            "ready_for_retrieval": False,
        }

    # ── Extract & validate new attributes from LLM ──────────────────────────
    raw_new_attrs = llm_result.get("attributes", {})
    validated_new = _validate_attributes(raw_new_attrs)

    # ── Merge with accumulated state (never lose old values) ─────────────────
    merged = _merge_attributes(accumulated_attributes, validated_new)

    # ── Re-check readiness based on merged state (not LLM's claim) ──────────
    ready = _is_ready(merged)

    # If LLM says ready but we know values are missing, override it
    if ready != llm_result.get("ready_for_retrieval", False):
        print(
            f"[LLM HANDLER] Overriding LLM ready_for_retrieval: "
            f"LLM={llm_result.get('ready_for_retrieval')} → Actual={ready}"
        )

    # If ready, message must be empty (retrieval handles the response)
    message = "" if ready else llm_result.get("message", "")

    # ── Debug logging ────────────────────────────────────────────────────────
    print(f"[ATTR EXTRACT] Raw LLM attrs : {raw_new_attrs}")
    print(f"[ATTR EXTRACT] Validated     : {validated_new}")
    print(f"[ATTR EXTRACT] Merged state  : {merged}")
    print(f"[ATTR EXTRACT] Ready         : {ready}")

    return {
        "message":             message,
        "attributes":          merged,
        "ready_for_retrieval": ready,
    }