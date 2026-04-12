import os
import json
import math
import hashlib
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client, Client
from groq import Groq


# =========================================================
# ENV
# =========================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

if not SUPABASE_URL or not SUPABASE_KEY or not GROQ_API_KEY:
    raise RuntimeError("Missing SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY/SUPABASE_KEY, or GROQ_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)


# =========================================================
# APP
# =========================================================
app = FastAPI(title="Credit Card Recommender API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# REQUEST / RESPONSE MODELS
# =========================================================
class RecommendationRequest(BaseModel):
    monthly_expense: float = Field(..., gt=0)
    expense_type: str
    desired_output: str
    max_annual_fee: float = Field(..., ge=0)
    credit_score: int = Field(..., ge=300, le=900)
    use_cache: bool = True
    debug: bool = False


# =========================================================
# HELPERS
# =========================================================
SUPPORTED_EXPENSE_TYPES = {
    "shopping", "dining", "travel", "hotel", "movies",
    "fuel", "grocery", "online", "lounge", "forex", "general"
}

SUPPORTED_OUTPUT_TYPES = {
    "cashback", "movie", "dining", "travel", "hotel",
    "fuel", "lounge", "rewards", "shopping", "forex", "general"
}

REWARD_NORMALIZATION = {
    "cashback": "cashback",
    "cash back": "cashback",
    "reward_points": "rewards",
    "reward points": "rewards",
    "fuel_benefits": "fuel",
    "dining_benefits": "dining",
    "welcome_bonus": "welcome_bonus",
    "milestone_benefits": "milestone",
    "fee_waiver": "fee_waiver",
    "lounge": "lounge",
    "travel": "travel",
    "hotel": "hotel",
    "movie": "movies",
    "movies": "movies",
    "shopping": "shopping",
    "forex": "forex",
    "insurance": "insurance",
    "golf": "lifestyle",
}

USER_INTENT_SYNONYMS = {
    "shopping": ["shopping", "online", "ecommerce", "retail", "department stores"],
    "dining": ["dining", "restaurant", "food"],
    "travel": ["travel", "flight", "airline", "trip", "miles"],
    "hotel": ["hotel", "stay", "travel"],
    "movies": ["movie", "movies", "cinema", "ticket", "bookmyshow"],
    "fuel": ["fuel", "petrol", "diesel", "surcharge"],
    "cashback": ["cashback", "cash back"],
    "movie": ["movie", "entertainment", "ticket", "bookmyshow"],
    "lounge": ["lounge", "priority pass", "airport lounge"],
    "rewards": ["reward points", "rewards", "points"],
}

SECTION_BOOSTS = {
    "fees": 8.0,
    "cashback": 10.0,
    "lounge": 10.0,
    "fuel": 8.0,
    "welcome_bonus": 4.0,
    "eligibility": 3.0,
    "milestone": 6.0,
    "travel": 8.0,
    "dining": 8.0,
    "shopping": 8.0,
    "movies": 8.0,
}


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def normalize_reward_values(values: List[str]) -> List[str]:
    out = []
    for v in values:
        key = normalize_text(v)
        out.append(REWARD_NORMALIZATION.get(key, key))
    return list(dict.fromkeys(out))


def extract_reward_types(card: Dict[str, Any]) -> List[str]:
    raw = card.get("reward_type")
    if raw is None:
        return []
    if isinstance(raw, list):
        return [normalize_text(x) for x in raw if x]
    if isinstance(raw, str):
        cleaned = raw.strip("{}[]")
        parts = [p.strip().strip('"').strip("'") for p in cleaned.split(",") if p.strip()]
        return [normalize_text(x) for x in parts]
    return []


def normalize_user_profile(payload: RecommendationRequest) -> Dict[str, Any]:
    expense_type = normalize_text(payload.expense_type)
    desired_output = normalize_text(payload.desired_output)

    if expense_type not in SUPPORTED_EXPENSE_TYPES:
        expense_type = "general"
    if desired_output not in SUPPORTED_OUTPUT_TYPES:
        desired_output = "general"

    return {
        "monthly_expense": round(float(payload.monthly_expense), 2),
        "annual_expense": round(float(payload.monthly_expense) * 12, 2),
        "expense_type": expense_type,
        "desired_output": desired_output,
        "max_annual_fee": round(float(payload.max_annual_fee), 2),
        "credit_score": int(payload.credit_score),
    }


# =========================================================
# SIMILAR PROFILE CACHE
# =========================================================
def bucket_monthly_expense(x: float) -> int:
    return int(round(float(x) / 10000.0) * 10000)


def bucket_annual_fee(x: float) -> int:
    return int(round(float(x) / 500.0) * 500)


def bucket_credit_score(score: int) -> str:
    score = int(score)
    if score < 650:
        return "below_650"
    elif score < 700:
        return "650_699"
    elif score < 750:
        return "700_749"
    elif score < 800:
        return "750_799"
    return "800_plus"


def build_similarity_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "expense_bucket": bucket_monthly_expense(profile["monthly_expense"]),
        "annual_fee_bucket": bucket_annual_fee(profile["max_annual_fee"]),
        "credit_score_bucket": bucket_credit_score(profile["credit_score"]),
        "expense_type": profile["expense_type"],
        "desired_output": profile["desired_output"],
    }


def make_similarity_hash(profile: Dict[str, Any]) -> str:
    similarity_profile = build_similarity_profile(profile)
    canonical = json.dumps(similarity_profile, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_cache(profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    request_hash = make_similarity_hash(profile)

    response = (
        supabase.table("recommendation_cache")
        .select("*")
        .eq("request_hash", request_hash)
        .limit(1)
        .execute()
    )
    rows = response.data or []
    if not rows:
        return None

    row = rows[0]

    try:
        current = to_int(row.get("hit_count"), 1)
        (
            supabase.table("recommendation_cache")
            .update({"hit_count": current + 1})
            .eq("id", row["id"])
            .execute()
        )
    except Exception:
        pass

    return row.get("response_json")


def save_cache(profile: Dict[str, Any], response_json: Dict[str, Any], model_name: str) -> None:
    similarity_profile = build_similarity_profile(profile)
    request_hash = make_similarity_hash(profile)

    payload = {
        "request_hash": request_hash,
        "normalized_profile": similarity_profile,
        "response_json": response_json,
        "llm_provider": "groq",
        "llm_model": model_name,
        "expense_bucket": similarity_profile["expense_bucket"],
        "annual_fee_bucket": similarity_profile["annual_fee_bucket"],
        "credit_score_bucket": similarity_profile["credit_score_bucket"],
        "expense_type": similarity_profile["expense_type"],
        "desired_output": similarity_profile["desired_output"],
    }

    (
        supabase.table("recommendation_cache")
        .upsert(payload, on_conflict="request_hash")
        .execute()
    )


# =========================================================
# DB FETCH
# =========================================================
def fetch_eligible_cards(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    response = (
        supabase.table("credit_cards")
        .select("*")
        .eq("is_active", True)
        .lte("annual_fee", profile["max_annual_fee"])
        .execute()
    )

    rows = response.data or []
    filtered = []
    for row in rows:
        min_cs = row.get("min_credit_score")
        if min_cs is not None and to_int(min_cs, 0) > profile["credit_score"]:
            continue
        filtered.append(row)
    return filtered


def fetch_benefit_facts(card_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    if not card_ids:
        return {}

    response = (
        supabase.table("card_benefit_facts")
        .select("*")
        .in_("card_id", card_ids)
        .execute()
    )
    rows = response.data or []

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["card_id"], []).append(row)
    return grouped


def fetch_chunks_for_cards(card_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    if not card_ids:
        return {}

    response = (
        supabase.table("card_chunks")
        .select("id, card_id, document_id, section, heading, chunk_text, page_no, metadata")
        .in_("card_id", card_ids)
        .execute()
    )
    rows = response.data or []

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["card_id"], []).append(row)
    return grouped


# =========================================================
# RAG RETRIEVAL
# =========================================================
def get_query_terms(profile: Dict[str, Any]) -> List[str]:
    base_terms = [
        profile["expense_type"],
        profile["desired_output"],
        "annual fee",
        "annual fee waiver",
        "fee waiver",
        "reward",
        "benefit",
        "cashback",
        "points",
        "lounge",
        "travel",
        "hotel",
        "dining",
        "movie",
        "fuel",
        "forex",
    ]

    for x in USER_INTENT_SYNONYMS.get(profile["expense_type"], []):
        base_terms.append(x)
    for x in USER_INTENT_SYNONYMS.get(profile["desired_output"], []):
        base_terms.append(x)

    seen = set()
    final_terms = []
    for t in base_terms:
        t = normalize_text(t)
        if t and t not in seen:
            seen.add(t)
            final_terms.append(t)
    return final_terms


def score_chunk_relevance(chunk: Dict[str, Any], profile: Dict[str, Any]) -> float:
    section = normalize_text(chunk.get("section"))
    text = " ".join([
        normalize_text(chunk.get("heading")),
        normalize_text(chunk.get("section")),
        normalize_text(chunk.get("chunk_text")),
    ])

    score = SECTION_BOOSTS.get(section, 0.0)

    for term in get_query_terms(profile):
        if term and term in text:
            score += 2.5

    if section == profile["desired_output"]:
        score += 8.0
    if section == profile["expense_type"]:
        score += 6.0
    if profile["desired_output"] in text:
        score += 6.0
    if profile["expense_type"] in text:
        score += 5.0

    if "complimentary" in text:
        score += 1.5
    if "milestone" in text:
        score += 1.5
    if "priority pass" in text:
        score += 2.0
    if "bookmyshow" in text:
        score += 2.0

    if len(text) < 40:
        score -= 2.0

    return round(score, 2)


def retrieve_top_chunks_per_card(
    all_chunks_by_card: Dict[str, List[Dict[str, Any]]],
    profile: Dict[str, Any],
    top_k: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    result = {}

    for card_id, chunks in all_chunks_by_card.items():
        scored = []
        for ch in chunks:
            item = dict(ch)
            item["relevance_score"] = score_chunk_relevance(ch, profile)
            scored.append(item)

        scored.sort(key=lambda x: x["relevance_score"], reverse=True)
        top = scored[:top_k]

        fees_chunk = next((x for x in scored if normalize_text(x.get("section")) == "fees"), None)
        if fees_chunk and not any(x["id"] == fees_chunk["id"] for x in top):
            if top:
                top[-1] = fees_chunk
            else:
                top = [fees_chunk]

        top.sort(key=lambda x: x["relevance_score"], reverse=True)
        result[card_id] = top

    return result


# =========================================================
# SCORING
# =========================================================
def basic_constraint_fit(card: Dict[str, Any], profile: Dict[str, Any]) -> float:
    score = 0.0
    annual_fee = to_float(card.get("annual_fee"), 0.0)
    max_fee = profile["max_annual_fee"]

    if annual_fee <= max_fee:
        ratio = annual_fee / max_fee if max_fee > 0 else 0
        if annual_fee == 0:
            score += 20
        elif ratio <= 0.25:
            score += 18
        elif ratio <= 0.50:
            score += 14
        elif ratio <= 0.75:
            score += 10
        else:
            score += 6

    min_cs = card.get("min_credit_score")
    if min_cs is not None:
        gap = profile["credit_score"] - to_int(min_cs)
        if gap >= 100:
            score += 10
        elif gap >= 50:
            score += 8
        elif gap >= 0:
            score += 5

    return score


def structured_reward_fit(card: Dict[str, Any], profile: Dict[str, Any]) -> float:
    desired = profile["desired_output"]
    expense = profile["expense_type"]

    reward_types = normalize_reward_values(extract_reward_types(card))
    key_benefits = normalize_text(card.get("key_benefits", ""))
    eligibility_text = normalize_text(card.get("eligibility_text", ""))

    score = 0.0
    if desired in reward_types:
        score += 20
    if expense in reward_types:
        score += 12
    if desired in key_benefits:
        score += 14
    if expense in key_benefits:
        score += 10
    if desired in eligibility_text:
        score += 4

    if desired == "lounge":
        score += min(
            to_int(card.get("lounge_domestic"), 0) * 1.5 +
            to_int(card.get("lounge_international"), 0) * 3.0,
            12
        )

    if desired in {"travel", "hotel", "forex"}:
        fx = card.get("forex_markup_pct")
        if fx is not None:
            f = to_float(fx, 10)
            if f <= 0:
                score += 10
            elif f <= 1:
                score += 8
            elif f <= 2:
                score += 5
            elif f <= 3.5:
                score += 2

    return round(score, 2)


def benefit_fact_fit(benefits: List[Dict[str, Any]], profile: Dict[str, Any]) -> float:
    desired = profile["desired_output"]
    expense = profile["expense_type"]
    score = 0.0

    for b in benefits:
        benefit_type = REWARD_NORMALIZATION.get(normalize_text(b.get("benefit_type")), normalize_text(b.get("benefit_type")))
        benefit_value = REWARD_NORMALIZATION.get(normalize_text(b.get("benefit_value")), normalize_text(b.get("benefit_value")))

        text = " ".join([
            benefit_type,
            benefit_value,
            normalize_text(b.get("benefit_unit")),
            normalize_text(b.get("condition_text")),
        ])

        if desired and desired in text:
            score += 4.0
        if expense and expense in text:
            score += 3.0

        for sp in ["cashback", "reward points", "dining", "movie", "lounge", "travel", "hotel", "fuel"]:
            if sp in text:
                score += 1.0

    return round(min(score, 20.0), 2)


def retrieval_fit(top_chunks: List[Dict[str, Any]], profile: Dict[str, Any]) -> Tuple[float, List[str]]:
    desired = profile["desired_output"]
    expense = profile["expense_type"]

    score = 0.0
    evidence_points = []

    for ch in top_chunks:
        rel = to_float(ch.get("relevance_score"), 0.0)
        text = normalize_text(ch.get("chunk_text"))
        section = normalize_text(ch.get("section"))

        score += min(rel, 8.0)

        if desired in text or section == desired:
            evidence_points.append(f"Mentions {desired} benefit")
        if expense in text or section == expense:
            evidence_points.append(f"Mentions {expense} use case")
        if "annual fee waiver" in text:
            evidence_points.append("Contains annual fee waiver detail")
        if "complimentary" in text:
            evidence_points.append("Contains complimentary benefit")
        if "milestone" in text:
            evidence_points.append("Contains milestone/spend condition")

    clean = []
    seen = set()
    for e in evidence_points:
        if e not in seen:
            seen.add(e)
            clean.append(e)

    return round(min(score, 30.0), 2), clean[:5]


def estimate_annual_value(card: Dict[str, Any], profile: Dict[str, Any], top_chunks: List[Dict[str, Any]]) -> float:
    annual_spend = profile["annual_expense"]
    desired = profile["desired_output"]
    annual_fee = to_float(card.get("annual_fee"), 0.0)

    reward_rate = 0.005
    text_blob = " ".join(
        [normalize_text(card.get("key_benefits", ""))] +
        [normalize_text(ch.get("chunk_text")) for ch in top_chunks]
    )

    if desired == "cashback":
        reward_rate = 0.01
    elif desired in {"shopping", "dining", "movies"}:
        reward_rate = 0.008
    elif desired in {"travel", "hotel"}:
        reward_rate = 0.012
    elif desired == "fuel":
        reward_rate = 0.007

    if "accelerated" in text_blob:
        reward_rate += 0.003
    if "cashback" in text_blob:
        reward_rate += 0.002
    if "reward points" in text_blob:
        reward_rate += 0.002

    estimated_rewards = annual_spend * reward_rate

    lounge_value = 0.0
    if desired in {"travel", "lounge", "hotel"}:
        lounge_value = (
            to_int(card.get("lounge_domestic"), 0) * 500 +
            to_int(card.get("lounge_international"), 0) * 1200
        )
        lounge_value = min(lounge_value, 8000)

    return round(estimated_rewards + lounge_value - annual_fee, 2)


def score_card_v2(
    card: Dict[str, Any],
    benefit_facts: List[Dict[str, Any]],
    top_chunks: List[Dict[str, Any]],
    profile: Dict[str, Any]
) -> Dict[str, Any]:
    structured_score = basic_constraint_fit(card, profile) + structured_reward_fit(card, profile)
    benefit_score = benefit_fact_fit(benefit_facts, profile)
    retrieval_score, evidence_points = retrieval_fit(top_chunks, profile)
    estimated_value = estimate_annual_value(card, profile, top_chunks)

    value_score = 0.0
    if estimated_value >= 10000:
        value_score = 20.0
    elif estimated_value >= 5000:
        value_score = 15.0
    elif estimated_value >= 2000:
        value_score = 10.0
    elif estimated_value >= 0:
        value_score = 5.0

    total_score = round(structured_score + benefit_score + retrieval_score + value_score, 2)

    missing_data_notes = []
    if not benefit_facts:
        missing_data_notes.append("No structured benefit facts found")
    if not top_chunks:
        missing_data_notes.append("No supporting chunks retrieved")
    if card.get("forex_markup_pct") is None and profile["desired_output"] in {"travel", "hotel", "forex"}:
        missing_data_notes.append("Forex markup missing for travel-like use case")
    if not card.get("annual_fee_waiver_text"):
        missing_data_notes.append("Annual fee waiver detail missing in structured fields")

    return {
        "card_id": card["id"],
        "card_name": card.get("card_name"),
        "issuer": card.get("issuer"),
        "network": card.get("network"),
        "card_type": card.get("card_type"),
        "annual_fee": to_float(card.get("annual_fee"), 0.0),
        "joining_fee": to_float(card.get("joining_fee"), 0.0),
        "annual_fee_waiver_text": card.get("annual_fee_waiver_text"),
        "forex_markup_pct": card.get("forex_markup_pct"),
        "lounge_domestic": to_int(card.get("lounge_domestic"), 0),
        "lounge_international": to_int(card.get("lounge_international"), 0),
        "reward_type": normalize_reward_values(extract_reward_types(card)),
        "key_benefits": card.get("key_benefits"),
        "official_url": card.get("official_url"),
        "score_breakdown": {
            "structured_score": round(structured_score, 2),
            "benefit_score": round(benefit_score, 2),
            "retrieval_score": round(retrieval_score, 2),
            "value_score": round(value_score, 2),
        },
        "estimated_annual_value": estimated_value,
        "evidence_points": evidence_points,
        "missing_data_notes": missing_data_notes,
        "top_chunks_used": top_chunks[:5],
        "benefit_facts_used": benefit_facts[:10],
        "total_score": total_score,
    }


# =========================================================
# GROQ EXPLANATION
# =========================================================
def build_llm_card_summary(scored_card: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "card_id": scored_card["card_id"],
        "card_name": scored_card["card_name"],
        "issuer": scored_card["issuer"],
        "network": scored_card["network"],
        "card_type": scored_card["card_type"],
        "annual_fee": scored_card["annual_fee"],
        "joining_fee": scored_card["joining_fee"],
        "annual_fee_waiver_text": scored_card["annual_fee_waiver_text"],
        "forex_markup_pct": scored_card["forex_markup_pct"],
        "lounge_domestic": scored_card["lounge_domestic"],
        "lounge_international": scored_card["lounge_international"],
        "reward_type": scored_card["reward_type"],
        "key_benefits": scored_card["key_benefits"],
        "estimated_annual_value": scored_card["estimated_annual_value"],
        "total_score": scored_card["total_score"],
        "score_breakdown": scored_card["score_breakdown"],
        "evidence_points": scored_card["evidence_points"],
        "missing_data_notes": scored_card["missing_data_notes"],
        "benefit_facts": [
            {
                "benefit_type": b.get("benefit_type"),
                "benefit_value": b.get("benefit_value"),
                "benefit_unit": b.get("benefit_unit"),
                "condition_text": b.get("condition_text"),
            }
            for b in scored_card.get("benefit_facts_used", [])[:6]
        ],
        "retrieved_evidence": [
            {
                "section": c.get("section"),
                "heading": c.get("heading"),
                "chunk_text": c.get("chunk_text"),
                "relevance_score": c.get("relevance_score"),
            }
            for c in scored_card.get("top_chunks_used", [])[:4]
        ],
    }


def groq_compare_cards_v2(profile: Dict[str, Any], scored_cards: List[Dict[str, Any]]) -> Dict[str, Any]:
    llm_cards = [build_llm_card_summary(c) for c in scored_cards]

    system_prompt = """
You are a credit card recommendation assistant.

You will receive:
1. A user profile
2. Structured card facts
3. Retrieved evidence from card offering text
4. Deterministic backend scores

Rules:
- Use only the provided data.
- Do not invent benefits.
- Prefer cards supported by both structured data and retrieved evidence.
- If data is missing, say that clearly.
- Respect the user's fee budget, credit score, expense style, and desired reward output.
- The backend score is important, but you should still mention tradeoffs.

Return strict JSON only in this schema:
{
  "recommended_card": {
    "card_id": "string",
    "card_name": "string",
    "issuer": "string",
    "confidence": "high|medium|low",
    "why_it_wins": ["string", "string", "string"],
    "tradeoffs": ["string", "string"]
  },
  "runner_ups": [
    {
      "card_id": "string",
      "card_name": "string",
      "issuer": "string",
      "why_consider": ["string", "string"]
    }
  ],
  "decision_summary": "string",
  "missing_data_notes": ["string", "string"],
  "fit_analysis": {
    "expense_type_fit": "string",
    "desired_output_fit": "string",
    "fee_fit": "string",
    "eligibility_fit": "string"
  }
}
""".strip()

    user_prompt = {
        "user_profile": profile,
        "ranked_cards": llm_cards,
        "instructions": [
            "Choose one best card.",
            "Use retrieved text evidence as support.",
            "Prefer stronger evidence-backed matches.",
            "Mention uncertainty where data is incomplete.",
            "Do not overclaim unsupported benefits."
        ]
    }

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, default=str)},
        ],
    )

    raw_response = completion.choices[0].message.content
    parsed = json.loads(raw_response)

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw_response": raw_response,
        "parsed_response": parsed,
    }


# =========================================================
# MAIN PIPELINE
# =========================================================
def recommend_credit_card_v2(payload: RecommendationRequest) -> Dict[str, Any]:
    profile = normalize_user_profile(payload)
    similarity_profile = build_similarity_profile(profile)

    if payload.use_cache:
        cached = get_cache(profile)
        if cached:
            return {
                "source": "similar_profile_cache",
                "profile": profile,
                "similarity_profile": similarity_profile,
                "result": cached,
            }

    eligible_cards = fetch_eligible_cards(profile)

    if not eligible_cards:
        response_json = {
            "recommended_by": "none",
            "cache_metadata": {
                "generated_from_profile": profile,
                "similarity_profile": similarity_profile,
            },
            "user_profile": profile,
            "final_decision": {
                "recommended_card": None,
                "runner_ups": [],
                "decision_summary": "No eligible cards found for this fee budget and credit score.",
                "missing_data_notes": [],
                "fit_analysis": {
                    "expense_type_fit": "No eligible cards available",
                    "desired_output_fit": "No eligible cards available",
                    "fee_fit": "Budget may be too restrictive",
                    "eligibility_fit": "Credit score may be below available card thresholds"
                }
            }
        }
        save_cache(profile, response_json, GROQ_MODEL)
        return {
            "source": "live",
            "profile": profile,
            "similarity_profile": similarity_profile,
            "result": response_json,
        }

    card_ids = [c["id"] for c in eligible_cards]
    benefit_map = fetch_benefit_facts(card_ids)
    all_chunks = fetch_chunks_for_cards(card_ids)
    top_chunk_map = retrieve_top_chunks_per_card(all_chunks, profile, top_k=5)

    scored_cards = []
    for card in eligible_cards:
        cid = card["id"]
        scored_cards.append(
            score_card_v2(
                card=card,
                benefit_facts=benefit_map.get(cid, []),
                top_chunks=top_chunk_map.get(cid, []),
                profile=profile
            )
        )

    scored_cards.sort(key=lambda x: x["total_score"], reverse=True)
    shortlisted = scored_cards[:5]
    groq_result = groq_compare_cards_v2(profile, shortlisted)

    final_response = {
        "recommended_by": "hybrid_rag_plus_rules_plus_groq",
        "cache_metadata": {
            "generated_from_profile": profile,
            "similarity_profile": similarity_profile,
        },
        "user_profile": profile,
        "ranked_cards": shortlisted,
        "llm_debug": {
            "system_prompt": groq_result["system_prompt"],
            "user_prompt": groq_result["user_prompt"],
            "raw_response": groq_result["raw_response"],
        } if payload.debug else None,
        "final_decision": groq_result["parsed_response"],
    }

    save_cache(profile, final_response, GROQ_MODEL)

    return {
        "source": "live",
        "profile": profile,
        "similarity_profile": similarity_profile,
        "result": final_response,
    }


# =========================================================
# ROUTES
# =========================================================
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/recommend")
def recommend(payload: RecommendationRequest):
    try:
        return recommend_credit_card_v2(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================
# LOCAL RUN
# =========================================================
# Run with:
# uvicorn app:app --host 0.0.0.0 --port 8000 --reload
