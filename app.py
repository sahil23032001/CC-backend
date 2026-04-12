import os
import json
import hashlib
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Header
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
FRONTEND_PROXY_SECRET = os.getenv("FRONTEND_PROXY_SECRET")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "https://your-frontend.vercel.app")

# Make environment check more lenient for initial deployment
supabase: Optional[Client] = None
groq_client: Optional[Groq] = None

if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)


# =========================================================
# APP
# =========================================================
app = FastAPI(title="Credit Card Recommender API", version="1.0.0")

# More permissive CORS for Railway deployment
allowed_origins = [FRONTEND_ORIGIN]
if os.getenv("RAILWAY_PUBLIC_DOMAIN"):
    railway_url = f"https://{os.getenv('RAILWAY_PUBLIC_DOMAIN')}"
    allowed_origins.append(railway_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins + ["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# =========================================================
# HEALTH CHECK ENDPOINTS (CRITICAL FOR RAILWAY)
# =========================================================
@app.get("/")
async def root():
    """Root endpoint - Railway uses this to check if app is alive"""
    return {
        "service": "Credit Card Recommender API",
        "status": "running",
        "version": "1.0.0",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    health_status = {
        "status": "healthy",
        "service": "Credit Card Recommender API",
        "version": "1.0.0",
        "checks": {
            "supabase": "connected" if supabase else "not_configured",
            "groq": "connected" if groq_client else "not_configured",
            "environment": {
                "SUPABASE_URL": "set" if SUPABASE_URL else "missing",
                "SUPABASE_KEY": "set" if SUPABASE_KEY else "missing",
                "GROQ_API_KEY": "set" if GROQ_API_KEY else "missing",
                "FRONTEND_PROXY_SECRET": "set" if FRONTEND_PROXY_SECRET else "missing",
            }
        }
    }
    return health_status


# =========================================================
# REQUEST MODEL
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
# CONSTANTS
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


# =========================================================
# HELPERS
# =========================================================
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
    """Fixed function - categorize credit score into buckets"""
    score = int(score)
    if score >= 750:
        return "excellent"
    elif score >= 700:
        return "good"
    elif score >= 650:
        return "fair"
    else:
        return "poor"


# =========================================================
# PLACEHOLDER RECOMMENDATION ENDPOINT
# =========================================================
@app.post("/recommend")
async def recommend_cards(
    request: RecommendationRequest,
    x_frontend_secret: Optional[str] = Header(None)
):
    """
    Placeholder recommendation endpoint
    Replace this with your full logic once deployment works
    """
    # Verify secret if configured
    if FRONTEND_PROXY_SECRET and x_frontend_secret != FRONTEND_PROXY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid or missing frontend secret")
    
    # Check if services are configured
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    if not groq_client:
        raise HTTPException(status_code=503, detail="Groq API not configured")
    
    # Normalize user profile
    profile = normalize_user_profile(request)
    
    # Return a placeholder response
    return {
        "status": "success",
        "message": "API is working! Add your full recommendation logic here.",
        "profile": profile,
        "recommendations": [
            {
                "card_name": "Example Card",
                "score": 95.0,
                "reason": "This is a placeholder response"
            }
        ]
    }


# =========================================================
# STARTUP EVENT
# =========================================================
@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    print("=" * 60)
    print("Credit Card Recommender API Starting...")
    print(f"Supabase: {'✓ Connected' if supabase else '✗ Not configured'}")
    print(f"Groq: {'✓ Connected' if groq_client else '✗ Not configured'}")
    print(f"Frontend Origin: {FRONTEND_ORIGIN}")
    print("=" * 60)
