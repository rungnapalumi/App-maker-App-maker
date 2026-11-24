"""
Machine-readable scoring rubric for GED Math Benchmark
"""

from typing import Dict, List, Literal
from pydantic import BaseModel


class RubricDimension(BaseModel):
    """A scoring dimension with anchor descriptions"""
    name: str
    max_score: int
    anchors: Dict[int, str]
    auto_scorable: bool = False


class CostMetrics(BaseModel):
    """Cost and performance metrics for a lesson"""
    input_tokens: int
    output_tokens: int
    model_id: str
    total_latency_ms: int
    cost_usd: float
    price_source: str
    price_verified: bool


class LessonScore(BaseModel):
    """Scores for a single lesson"""
    lesson_id: str
    mathematical_correctness: int  # 0-2
    explanation_clarity: int  # 0-4
    misconception_diagnosis: int  # 0-4
    adaptive_practice: int  # 0-4
    cost_metrics: CostMetrics
    
    @property
    def quality_score(self) -> int:
        """Sum of scored dimensions (max 14)"""
        return (
            self.mathematical_correctness +
            self.explanation_clarity +
            self.misconception_diagnosis +
            self.adaptive_practice
        )


# Rubric definition
RUBRIC_VERSION = "1.0"
RUBRIC_DATE = "2026-11-24"

DIMENSIONS: List[RubricDimension] = [
    RubricDimension(
        name="mathematical_correctness",
        max_score=2,
        auto_scorable=True,
        anchors={
            0: "Final answer is incorrect",
            1: "Final answer correct but solution has errors",
            2: "Final answer correct AND all steps valid"
        }
    ),
    RubricDimension(
        name="explanation_clarity",
        max_score=4,
        auto_scorable=False,
        anchors={
            0: "Incomprehensible - jargon, no structure, missing steps",
            1: "Poor - too technical or simple, confusing, missing examples",
            2: "Fair - mostly clear but has issues (some jargon, logical gaps)",
            3: "Good - clear, organized, appropriate vocabulary, helpful examples",
            4: "Excellent - exceptionally clear, perfect level, strong analogies"
        }
    ),
    RubricDimension(
        name="misconception_diagnosis",
        max_score=4,
        auto_scorable=False,
        anchors={
            0: "No diagnosis - generic encouragement only",
            1: "Vague diagnosis - acknowledges error but doesn't pinpoint",
            2: "Partial diagnosis - identifies general area, not specific misconception",
            3: "Accurate diagnosis - identifies specific predefined misconception",
            4: "Accurate + educational - identifies misconception AND explains why"
        }
    ),
    RubricDimension(
        name="adaptive_practice",
        max_score=4,
        auto_scorable=False,
        anchors={
            0: "Inappropriate or missing - no questions or off-topic",
            1: "Generic - doesn't target misconception, wrong difficulty",
            2: "Somewhat targeted - same skill but not addressing misconception",
            3: "Well-targeted - directly practices misconception, appropriate difficulty",
            4: "Excellent - perfectly targets misconception, progressive difficulty"
        }
    )
]


# Official pricing verified from provider documentation
MODEL_PRICING = {
    "gpt-4o-mini": {
        "input_price_per_1m": 0.15,  # $0.15 per 1M input tokens
        "output_price_per_1m": 0.60,  # $0.60 per 1M output tokens
        "cached_input_price_per_1m": 0.075,  # $0.075 per 1M cached input tokens
        "source": "https://platform.openai.com/docs/pricing",
        "last_verified": "2026-11-24",
        "verified_by": "founder's assistant",
        "timezone": "Asia/Bangkok",
        "verified": True
    },
    "gpt-4o": {
        "input_price_per_1m": 2.50,  # $2.50 per 1M input tokens (not verified)
        "output_price_per_1m": 10.00,  # $10.00 per 1M output tokens (not verified)
        "source": "https://platform.openai.com/docs/pricing (not verified for gpt-4o)",
        "last_verified": None,
        "verified": False
    },
    # Placeholder for other models
    "unknown": {
        "input_price_per_1m": 0.0,
        "output_price_per_1m": 0.0,
        "source": "UNVERIFIED",
        "last_verified": None,
        "verified": False
    }
}


def calculate_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int
) -> tuple[float, str, bool]:
    """
    Calculate cost for a lesson.
    Returns: (cost_usd, price_source, verified)
    """
    pricing = MODEL_PRICING.get(model_id, MODEL_PRICING["unknown"])
    
    cost = (
        (input_tokens * pricing["input_price_per_1m"]) +
        (output_tokens * pricing["output_price_per_1m"])
    ) / 1_000_000
    
    return cost, pricing["source"], pricing["verified"]


def get_max_quality_score() -> int:
    """Maximum possible quality score (sum of all dimension maxes)"""
    return sum(d.max_score for d in DIMENSIONS if d.name != "cost")


def validate_scores(lesson_score: LessonScore) -> List[str]:
    """
    Validate that scores are within rubric bounds.
    Returns list of error messages (empty if valid).
    """
    errors = []
    
    for dim in DIMENSIONS:
        score = getattr(lesson_score, dim.name)
        if not (0 <= score <= dim.max_score):
            errors.append(
                f"{dim.name} score {score} out of range [0, {dim.max_score}]"
            )
    
    return errors
