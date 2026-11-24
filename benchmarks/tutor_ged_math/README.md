# AIHive Tutor GED Math Benchmark

A controlled learning-quality benchmark for the AIHive Tutor agent using 20 original GED-style math questions.

## Purpose

Evaluate AI tutoring systems on **single-session response quality** across five dimensions:
1. Mathematical correctness
2. Explanation clarity
3. Misconception diagnosis
4. Adaptive practice questions
5. Cost per lesson

This measures response quality as a proxy for educational effectiveness, not actual learning gains.

## Structure

```
benchmarks/tutor_ged_math/
├── README.md                           # This file
├── items/
│   ├── questions.py                    # 20 original GED-style questions
│   └── test_answer_keys.py            # Verification tests for all answers
├── rubric/
│   ├── scoring_rubric.md              # Human-readable rubric (frozen)
│   └── rubric.py                       # Machine-readable rubric
├── harness/
│   ├── lesson_protocol.py             # 3-turn lesson protocol
│   ├── tutor_arms.py                  # Arm implementations
│   ├── runner.py                       # Main benchmark runner
│   └── test_harness.py                # Harness tests
├── results/                            # Output directory (created on run)
│   ├── raw_results.json               # Full results with tokens/cost
│   ├── grading_sheet.csv              # Blinded for scoring
│   ├── arm_key.json                   # Arm mapping (keep secret)
│   └── cost_breakdown.csv             # Detailed cost metrics
├── SOFTWARE_TEST_RESULTS.md           # Software test report
└── EDUCATIONAL_EFFECTIVENESS.md       # Effectiveness report (template)
```

## Questions

20 original GED-style math questions covering:
- **Domain distribution:** ~45% quantitative problem solving, ~55% algebraic problem solving (per GED specs)
- **Difficulty mix:** Easy, medium, and hard
- **Format mix:** Multiple-choice and fill-in
- **Coverage:** Arithmetic, percentages, fractions, algebra, geometry, probability, etc.

Each question includes:
- Verified correct answer (with computational proof)
- Worked solution
- At least 2 predefined misconceptions with scripted student attempts

**Note:** A qualified mathematics educator should review these items before publishing results.

## Lesson Protocol

Each lesson = 1 question = 3 turns (fresh context):

1. **Turn 1:** "Please solve this problem and explain your solution"
2. **Turn 2:** Student submits scripted wrong attempt → "What did I do wrong?"
3. **Turn 3:** "Can you give me 2 practice questions to help me understand this better?"

## Arms

### Arm A: AIHive Tutor
- Uses real Tutor implementation from `tutor/tutor_ai.py`
- Calls app's own functions (no reimplementation)
- Model: `gpt-4o-mini` (default from config)
- Temperature: 0.7
- Max tokens: 500 for lessons, 200-300 for summaries

### Arm B: Generic Assistant
- Same model (`gpt-4o-mini`)
- Same temperature (0.7)
- Same max tokens
- Generic prompt: "You are a helpful assistant."

### Arm C: Scripted Demo (Optional)
- AIHive Tutor in demo mode (no API)
- Labeled as scripted baseline, not AI
- Useful for cost comparison

## Running the Benchmark

### Mock Mode (Default - No API Calls)

```bash
cd /agent/repos/App-maker-App-maker
PYTHONPATH=.:./aihive_coach python benchmarks/tutor_ged_math/harness/runner.py
```

### Live Mode (Real API Calls)

**Prerequisites:**
1. Set `OPENAI_API_KEY` environment variable
2. Confirm budget and safety flags

```bash
export OPENAI_API_KEY="your-key-here"

# Estimate cost first
python benchmarks/tutor_ged_math/harness/runner.py --confirm-live --max-budget-usd 5.0

# If estimate looks good, it will run automatically
```

**Safety features:**
- Refuses to run without `--confirm-live` and `--max-budget-usd`
- Pre-estimates cost and aborts if over budget
- Never prints or logs API key
- Uses mock client by default

### Options

```bash
--output-dir DIR          # Output directory (default: benchmarks/tutor_ged_math/results)
--confirm-live            # Confirm real API calls (required for live mode)
--max-budget-usd FLOAT    # Max spending limit (required for live mode)
--random-seed INT         # Random seed (default: 42)
--include-arm-c           # Include scripted demo baseline
--rubric-commit HASH      # Git commit hash of rubric
```

## Scoring

Uses predefined rubric (`rubric/scoring_rubric.md`) committed **before** collecting results.

**Blinded grading:**
1. Harness exports responses to `grading_sheet.csv` with arm labels removed (A/B → X/Y)
2. Order shuffled with recorded seed
3. Arm mapping saved separately in `arm_key.json`
4. Grader scores responses using rubric without knowing which system produced them
5. Merge mapping with scores after grading complete

**Optional:** LLM-as-judge mode (behind flag, OFF by default, labeled preliminary)

## Cost Tracking

All runs record:
- Input/output tokens per turn
- API latency
- Cost calculation using official pricing
- Price source URL and verification date

Prices are from official OpenAI pricing page (verified 2026-11-24).

## Testing

### Verify Answer Keys

```bash
PYTHONPATH=.:./aihive_coach python -m pytest benchmarks/tutor_ged_math/items/test_answer_keys.py -v
```

All 20 questions have computational verification that:
- Correct answer is mathematically correct
- Misconception answers are wrong
- For MC, exactly one choice is correct

### Test Harness

```bash
PYTHONPATH=.:./aihive_coach python -m pytest benchmarks/tutor_ged_math/harness/test_harness.py -v
```

Tests mock end-to-end run, cost estimation, blinding, etc.

### Existing Tutor Tests

```bash
PYTHONPATH=.:./aihive_coach python -m pytest tutor/tests/test_tutor_ai.py -v
```

## Reports

### SOFTWARE_TEST_RESULTS.md
- Harness unit tests (pass/fail counts)
- Answer key verification (all 20 questions)
- Mock end-to-end run results
- Existing Tutor test suite results

### EDUCATIONAL_EFFECTIVENESS.md
- Template stating no results yet
- Requires live run + blinded grading
- States limitations (single-session proxy, not actual learning gains)

## Important Limitations

**This benchmark measures single-session response quality, NOT actual learning effectiveness.**

Actual learning requires:
- Multiple sessions over time
- Pre-test and post-test with learners
- Transfer to novel problems
- Retention measurement
- Learner satisfaction surveys

## Pricing Information

Prices verified from official OpenAI pricing page on 2026-11-24:
- **gpt-4o-mini:** $0.15 per 1M input tokens, $0.60 per 1M output tokens  
- **Source:** https://platform.openai.com/docs/pricing
- **Verified by:** Founder's assistant (Asia/Bangkok timezone)

## Cost Estimate (Worst-Case Upper Bound)

For **A1 + A2 + B** (3 arms, 20 lessons each):

**Repeats = 1 (60 lessons total):**
- Input: 26,050 tokens (measured with full history)
- Output: 108,000 tokens (sum of all max_tokens)
- **Cost: $0.07**

**Repeats = 3 (180 lessons total):**
- Input: 78,150 tokens
- Output: 324,000 tokens
- **Cost: $0.21**

**Method:**
- Output = sum of max_tokens for all calls (worst case)
- Input = measured prompt tokens including system prompts and history
- History sized as if every previous reply used full max_tokens
- Uses verified gpt-4o-mini prices ($0.15/$0.60 per 1M)

## License

This benchmark is part of the AIHiveHQ project. Questions are original work created specifically for this benchmark.

## Author

Created: 2026-11-24  
Rubric Version: 1.0
