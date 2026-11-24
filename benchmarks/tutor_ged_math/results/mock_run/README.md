# Mock Run Results

**⚠️ THESE ARE MOCK RESULTS - NOT REAL API CALLS**

This directory contains output from a **mock benchmark run** using simulated API responses.

## What This Is

- **Purpose:** Demonstrate the harness output format
- **API Calls:** Zero real API calls were made
- **Token Counts:** Estimated using realistic mock client
- **Costs:** Based on estimated tokens (not actual usage)
- **Responses:** Mock/scripted text, not real AI output

## Files

- `raw_results.json` - Full mock results with metadata
- `grading_sheet.csv` - Blinded export format (40 mock lessons)
- `arm_key.json` - Arm mapping (Arm_A_AIHive → X, Arm_B_Generic → Y)
- `cost_breakdown.csv` - Per-lesson cost estimates

## Do Not Use For

❌ **Do not use these results for any evaluation or analysis**  
❌ **Do not cite these as real benchmark results**  
❌ **Do not grade these responses**

## Real Results

To generate real results:
```bash
export OPENAI_API_KEY="your-key"
python3 benchmarks/tutor_ged_math/harness/runner.py \
  --confirm-live --max-budget-usd 1.0 \
  --output-dir benchmarks/tutor_ged_math/results/live_run_YYYYMMDD
```

Real results will have:
- Actual API token counts from response.usage
- Real response text from the models
- Accurate cost calculation
- Measurable latency

---

**Created:** 2026-11-24  
**Run Mode:** MOCK (no real API calls)
