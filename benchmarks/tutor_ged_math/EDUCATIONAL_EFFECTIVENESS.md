# Educational Effectiveness Report - AIHive Tutor GED Math Benchmark

**Date:** 2026-11-24  
**Benchmark Version:** 1.0  
**Status:** 🔄 RESULTS PENDING

---

## Executive Summary

**NO RESULTS AVAILABLE YET**

This benchmark requires:
1. ✅ Live run with real API calls → **PENDING** (mock run completed)
2. ⏳ Blinded human grading using rubric → **NOT STARTED**
3. ⏳ Statistical analysis → **NOT STARTED**

---

## What This Benchmark Measures

This benchmark evaluates **single-session response quality** as a proxy for educational effectiveness. It compares:

- **Arm A:** AIHive Tutor (real implementation with `gpt-4o-mini`)
- **Arm B:** Generic AI Assistant (same model, generic prompt)
- **Arm C:** Scripted Demo (optional baseline, non-AI)

Across five dimensions:
1. Mathematical correctness (0-2 points)
2. Explanation clarity (0-4 points)
3. Misconception diagnosis (0-4 points)
4. Adaptive practice questions (0-4 points)
5. Cost per lesson (descriptive)

**Maximum quality score:** 14 points per lesson

---

## Important Limitations

### This Benchmark Does NOT Measure Actual Learning

**What we measure:**
- Single-session response quality
- Error diagnosis accuracy
- Explanation clarity for GED-level readers
- Practice question targeting

**What we DO NOT measure:**
- Actual learning gains over time
- Retention after days/weeks
- Transfer to novel problems
- Long-term student engagement
- Real learner satisfaction

### Real Learning Effectiveness Would Require:

1. **Pre-test and post-test with actual learners**
   - Baseline knowledge assessment
   - Learning intervention (multiple sessions)
   - Post-test on same skills
   - Retention test after delay

2. **Control group comparison**
   - Random assignment to conditions
   - Matched difficulty and time
   - Blinded assessment

3. **Transfer assessment**
   - Novel problems not seen during training
   - Application to new contexts

4. **Longitudinal tracking**
   - Multiple sessions over weeks
   - Spaced practice
   - Forgetting curve measurement

5. **Learner feedback**
   - Satisfaction surveys
   - Perceived usefulness
   - Motivation and engagement
   - Preferred learning style match

**This benchmark is a quality proxy only.** It measures whether responses are pedagogically sound, not whether students actually learn.

---

## Methodology

### Questions
- 20 original GED-style math questions
- ~45% quantitative, ~55% algebraic (per GED specs)
- Mix of easy (6), medium (10), hard (4)
- Multiple-choice (9) and fill-in (11)
- Each with verified answer key and predefined misconceptions

### Protocol
Each lesson = 1 question = 3 turns (fresh context):
1. Turn 1: "Solve and explain"
2. Turn 2: Scripted wrong attempt → "What did I do wrong?"
3. Turn 3: "Give me 2 practice questions"

### Arms
- **Arm A:** AIHive Tutor real implementation
- **Arm B:** Generic assistant, same model/temperature
- **Arm C:** Scripted demo (optional, non-AI baseline)

### Blinding
- Responses exported with anonymized labels (X, Y, Z)
- Order shuffled with recorded seed
- Grader scores without knowing which system produced responses
- Arm mapping revealed only after grading complete

### Rubric
Predefined scoring rubric (version 1.0) committed before collecting results.
See: `benchmarks/tutor_ged_math/rubric/scoring_rubric.md`

---

## Results

### Status: RESULTS PENDING

**To collect results:**

1. **Run live benchmark:**
   ```bash
   export OPENAI_API_KEY="your-key"
   python3 benchmarks/tutor_ged_math/harness/runner.py \
     --confirm-live --max-budget-usd 1.0 \
     --rubric-commit $(git rev-parse HEAD)
   ```

2. **Conduct blinded grading:**
   - Use `results/grading_sheet.csv`
   - Score each lesson using rubric
   - Do not look at `arm_key.json` during grading

3. **Merge results:**
   - Join grading scores with arm mapping
   - Calculate aggregate metrics per arm
   - Statistical analysis (t-tests, effect sizes)

---

## Estimated Cost

**Model:** `gpt-4o-mini`  
**Pricing (verified 2026-11-24):**
- Input: $0.150 per 1M tokens
- Output: $0.600 per 1M tokens
- Source: https://openai.com/api/pricing/

**Estimated cost for full benchmark:**
- 20 questions × 3 turns × 2 arms = 120 API calls
- ~200 input tokens/turn, ~400 output tokens/turn
- **Total estimated: $0.10 - $0.20** for Arm A + Arm B

**Note:** This is an estimate. Run with `--max-budget-usd` to set spending limit.

---

## Expected Sections (Once Results Available)

### 1. Overall Quality Scores
- Mean quality score per arm (out of 14)
- Distribution histogram
- Statistical significance tests

### 2. Dimension Breakdown
- Mean score per dimension (1-4) for each arm
- Which arm performs better on each dimension
- Effect sizes (Cohen's d)

### 3. Question-Level Analysis
- Performance by difficulty level
- Performance by domain (quantitative vs. algebraic)
- Performance by format (MC vs. fill-in)
- Hardest questions for each arm

### 4. Misconception Diagnosis
- How often did each arm correctly identify the specific misconception?
- Examples of good vs. poor diagnosis

### 5. Explanation Quality
- Clarity score distribution
- Examples of excellent vs. poor explanations
- Reading level analysis

### 6. Practice Question Quality
- Targeting accuracy
- Difficulty appropriateness
- Answer verification results

### 7. Cost Analysis
- Total cost per arm
- Cost per quality point
- Latency per lesson
- Cost-effectiveness comparison

### 8. Representative Examples
- Best responses from each arm
- Worst responses from each arm
- Interesting cases where arms diverged

---

## Interpretation Guidelines (For Future Use)

When results are available, interpret carefully:

### What Strong Results Would Mean:
✅ Responses are pedagogically sound  
✅ Explanations are clear for GED-level readers  
✅ Error diagnosis is accurate  
✅ Practice questions are well-targeted  

### What Strong Results Would NOT Mean:
❌ Students necessarily learn better  
❌ Long-term retention is better  
❌ System works for all learning styles  
❌ System is effective across all subjects  
❌ Cost is justified by learning outcomes  

### Statistical Significance ≠ Practical Significance
- A statistically significant difference (p < 0.05) means the difference is unlikely due to chance
- It does NOT mean the difference matters in practice
- Look at effect sizes and absolute differences
- Consider cost differences

### External Validity Concerns
- These are original questions, not official GED items
- Misconceptions are predefined, not from real students
- Protocol is artificial (3 turns, scripted mistakes)
- Real tutoring involves many sessions and adaptive pacing

---

## Next Steps

To complete this evaluation:

1. ✅ Commit rubric (get commit hash)
2. ⏳ Run live benchmark with real API
3. ⏳ Blinded human grading
4. ⏳ Statistical analysis
5. ⏳ Peer review by math educator
6. ⏳ Consider pilot with real learners (if results promising)

---

## Expert Review Status

**Mathematical Content:** ⏳ PENDING  
A qualified mathematics educator should review:
- Question accuracy and GED-appropriateness
- Misconception realism
- Solution correctness
- Difficulty calibration

**Educational Design:** ⏳ PENDING  
An instructional designer should review:
- Rubric validity
- Protocol appropriateness
- Limitation disclosures
- Interpretation guidelines

---

## Conclusion

**This report is a template.** No effectiveness results are available yet.

The benchmark harness is ready (all software tests pass), but educational effectiveness can only be assessed after:
1. Live API run
2. Blinded human grading
3. Statistical analysis

Even then, results measure **response quality**, not actual learning. Real learning effectiveness requires testing with actual learners over time.

**Do not claim learning outcomes without learner-based evaluation.**

---

## References

### GED Testing Service
- GED Mathematical Reasoning: https://ged.com/about_test/test_subjects/mathematical_reasoning/

### Educational Assessment Literature
- Bloom, B. S. (1984). The 2 sigma problem: The search for methods of group instruction as effective as one-to-one tutoring.
- VanLehn, K. (2011). The relative effectiveness of human tutoring, intelligent tutoring systems, and other tutoring systems.
- Koedinger, K. R., & Aleven, V. (2007). Exploring the assistance dilemma in experiments with cognitive tutors.

### Rubric Design
- Panadero, E., & Jonsson, A. (2013). The use of scoring rubrics for formative assessment purposes revisited: A review.
- Reddy, Y. M., & Andrade, H. (2010). A review of rubric use in higher education.

---

**Last Updated:** 2026-11-24  
**Next Review:** After live run completion
