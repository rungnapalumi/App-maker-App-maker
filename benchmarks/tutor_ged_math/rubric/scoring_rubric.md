# AIHive Tutor GED Math Benchmark - Scoring Rubric

**Version:** 1.0  
**Created:** 2026-11-24  
**Status:** FROZEN - This rubric is committed before collecting any results

## Purpose

This rubric evaluates the learning quality of AI tutoring systems on 20 original GED-style math questions. It measures **single-session response quality** as a proxy for educational effectiveness, not actual learning gains.

## Scoring Dimensions

Each lesson (one question, 3 turns) is scored on five dimensions:

---

### 1. Mathematical Correctness (AUTO + MANUAL)

**0 points:** Final answer is incorrect

**1 point:** Final answer is correct (within tolerance), but solution contains one or more mathematical errors in steps or reasoning

**2 points:** Final answer is correct AND all steps are mathematically valid (no errors in reasoning or computation)

**Auto-scoring:**
- Final answer parsed with tolerance (e.g., "7.5", "7.50", "15/2" all match)
- For multiple-choice: exact match required
- Grader manually checks steps for errors

---

### 2. Explanation Clarity (0-4)

Appropriate for GED-level adult learners (ages 16+, typical reading level 6th-8th grade).

**0 - Incomprehensible**
- Uses technical jargon without definition
- No structure or logical flow
- Missing critical steps
- Example: "Apply the binomial theorem to factor..."

**1 - Poor**
- Overly technical OR overly simplistic for GED level
- Confusing organization
- Missing analogies or examples where needed
- Example: Explains quadratic factoring with no concrete example

**2 - Fair**
- Mostly clear but has some issues:
  - Occasional jargon without explanation
  - Could use better examples
  - Some logical gaps
- Example: Explains slope correctly but doesn't relate it to real-world "steepness"

**3 - Good**
- Clear and organized
- Appropriate vocabulary for GED level
- Uses helpful examples or analogies
- Minor improvements possible
- Example: Explains percentage with shopping discount analogy, clear steps

**4 - Excellent**
- Exceptionally clear and engaging
- Perfect level for GED learners
- Strong analogies or examples
- Builds understanding step-by-step
- Example: Explains Pythagorean theorem with real triangle, visual language, relates to real construction/distance

---

### 3. Misconception Diagnosis (0-4)

Given the scripted student wrong attempt in Turn 2, does the tutor identify the **specific predefined misconception**?

**0 - No diagnosis**
- Generic encouragement only ("Try again!")
- No attempt to identify what went wrong
- Example: "That's not quite right. Let me re-explain everything."

**1 - Vague diagnosis**
- Acknowledges error but doesn't pinpoint it
- Example: "You made a mistake in your calculation."

**2 - Partial diagnosis**
- Identifies the general area of error
- Example: "You mixed up the order of operations."
- Doesn't name the SPECIFIC misconception from our list

**3 - Accurate diagnosis**
- Correctly identifies the specific misconception
- Example: "You added the numerators and denominators separately, but that's not how fraction addition works."
- Matches the predefined misconception description

**4 - Accurate + educational**
- Correctly identifies specific misconception AND explains WHY it's wrong and how to avoid it
- Example: "You added 2+1=3 and 3+4=7 to get 3/7. That doesn't work because fractions represent division, not separate numbers. We need a common denominator first..."

**Note:** Only score 3 or 4 if the diagnosis matches the specific predefined misconception for that student attempt.

---

### 4. Adaptive Practice Questions (0-4)

Evaluates the TWO practice questions generated in Turn 3 that target the diagnosed misconception.

**0 - Inappropriate or missing**
- No practice questions provided
- Questions are off-topic or unrelated
- Questions are impossible to answer

**1 - Generic**
- Generic practice that doesn't target the misconception
- Example after "added instead of multiplied" error: "Solve 5 + 8"
- May be too easy or too hard

**2 - Somewhat targeted**
- Questions involve the same skill but don't directly address the misconception
- Example after "forgot common denominator" error: "What is 1/2 + 1/4?" (too similar to original)
- Difficulty not quite appropriate

**3 - Well-targeted**
- Questions directly practice the diagnosed misconception
- Appropriate difficulty (similar or slightly easier than original)
- Example after "divided instead of multiplied for unit conversion": Two questions converting between units requiring multiplication
- Answers are verifiable and correct

**4 - Excellent practice set**
- Questions perfectly target the misconception
- Progressive difficulty (one easier, one similar level)
- Include verification of reasoning, not just answer
- Example after "percent of new vs. original" error: (1) Clear percent increase with explicit "of original", (2) Real-world percent change scenario
- All answers are verifiable and correct

**Auto-verification:** Where feasible, practice question answers are automatically verified for correctness.

---

### 5. Cost per Lesson (DESCRIPTIVE)

This dimension is **descriptive**, not scored on a 0-4 scale.

**Recorded metrics:**
- Input tokens (Turn 1 + Turn 2 + Turn 3)
- Output tokens (Turn 1 + Turn 2 + Turn 3)
- Model identifier (e.g., "gpt-4o-mini", "gpt-4o")
- API latency (total wall-clock time for 3 API calls)
- Cost calculation: (input_tokens × input_price_per_1M + output_tokens × output_price_per_1M) / 1,000,000
- **Price source:** Must reference official provider pricing page and date checked
- **Placeholder:** If price cannot be verified, record as "UNVERIFIED"

**Comparison context:**
- Average cost per lesson for each arm
- Median latency per lesson
- Cost per point earned on other dimensions

---

## Aggregate Scoring

For each lesson (one question):
- **Quality score:** Sum of dimensions 1-4 (max 14 points)
- **Cost:** Recorded separately

For each arm (20 lessons):
- **Mean quality score:** Average across 20 lessons
- **Quality score distribution:** Counts at each score level
- **Total cost:** Sum of all 20 lesson costs
- **Average cost per lesson**
- **Dimension breakdown:** Mean score for each of dimensions 1-4

---

## Blinding Procedure

1. Harness exports responses to grading CSV with:
   - `lesson_id`: GED-Q001 through GED-Q020
   - `turn_1_response`, `turn_2_response`, `turn_3_response`
   - `arm_label`: Randomly assigned letters (A, B, C, etc.)
   - Order shuffled

2. Arm mapping kept in separate `arm_key.json` (not visible during grading)

3. Grader scores using this rubric without knowing which system produced each response

4. After scoring complete, merge arm mapping with scores for analysis

---

## Important Limitations

**This benchmark measures single-session response quality, NOT actual learning effectiveness.**

Actual learning requires:
- Multiple sessions over time
- Pre-test and post-test with learners
- Transfer to novel problems
- Retention measurement
- Learner satisfaction and engagement surveys

This benchmark is a **necessary but not sufficient** proxy for educational quality.

---

## Human Expert Review Required

These 20 questions are original items written for this benchmark. A qualified mathematics educator should review:
- Mathematical accuracy of items and solutions
- Appropriateness for GED level
- Realism of misconceptions
- Fairness across difficulty levels

Before publishing results externally, obtain expert review.

---

## Rubric Commit Hash

This rubric was committed before any harness run. The commit hash is recorded in all result files to ensure scores were not adjusted after seeing results.

**Commit hash:** (to be recorded after commit)
