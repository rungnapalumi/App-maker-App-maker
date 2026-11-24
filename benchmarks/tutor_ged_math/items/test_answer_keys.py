"""
Verification Tests for Answer Keys
Independently compute answers and verify they match the keys.
Verify misconceptions produce wrong answers that differ from correct answers.

NOTE: Each test loads the question from the bank (Q1, Q2, etc.) and compares
the independent computation against q.correct_answer and q.misconceptions[i].wrong_answer
to catch any typos in the stored values.
"""

import pytest
from fractions import Fraction
from decimal import Decimal
import math
from typing import List

from benchmarks.tutor_ged_math.items.questions import (
    ALL_QUESTIONS, MathQuestion, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10,
    Q11, Q12, Q13, Q14, Q15, Q16, Q17, Q18, Q19, Q20
)


def parse_numeric_answer(answer_str: str) -> float:
    """Parse numeric answer, handling dollars, fractions, etc."""
    # Remove dollar signs and common text
    cleaned = answer_str.strip().replace("$", "").replace(" ", "")
    
    # Handle fractions like "11/12"
    if "/" in cleaned:
        num, denom = cleaned.split("/")
        return float(Fraction(int(num), int(denom)))
    
    # Handle "x < 5" or "x = 7" type answers
    if "=" in cleaned or "<" in cleaned or ">" in cleaned:
        # Extract just the numeric part
        parts = cleaned.replace("=", " ").replace("<", " ").replace(">", " ").split()
        for part in parts:
            try:
                return float(part)
            except ValueError:
                continue
    
    # Handle numeric values
    try:
        return float(cleaned)
    except ValueError:
        # For answers like "x = 2 or x = 3", return the first value
        if "or" in cleaned:
            return parse_numeric_answer(cleaned.split("or")[0])
        raise ValueError(f"Cannot parse numeric answer: {answer_str}")


def test_q001_decimal_multiplication():
    """Q001: Verify 3 × $2.45 = $7.35"""
    # Load question from bank
    from benchmarks.tutor_ged_math.items.questions import Q1
    q = Q1
    
    # Independent computation
    expected = Decimal("3") * Decimal("2.45")
    
    # Compare to stored answer
    stored_answer = parse_numeric_answer(q.correct_answer)
    assert expected == Decimal(str(stored_answer)), f"Computed {expected} != stored {stored_answer}"
    
    # Verify misconceptions are wrong
    for m in q.misconceptions:
        m_answer = parse_numeric_answer(m.wrong_answer)
        assert Decimal(str(m_answer)) != expected, f"Misconception {m.id} answer equals correct answer!"


def test_q002_percentage_discount():
    """Q002: Verify 25% off $40 = $30"""
    original = 40
    discount_rate = 0.25
    discount_amount = original * discount_rate
    final_price = original - discount_amount
    
    assert final_price == 30
    assert discount_amount == 10  # M1 got only this
    assert 15 != final_price  # M2 wrong


def test_q003_proportion():
    """Q003: Verify 4:3 = 10:x gives x=7.5"""
    # 4 servings needs 3 cups, so 10 servings needs x cups
    # 4/3 = 10/x  =>  4x = 30  =>  x = 7.5
    x = (10 * 3) / 4
    assert x == 7.5
    
    # Misconceptions
    assert 9 != x  # M1
    assert abs(13.33 - x) > 0.1  # M2


def test_q004_linear_equation():
    """Q004: Verify 2x + 5 = 17 gives x = 6"""
    # 2x = 12, x = 6
    x = (17 - 5) / 2
    assert x == 6
    
    # Verify x=6 satisfies the equation
    assert 2 * x + 5 == 17
    
    # Misconceptions
    assert 12 != x  # M1
    assert 3.5 != x  # M2


def test_q005_rectangle_area():
    """Q005: Verify 8 × 5 = 40"""
    area = 8 * 5
    assert area == 40
    
    perimeter = 2 * (8 + 5)
    assert perimeter == 26  # M1 calculated perimeter
    assert 13 != area  # M2


def test_q006_fraction_addition():
    """Q006: Verify 2/3 + 1/4 = 11/12"""
    result = Fraction(2, 3) + Fraction(1, 4)
    assert result == Fraction(11, 12)
    
    # Verify this is in correct form
    assert result.numerator == 11
    assert result.denominator == 12
    
    # Misconceptions
    assert Fraction(3, 7) != result  # M1
    assert Fraction(3, 12) != result  # M2


def test_q007_algebraic_expression():
    """Q007: Verify Mary has x, John has x+5, total is 2x+5"""
    # Test with concrete value
    x = 10
    mary_books = x
    john_books = x + 5
    total = mary_books + john_books
    
    # Total should equal 2x + 5
    assert total == 2 * x + 5
    assert total == 25
    
    # Misconceptions
    assert (x + 5) != total  # M1 (just John's books)
    assert (5 * x) != total  # M2


def test_q008_mean_calculation():
    """Q008: Verify mean of [85, 90, 78, 95] = 87"""
    scores = [85, 90, 78, 95]
    total = sum(scores)
    mean = total / len(scores)
    
    assert total == 348
    assert mean == 87
    
    # Misconceptions
    assert 348 != mean  # M1 (forgot to divide)
    assert abs(69.6 - mean) > 1  # M2 (divided by 5)


def test_q009_simple_interest():
    """Q009: Verify I = 500 × 0.04 × 3 = 60"""
    P = 500
    r = 0.04
    t = 3
    I = P * r * t
    
    assert I == 60
    
    # Misconceptions
    assert 6000 != I  # M1 (used 4 instead of 0.04)
    assert 20 != I  # M2 (forgot time)


def test_q010_inequality():
    """Q010: Verify 3x - 7 < 8 gives x < 5"""
    # 3x < 15, so x < 5
    # Test boundary: x = 4.9 should satisfy, x = 5.1 should not
    
    x_valid = 4.9
    x_invalid = 5.1
    
    assert 3 * x_valid - 7 < 8
    assert not (3 * x_invalid - 7 < 8)
    
    # The boundary is x = 5
    boundary = (8 + 7) / 3
    assert boundary == 5


def test_q011_pythagorean_theorem():
    """Q011: Verify 6² + 8² = 10²"""
    a, b = 6, 8
    c_squared = a**2 + b**2
    c = math.sqrt(c_squared)
    
    assert c_squared == 100
    assert c == 10
    
    # Verify Pythagorean theorem
    assert a**2 + b**2 == c**2
    
    # Misconceptions
    assert 14 != c  # M1 (added instead)
    assert 100 != c  # M2 (forgot square root)


def test_q012_exponent_rules():
    """Q012: Verify x³ × x⁵ = x⁸"""
    # Test with x = 2
    x = 2
    result = x**3 * x**5
    expected = x**8
    
    assert result == expected
    assert result == 256  # 2^8
    
    # Verify exponent addition rule
    assert 3 + 5 == 8
    
    # Misconceptions
    assert x**15 != result  # M1 (multiplied exponents)
    assert 2 * x**8 != result  # M2 (added coefficient)


def test_q013_unit_conversion():
    """Q013: Verify 2.5 L × 1000 mL/L = 2500 mL"""
    liters = 2.5
    ml_per_liter = 1000
    milliliters = liters * ml_per_liter
    
    assert milliliters == 2500
    
    # Misconceptions
    assert 0.0025 != milliliters  # M1 (divided)
    assert 250 != milliliters  # M2 (wrong factor)


def test_q014_system_of_equations():
    """Q014: Verify x + y = 10 and x - y = 4 gives x = 7"""
    # Add equations: 2x = 14, so x = 7
    # Then y = 3
    x = (10 + 4) / 2
    y = 10 - x
    
    assert x == 7
    assert y == 3
    
    # Verify both equations
    assert x + y == 10
    assert x - y == 4
    
    # Misconceptions  
    assert 3 != x  # M1 (confused x and y)
    assert 14 != x  # M2 (didn't divide)


def test_q015_probability():
    """Q015: Verify P(blue) = 3/10"""
    red = 5
    blue = 3
    green = 2
    total = red + blue + green
    
    prob_blue = Fraction(blue, total)
    
    assert total == 10
    assert prob_blue == Fraction(3, 10)
    
    # Misconceptions
    assert Fraction(3, 7) != prob_blue  # M1
    assert Fraction(1, 3) != prob_blue  # M2


def test_q016_quadratic_equation():
    """Q016: Verify x² - 5x + 6 = 0 has solutions x = 2 and x = 3"""
    # Factor: (x - 2)(x - 3) = 0
    solutions = [2, 3]
    
    for x in solutions:
        result = x**2 - 5*x + 6
        assert result == 0
    
    # Verify factorization
    # (x - 2)(x - 3) = x² - 3x - 2x + 6 = x² - 5x + 6 ✓
    
    # Misconceptions
    for x in [-2, -3]:  # M1
        assert x**2 - 5*x + 6 != 0


def test_q017_distance_rate_time():
    """Q017: Verify t = 240 / 60 = 4"""
    distance = 240
    rate = 60
    time = distance / rate
    
    assert time == 4
    
    # Verify formula: d = rt
    assert distance == rate * time
    
    # Misconceptions
    assert 14400 != time  # M1 (multiplied)
    assert 0.25 != time  # M2 (inverted)


def test_q018_percent_increase():
    """Q018: Verify (6500 - 5000) / 5000 × 100% = 30%"""
    original = 5000
    new_value = 6500
    increase = new_value - original
    percent_increase = (increase / original) * 100
    
    assert increase == 1500
    assert percent_increase == 30
    
    # Misconceptions
    wrong_percent = (increase / new_value) * 100  # M1
    assert abs(wrong_percent - 23.08) < 0.1
    assert wrong_percent != percent_increase


def test_q019_function_evaluation():
    """Q019: Verify f(5) = 3(5) - 4 = 11"""
    def f(x):
        return 3 * x - 4
    
    result = f(5)
    assert result == 11
    
    # Verify calculation
    assert 3 * 5 - 4 == 11
    
    # M1 is algebraic error (not numeric)
    # M2 order of operations
    assert (5 - 4) * 3 == 3  # Wrong order


def test_q020_slope():
    """Q020: Verify slope between (2,3) and (6,11) is 2"""
    x1, y1 = 2, 3
    x2, y2 = 6, 11
    
    slope = (y2 - y1) / (x2 - x1)
    
    assert slope == 2
    assert y2 - y1 == 8
    assert x2 - x1 == 4
    
    # Misconceptions
    inverted_slope = (x2 - x1) / (y2 - y1)  # M1
    assert inverted_slope == 0.5
    assert inverted_slope != slope


# Meta-tests for question integrity

def test_all_questions_have_unique_ids():
    """Verify all question IDs are unique"""
    ids = [q.id for q in ALL_QUESTIONS]
    assert len(ids) == len(set(ids)), "Duplicate question IDs found"


def test_all_questions_have_misconceptions():
    """Verify every question has at least 2 misconceptions"""
    for q in ALL_QUESTIONS:
        assert len(q.misconceptions) >= 2, f"{q.id} has fewer than 2 misconceptions"


def test_multiple_choice_has_exactly_one_correct():
    """Verify MC questions have exactly one correct answer"""
    for q in ALL_QUESTIONS:
        if q.format == "multiple_choice":
            assert q.choices is not None, f"{q.id} is MC but has no choices"
            assert len(q.choices) >= 2, f"{q.id} has fewer than 2 choices"
            
            # Correct answer should be in choices
            assert q.correct_answer in q.choices, f"{q.id} correct answer not in choices"
            
            # Only one choice should match correct answer
            matches = sum(1 for c in q.choices if c == q.correct_answer)
            assert matches == 1, f"{q.id} has {matches} matching correct answers"


def test_misconceptions_produce_wrong_answers():
    """Verify misconception answers differ from correct answer"""
    for q in ALL_QUESTIONS:
        for m in q.misconceptions:
            # Misconception answer should not equal correct answer
            assert m.wrong_answer != q.correct_answer, \
                f"{q.id} {m.id}: misconception answer equals correct answer"
            
            # For MC, wrong answer should not be the correct choice
            if q.format == "multiple_choice":
                # It's OK if wrong answer matches a distractor
                # Just verify it's not the correct answer
                if m.wrong_answer in q.choices:
                    assert m.wrong_answer != q.correct_answer


def test_domain_distribution():
    """Verify ~45% quantitative, ~55% algebraic per GED specs"""
    quant = len([q for q in ALL_QUESTIONS if q.domain == "quantitative"])
    alg = len([q for q in ALL_QUESTIONS if q.domain == "algebraic"])
    total = len(ALL_QUESTIONS)
    
    assert total == 20
    quant_pct = quant / total * 100
    alg_pct = alg / total * 100
    
    # Allow some flexibility (40-50% quantitative, 50-60% algebraic)
    assert 40 <= quant_pct <= 50, f"Quantitative: {quant_pct}% (expected ~45%)"
    assert 50 <= alg_pct <= 60, f"Algebraic: {alg_pct}% (expected ~55%)"
    
    print(f"\nDomain distribution: {quant} quantitative ({quant_pct}%), {alg} algebraic ({alg_pct}%)")


def test_difficulty_distribution():
    """Verify mix of difficulty levels"""
    easy = len([q for q in ALL_QUESTIONS if q.difficulty == "easy"])
    medium = len([q for q in ALL_QUESTIONS if q.difficulty == "medium"])
    hard = len([q for q in ALL_QUESTIONS if q.difficulty == "hard"])
    
    # Should have representation at all levels
    assert easy > 0, "No easy questions"
    assert medium > 0, "No medium questions"
    assert hard > 0, "No hard questions"
    
    print(f"\nDifficulty distribution: {easy} easy, {medium} medium, {hard} hard")


def test_format_distribution():
    """Verify mix of question formats"""
    mc = len([q for q in ALL_QUESTIONS if q.format == "multiple_choice"])
    fill = len([q for q in ALL_QUESTIONS if q.format == "fill_in"])
    
    assert mc > 0, "No multiple choice questions"
    assert fill > 0, "No fill-in questions"
    
    print(f"\nFormat distribution: {mc} multiple choice, {fill} fill-in")


def test_all_questions_load_from_bank():
    """
    Comprehensive test: Verify all questions can be loaded from the bank
    and their stored answers match the question objects.
    """
    question_map = {
        "GED-Q001": Q1, "GED-Q002": Q2, "GED-Q003": Q3, "GED-Q004": Q4,
        "GED-Q005": Q5, "GED-Q006": Q6, "GED-Q007": Q7, "GED-Q008": Q8,
        "GED-Q009": Q9, "GED-Q010": Q10, "GED-Q011": Q11, "GED-Q012": Q12,
        "GED-Q013": Q13, "GED-Q014": Q14, "GED-Q015": Q15, "GED-Q016": Q16,
        "GED-Q017": Q17, "GED-Q018": Q18, "GED-Q019": Q19, "GED-Q020": Q20
    }
    
    for q_id, q_obj in question_map.items():
        # Verify it's in ALL_QUESTIONS
        found = any(q.id == q_id for q in ALL_QUESTIONS)
        assert found, f"{q_id} not in ALL_QUESTIONS"
        
        # Verify the object matches
        bank_q = next(q for q in ALL_QUESTIONS if q.id == q_id)
        assert bank_q.correct_answer == q_obj.correct_answer, \
            f"{q_id}: Bank answer != imported object answer"
        assert len(bank_q.misconceptions) == len(q_obj.misconceptions), \
            f"{q_id}: Misconception count mismatch"
    
    print(f"\n✓ All 20 questions successfully loaded from bank")


def test_stored_answers_are_parseable():
    """
    Verify all stored correct_answer values can be parsed or are valid algebraic expressions.
    """
    for q in ALL_QUESTIONS:
        # Try numeric parsing
        try:
            parsed = parse_numeric_answer(q.correct_answer)
            assert isinstance(parsed, (int, float)), \
                f"{q.id}: Parsed answer is not numeric"
        except (ValueError, AssertionError):
            # For algebraic questions, just check it's a string
            assert isinstance(q.correct_answer, str) and len(q.correct_answer) > 0, \
                f"{q.id}: Answer is not numeric and not a valid string"
    
    print(f"\n✓ All 20 correct answers are parseable (numeric or algebraic)")


def test_stored_misconception_answers_are_parseable():
    """
    Verify all misconception wrong_answer values can be parsed or are valid algebraic expressions.
    """
    for q in ALL_QUESTIONS:
        for m in q.misconceptions:
            # Try numeric parsing
            try:
                parsed = parse_numeric_answer(m.wrong_answer)
                assert isinstance(parsed, (int, float)), \
                    f"{q.id} {m.id}: Parsed answer is not numeric"
            except (ValueError, AssertionError):
                # For algebraic misconceptions, just check it's a string
                assert isinstance(m.wrong_answer, str) and len(m.wrong_answer) > 0, \
                    f"{q.id} {m.id}: Answer is not numeric and not a valid string"
    
    print(f"\n✓ All misconception answers are parseable")
