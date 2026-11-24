"""
20 Original GED-Style Math Questions
AIHive Tutor Benchmark - Mathematical Reasoning

NOTE: These questions are ORIGINAL, written specifically for this benchmark.
They follow GED Mathematical Reasoning style (~45% quantitative, ~55% algebraic).
A qualified math educator should review these items before publishing results.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel


class Misconception(BaseModel):
    """A specific misconception a student might have"""
    id: str
    description: str
    wrong_answer: str  # The specific incorrect answer this misconception produces
    student_attempt: str  # A scripted student attempt showing this error


class MathQuestion(BaseModel):
    """A single math question with verified answer and misconceptions"""
    id: str
    domain: Literal["quantitative", "algebraic"]
    skill: str
    difficulty: Literal["easy", "medium", "hard"]
    stem: str  # The question text
    format: Literal["multiple_choice", "fill_in"]
    choices: Optional[List[str]] = None  # For MC only
    correct_answer: str
    worked_solution: str  # Step-by-step solution
    misconceptions: List[Misconception]


# Question 1: Basic arithmetic with decimals
Q1 = MathQuestion(
    id="GED-Q001",
    domain="quantitative",
    skill="decimal_operations",
    difficulty="easy",
    stem="Maria bought 3 notebooks for $2.45 each. How much did she spend in total?",
    format="fill_in",
    correct_answer="7.35",
    worked_solution="""
Step 1: Multiply the price by quantity
3 × $2.45 = $7.35

Answer: $7.35
""",
    misconceptions=[
        Misconception(
            id="Q001-M1",
            description="Added instead of multiplied",
            wrong_answer="4.90",
            student_attempt="I added 2.45 + 2.45 = 4.90"
        ),
        Misconception(
            id="Q001-M2",
            description="Decimal point error - multiplied as whole numbers then placed decimal incorrectly",
            wrong_answer="73.5",
            student_attempt="I got 3 times 245 is 735, so I put the decimal to get 73.5"
        )
    ]
)

# Question 2: Percentage calculation
Q2 = MathQuestion(
    id="GED-Q002",
    domain="quantitative",
    skill="percentages",
    difficulty="easy",
    stem="A shirt originally costs $40. It is on sale for 25% off. What is the sale price?",
    format="multiple_choice",
    choices=["$10", "$25", "$30", "$35"],
    correct_answer="$30",
    worked_solution="""
Step 1: Calculate discount amount
25% of $40 = 0.25 × 40 = $10

Step 2: Subtract discount from original price
$40 - $10 = $30

Answer: $30
""",
    misconceptions=[
        Misconception(
            id="Q002-M1",
            description="Calculated only the discount, not the final price",
            wrong_answer="$10",
            student_attempt="25% of 40 is 10, so the answer is $10"
        ),
        Misconception(
            id="Q002-M2",
            description="Subtracted 25 instead of 25%",
            wrong_answer="$15",
            student_attempt="I subtracted 25 from 40 and got 15"
        )
    ]
)

# Question 3: Ratio and proportion
Q3 = MathQuestion(
    id="GED-Q003",
    domain="quantitative",
    skill="ratios_proportions",
    difficulty="medium",
    stem="A recipe for 4 servings requires 3 cups of flour. How many cups of flour are needed for 10 servings?",
    format="fill_in",
    correct_answer="7.5",
    worked_solution="""
Step 1: Set up proportion
4 servings / 3 cups = 10 servings / x cups

Step 2: Cross multiply
4x = 30

Step 3: Solve for x
x = 30/4 = 7.5

Answer: 7.5 cups
""",
    misconceptions=[
        Misconception(
            id="Q003-M1",
            description="Added the difference instead of using proportion",
            wrong_answer="9",
            student_attempt="10 is 6 more than 4, so I added 6 to 3 to get 9 cups"
        ),
        Misconception(
            id="Q003-M2",
            description="Set up proportion incorrectly (inverted ratio)",
            wrong_answer="13.33",
            student_attempt="I did 3/4 = 10/x, so x = 40/3 = 13.33"
        )
    ]
)

# Question 4: Linear equation solving
Q4 = MathQuestion(
    id="GED-Q004",
    domain="algebraic",
    skill="linear_equations",
    difficulty="easy",
    stem="Solve for x: 2x + 5 = 17",
    format="fill_in",
    correct_answer="6",
    worked_solution="""
Step 1: Subtract 5 from both sides
2x + 5 - 5 = 17 - 5
2x = 12

Step 2: Divide both sides by 2
x = 12/2 = 6

Answer: x = 6
""",
    misconceptions=[
        Misconception(
            id="Q004-M1",
            description="Subtracted 5 from one side only",
            wrong_answer="12",
            student_attempt="2x = 17 - 5 = 12, so x = 12"
        ),
        Misconception(
            id="Q004-M2",
            description="Divided before subtracting",
            wrong_answer="3.5",
            student_attempt="I divided 17 by 2 to get 8.5, then x = 8.5 - 5 = 3.5"
        )
    ]
)

# Question 5: Area calculation
Q5 = MathQuestion(
    id="GED-Q005",
    domain="quantitative",
    skill="area_perimeter",
    difficulty="easy",
    stem="A rectangular garden is 8 feet long and 5 feet wide. What is the area of the garden in square feet?",
    format="multiple_choice",
    choices=["13", "26", "40", "80"],
    correct_answer="40",
    worked_solution="""
Step 1: Use area formula for rectangle
Area = length × width

Step 2: Calculate
Area = 8 × 5 = 40 square feet

Answer: 40 square feet
""",
    misconceptions=[
        Misconception(
            id="Q005-M1",
            description="Calculated perimeter instead of area",
            wrong_answer="26",
            student_attempt="I added 8 + 5 + 8 + 5 = 26"
        ),
        Misconception(
            id="Q005-M2",
            description="Only added length and width",
            wrong_answer="13",
            student_attempt="8 plus 5 equals 13"
        )
    ]
)

# Question 6: Fraction operations
Q6 = MathQuestion(
    id="GED-Q006",
    domain="quantitative",
    skill="fraction_operations",
    difficulty="medium",
    stem="What is 2/3 + 1/4?",
    format="multiple_choice",
    choices=["3/7", "3/12", "11/12", "8/12"],
    correct_answer="11/12",
    worked_solution="""
Step 1: Find common denominator (LCD of 3 and 4 is 12)
2/3 = 8/12
1/4 = 3/12

Step 2: Add numerators
8/12 + 3/12 = 11/12

Answer: 11/12
""",
    misconceptions=[
        Misconception(
            id="Q006-M1",
            description="Added numerators and denominators separately",
            wrong_answer="3/7",
            student_attempt="2 + 1 = 3 and 3 + 4 = 7, so 3/7"
        ),
        Misconception(
            id="Q006-M2",
            description="Found common denominator but forgot to adjust numerators",
            wrong_answer="3/12",
            student_attempt="I made them both over 12, so 2/12 + 1/12 = 3/12"
        )
    ]
)

# Question 7: Word problem with algebraic expression
Q7 = MathQuestion(
    id="GED-Q007",
    domain="algebraic",
    skill="algebraic_expressions",
    difficulty="medium",
    stem="John has 5 more books than Mary. If Mary has x books, which expression represents the total number of books they have together?",
    format="multiple_choice",
    choices=["x + 5", "2x + 5", "5x", "x - 5"],
    correct_answer="2x + 5",
    worked_solution="""
Step 1: Express John's books in terms of x
Mary has x books
John has x + 5 books

Step 2: Add them together
Total = x + (x + 5) = 2x + 5

Answer: 2x + 5
""",
    misconceptions=[
        Misconception(
            id="Q007-M1",
            description="Only counted John's books, not the total",
            wrong_answer="x + 5",
            student_attempt="John has 5 more than Mary, so x + 5"
        ),
        Misconception(
            id="Q007-M2",
            description="Multiplied instead of adding",
            wrong_answer="5x",
            student_attempt="I thought 5 more means 5 times, so 5x"
        )
    ]
)

# Question 8: Mean (average) calculation
Q8 = MathQuestion(
    id="GED-Q008",
    domain="quantitative",
    skill="statistics_mean",
    difficulty="easy",
    stem="A student scored 85, 90, 78, and 95 on four tests. What is the average (mean) score?",
    format="fill_in",
    correct_answer="87",
    worked_solution="""
Step 1: Add all scores
85 + 90 + 78 + 95 = 348

Step 2: Divide by number of tests
348 ÷ 4 = 87

Answer: 87
""",
    misconceptions=[
        Misconception(
            id="Q008-M1",
            description="Added but forgot to divide",
            wrong_answer="348",
            student_attempt="I added 85 + 90 + 78 + 95 = 348"
        ),
        Misconception(
            id="Q008-M2",
            description="Divided by wrong number",
            wrong_answer="69.6",
            student_attempt="348 divided by 5 is 69.6"
        )
    ]
)

# Question 9: Simple interest
Q9 = MathQuestion(
    id="GED-Q009",
    domain="algebraic",
    skill="simple_interest",
    difficulty="medium",
    stem="Sarah deposits $500 in a savings account with 4% annual simple interest. How much interest will she earn after 3 years?",
    format="fill_in",
    correct_answer="60",
    worked_solution="""
Step 1: Use simple interest formula I = Prt
P = $500 (principal)
r = 0.04 (4% as decimal)
t = 3 years

Step 2: Calculate
I = 500 × 0.04 × 3 = 60

Answer: $60
""",
    misconceptions=[
        Misconception(
            id="Q009-M1",
            description="Used 4 instead of 0.04 for rate",
            wrong_answer="6000",
            student_attempt="500 times 4 times 3 = 6000"
        ),
        Misconception(
            id="Q009-M2",
            description="Multiplied only by rate, forgot time",
            wrong_answer="20",
            student_attempt="500 times 0.04 equals 20"
        )
    ]
)

# Question 10: Inequality solving
Q10 = MathQuestion(
    id="GED-Q010",
    domain="algebraic",
    skill="inequalities",
    difficulty="medium",
    stem="Solve for x: 3x - 7 < 8",
    format="multiple_choice",
    choices=["x < 1", "x < 5", "x > 5", "x < 15"],
    correct_answer="x < 5",
    worked_solution="""
Step 1: Add 7 to both sides
3x - 7 + 7 < 8 + 7
3x < 15

Step 2: Divide both sides by 3
x < 5

Answer: x < 5
""",
    misconceptions=[
        Misconception(
            id="Q010-M1",
            description="Made arithmetic error (8 - 7 = 1)",
            wrong_answer="x < 1",
            student_attempt="8 minus 7 is 1, so 3x < 1, which means x < 1"
        ),
        Misconception(
            id="Q010-M2",
            description="Incorrectly flipped inequality sign",
            wrong_answer="x > 5",
            student_attempt="I solved 3x < 15, then flipped the sign to get x > 5"
        )
    ]
)

# Question 11: Pythagorean theorem
Q11 = MathQuestion(
    id="GED-Q011",
    domain="quantitative",
    skill="pythagorean_theorem",
    difficulty="medium",
    stem="A right triangle has legs of length 6 cm and 8 cm. What is the length of the hypotenuse?",
    format="fill_in",
    correct_answer="10",
    worked_solution="""
Step 1: Use Pythagorean theorem a² + b² = c²
6² + 8² = c²

Step 2: Calculate
36 + 64 = c²
100 = c²

Step 3: Take square root
c = √100 = 10

Answer: 10 cm
""",
    misconceptions=[
        Misconception(
            id="Q011-M1",
            description="Added the legs instead of using Pythagorean theorem",
            wrong_answer="14",
            student_attempt="6 + 8 = 14 cm"
        ),
        Misconception(
            id="Q011-M2",
            description="Forgot to take square root",
            wrong_answer="100",
            student_attempt="6² + 8² = 36 + 64 = 100"
        )
    ]
)

# Question 12: Exponent rules
Q12 = MathQuestion(
    id="GED-Q012",
    domain="algebraic",
    skill="exponent_rules",
    difficulty="medium",
    stem="Simplify: x³ × x⁵",
    format="multiple_choice",
    choices=["x⁸", "x¹⁵", "x²", "2x⁸"],
    correct_answer="x⁸",
    worked_solution="""
Step 1: Apply exponent rule for multiplication
When multiplying same base, add exponents: x^a × x^b = x^(a+b)

Step 2: Calculate
x³ × x⁵ = x^(3+5) = x⁸

Answer: x⁸
""",
    misconceptions=[
        Misconception(
            id="Q012-M1",
            description="Multiplied exponents instead of adding",
            wrong_answer="x¹⁵",
            student_attempt="3 times 5 is 15, so x¹⁵"
        ),
        Misconception(
            id="Q012-M2",
            description="Thought coefficient should double",
            wrong_answer="2x⁸",
            student_attempt="Two x's multiply so 2x, and 3+5=8, giving 2x⁸"
        )
    ]
)

# Question 13: Unit conversion
Q13 = MathQuestion(
    id="GED-Q013",
    domain="quantitative",
    skill="unit_conversion",
    difficulty="easy",
    stem="A recipe calls for 2.5 liters of water. How many milliliters is this? (1 liter = 1000 milliliters)",
    format="fill_in",
    correct_answer="2500",
    worked_solution="""
Step 1: Multiply liters by conversion factor
2.5 liters × 1000 milliliters/liter

Step 2: Calculate
2.5 × 1000 = 2500 milliliters

Answer: 2500 milliliters
""",
    misconceptions=[
        Misconception(
            id="Q013-M1",
            description="Divided instead of multiplied",
            wrong_answer="0.0025",
            student_attempt="2.5 divided by 1000 equals 0.0025"
        ),
        Misconception(
            id="Q013-M2",
            description="Decimal point error",
            wrong_answer="250",
            student_attempt="2.5 times 100 is 250 milliliters"
        )
    ]
)

# Question 14: System of equations
Q14 = MathQuestion(
    id="GED-Q014",
    domain="algebraic",
    skill="systems_of_equations",
    difficulty="hard",
    stem="Solve the system: x + y = 10 and x - y = 4. What is the value of x?",
    format="fill_in",
    correct_answer="7",
    worked_solution="""
Step 1: Add the two equations to eliminate y
(x + y) + (x - y) = 10 + 4
2x = 14

Step 2: Solve for x
x = 7

Step 3: Verify (optional): If x=7, then from first equation y=3
Check: 7+3=10 ✓ and 7-3=4 ✓

Answer: x = 7
""",
    misconceptions=[
        Misconception(
            id="Q014-M1",
            description="Subtracted equations instead of adding",
            wrong_answer="3",
            student_attempt="(x+y) - (x-y) = 10-4, so 2y = 6, y = 3. I thought the question asked for y."
        ),
        Misconception(
            id="Q014-M2",
            description="Made sign error when adding equations",
            wrong_answer="14",
            student_attempt="x + y + x - y = 10 + 4, so x + x = 14, giving x = 14"
        )
    ]
)

# Question 15: Probability
Q15 = MathQuestion(
    id="GED-Q015",
    domain="quantitative",
    skill="probability",
    difficulty="medium",
    stem="A bag contains 5 red marbles, 3 blue marbles, and 2 green marbles. What is the probability of randomly selecting a blue marble?",
    format="multiple_choice",
    choices=["1/10", "3/10", "1/3", "3/5"],
    correct_answer="3/10",
    worked_solution="""
Step 1: Find total number of marbles
5 + 3 + 2 = 10 marbles

Step 2: Calculate probability
P(blue) = number of blue marbles / total marbles
P(blue) = 3/10

Answer: 3/10
""",
    misconceptions=[
        Misconception(
            id="Q015-M1",
            description="Used blue marbles over non-blue marbles",
            wrong_answer="3/7",
            student_attempt="There are 3 blue and 7 other marbles, so 3/7"
        ),
        Misconception(
            id="Q015-M2",
            description="Used blue over number of colors",
            wrong_answer="1/3",
            student_attempt="There are 3 colors and blue is one of them, so 1/3"
        )
    ]
)

# Question 16: Quadratic equation
Q16 = MathQuestion(
    id="GED-Q016",
    domain="algebraic",
    skill="quadratic_equations",
    difficulty="hard",
    stem="Solve for x: x² - 5x + 6 = 0",
    format="multiple_choice",
    choices=["x = 1 or x = 6", "x = 2 or x = 3", "x = -2 or x = -3", "x = 5 or x = 6"],
    correct_answer="x = 2 or x = 3",
    worked_solution="""
Step 1: Factor the quadratic
x² - 5x + 6 = (x - 2)(x - 3) = 0

Step 2: Apply zero product property
x - 2 = 0  or  x - 3 = 0

Step 3: Solve each equation
x = 2  or  x = 3

Answer: x = 2 or x = 3
""",
    misconceptions=[
        Misconception(
            id="Q016-M1",
            description="Made sign error in factoring",
            wrong_answer="x = -2 or x = -3",
            student_attempt="I factored as (x+2)(x+3) so x = -2 or -3"
        ),
        Misconception(
            id="Q016-M2",
            description="Found factors but didn't solve correctly",
            wrong_answer="x = 5 or x = 6",
            student_attempt="The factors are -5 and 6, so those are the answers"
        )
    ]
)

# Question 17: Distance-rate-time
Q17 = MathQuestion(
    id="GED-Q017",
    domain="algebraic",
    skill="distance_rate_time",
    difficulty="medium",
    stem="A car travels at 60 miles per hour. How long will it take to travel 240 miles?",
    format="fill_in",
    correct_answer="4",
    worked_solution="""
Step 1: Use formula time = distance / rate
t = d / r

Step 2: Substitute values
t = 240 miles / 60 mph

Step 3: Calculate
t = 4 hours

Answer: 4 hours
""",
    misconceptions=[
        Misconception(
            id="Q017-M1",
            description="Multiplied instead of dividing",
            wrong_answer="14400",
            student_attempt="240 times 60 equals 14400"
        ),
        Misconception(
            id="Q017-M2",
            description="Divided in wrong order",
            wrong_answer="0.25",
            student_attempt="60 divided by 240 equals 0.25"
        )
    ]
)

# Question 18: Percent increase
Q18 = MathQuestion(
    id="GED-Q018",
    domain="quantitative",
    skill="percent_change",
    difficulty="hard",
    stem="The population of a town increased from 5,000 to 6,500. What was the percent increase?",
    format="multiple_choice",
    choices=["13%", "23%", "30%", "50%"],
    correct_answer="30%",
    worked_solution="""
Step 1: Find the amount of increase
6,500 - 5,000 = 1,500

Step 2: Divide increase by original amount
1,500 / 5,000 = 0.30

Step 3: Convert to percentage
0.30 × 100% = 30%

Answer: 30%
""",
    misconceptions=[
        Misconception(
            id="Q018-M1",
            description="Divided by new value instead of original",
            wrong_answer="23%",
            student_attempt="1500 divided by 6500 is about 0.23, so 23%"
        ),
        Misconception(
            id="Q018-M2",
            description="Calculated ratio of old to new minus 1, got decimal error",
            wrong_answer="13%",
            student_attempt="6500/5000 = 1.3, subtract 1 to get 0.3, but I wrote 13% instead of 30%"
        )
    ]
)

# Question 19: Function evaluation
Q19 = MathQuestion(
    id="GED-Q019",
    domain="algebraic",
    skill="functions",
    difficulty="medium",
    stem="If f(x) = 3x - 4, what is f(5)?",
    format="fill_in",
    correct_answer="11",
    worked_solution="""
Step 1: Substitute x = 5 into the function
f(5) = 3(5) - 4

Step 2: Calculate
f(5) = 15 - 4 = 11

Answer: 11
""",
    misconceptions=[
        Misconception(
            id="Q019-M1",
            description="Added 5 to expression instead of substituting",
            wrong_answer="3x + 1",
            student_attempt="f(5) means add 5, so 3x - 4 + 5 = 3x + 1"
        ),
        Misconception(
            id="Q019-M2",
            description="Order of operations error",
            wrong_answer="5",
            student_attempt="3 times 5 minus 4 = 3 times 1 = 3... wait, I did 5-4 first to get 1, then 3×1 but that's wrong"
        )
    ]
)

# Question 20: Slope calculation
Q20 = MathQuestion(
    id="GED-Q020",
    domain="algebraic",
    skill="slope",
    difficulty="hard",
    stem="Find the slope of the line passing through points (2, 3) and (6, 11).",
    format="fill_in",
    correct_answer="2",
    worked_solution="""
Step 1: Use slope formula m = (y₂ - y₁) / (x₂ - x₁)
Point 1: (2, 3) so x₁=2, y₁=3
Point 2: (6, 11) so x₂=6, y₂=11

Step 2: Substitute
m = (11 - 3) / (6 - 2)

Step 3: Calculate
m = 8 / 4 = 2

Answer: 2
""",
    misconceptions=[
        Misconception(
            id="Q020-M1",
            description="Divided x-difference by y-difference (inverted)",
            wrong_answer="0.5",
            student_attempt="(6-2) divided by (11-3) = 4/8 = 0.5"
        ),
        Misconception(
            id="Q020-M2",
            description="Made subtraction error with signs",
            wrong_answer="-2",
            student_attempt="(3-11)/(2-6) = -8/-4... wait I got confused with negatives"
        )
    ]
)


# All questions list
ALL_QUESTIONS: List[MathQuestion] = [
    Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10,
    Q11, Q12, Q13, Q14, Q15, Q16, Q17, Q18, Q19, Q20
]


def get_question_by_id(question_id: str) -> Optional[MathQuestion]:
    """Get a question by its ID"""
    for q in ALL_QUESTIONS:
        if q.id == question_id:
            return q
    return None


def get_questions_by_domain(domain: Literal["quantitative", "algebraic"]) -> List[MathQuestion]:
    """Filter questions by domain"""
    return [q for q in ALL_QUESTIONS if q.domain == domain]


def get_questions_by_difficulty(difficulty: Literal["easy", "medium", "hard"]) -> List[MathQuestion]:
    """Filter questions by difficulty"""
    return [q for q in ALL_QUESTIONS if q.difficulty == difficulty]
