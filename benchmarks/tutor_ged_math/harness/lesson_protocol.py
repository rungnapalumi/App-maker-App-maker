"""
Lesson Protocol - Identical for all arms
One lesson = one question = 3 turns
"""

from typing import List, Tuple, Protocol
from pydantic import BaseModel
import time

from benchmarks.tutor_ged_math.items.questions import MathQuestion, Misconception


class Turn(BaseModel):
    """A single turn in the lesson"""
    turn_number: int
    user_input: str
    assistant_response: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


class LessonExecution(BaseModel):
    """Complete execution of one lesson"""
    question_id: str
    misconception_id: str  # Which misconception was tested
    turns: List[Turn]
    total_input_tokens: int
    total_output_tokens: int
    total_latency_ms: int
    
    @property
    def turn_1(self) -> Turn:
        return self.turns[0]
    
    @property
    def turn_2(self) -> Turn:
        return self.turns[1]
    
    @property
    def turn_3(self) -> Turn:
        return self.turns[2]


class TutorInterface(Protocol):
    """Interface that all tutor arms must implement"""
    
    def start_new_lesson(self, question: MathQuestion) -> None:
        """Initialize fresh context for a new lesson"""
        ...
    
    def turn_1_solve_and_explain(self, question: MathQuestion) -> Tuple[str, int, int, int]:
        """
        Turn 1: Present the question and get explanation.
        Returns: (response, input_tokens, output_tokens, latency_ms)
        """
        ...
    
    def turn_2_diagnose_error(
        self, 
        question: MathQuestion,
        misconception: Misconception
    ) -> Tuple[str, int, int, int]:
        """
        Turn 2: Student submits wrong answer with misconception.
        Returns: (response, input_tokens, output_tokens, latency_ms)
        """
        ...
    
    def turn_3_practice_questions(self, question: MathQuestion) -> Tuple[str, int, int, int]:
        """
        Turn 3: Student asks for practice questions.
        Returns: (response, input_tokens, output_tokens, latency_ms)
        """
        ...


def execute_lesson(
    tutor: TutorInterface,
    question: MathQuestion,
    misconception: Misconception
) -> LessonExecution:
    """
    Execute the full 3-turn lesson protocol.
    
    Protocol:
    1. Turn 1: Solve and explain the question
    2. Turn 2: Student submits scripted wrong attempt, ask what went wrong
    3. Turn 3: Ask for 2 practice questions targeting the misconception
    """
    # Start fresh context
    tutor.start_new_lesson(question)
    
    turns = []
    
    # Turn 1: Solve and explain
    response_1, in_1, out_1, lat_1 = tutor.turn_1_solve_and_explain(question)
    turns.append(Turn(
        turn_number=1,
        user_input=f"Please solve this problem and explain your solution:\n\n{question.stem}",
        assistant_response=response_1,
        input_tokens=in_1,
        output_tokens=out_1,
        latency_ms=lat_1
    ))
    
    # Turn 2: Diagnose error
    student_attempt = misconception.student_attempt
    response_2, in_2, out_2, lat_2 = tutor.turn_2_diagnose_error(question, misconception)
    turns.append(Turn(
        turn_number=2,
        user_input=f"I got {misconception.wrong_answer}. {student_attempt}. What did I do wrong?",
        assistant_response=response_2,
        input_tokens=in_2,
        output_tokens=out_2,
        latency_ms=lat_2
    ))
    
    # Turn 3: Practice questions
    response_3, in_3, out_3, lat_3 = tutor.turn_3_practice_questions(question)
    turns.append(Turn(
        turn_number=3,
        user_input="Can you give me 2 practice questions to help me understand this better?",
        assistant_response=response_3,
        input_tokens=in_3,
        output_tokens=out_3,
        latency_ms=lat_3
    ))
    
    return LessonExecution(
        question_id=question.id,
        misconception_id=misconception.id,
        turns=turns,
        total_input_tokens=in_1 + in_2 + in_3,
        total_output_tokens=out_1 + out_2 + out_3,
        total_latency_ms=lat_1 + lat_2 + lat_3
    )


def format_question_for_tutor(question: MathQuestion) -> str:
    """Format question for presentation to tutor"""
    formatted = question.stem
    
    if question.format == "multiple_choice" and question.choices:
        formatted += "\n\nChoices:"
        for i, choice in enumerate(question.choices, 1):
            formatted += f"\n{chr(64+i)}. {choice}"
    
    return formatted
