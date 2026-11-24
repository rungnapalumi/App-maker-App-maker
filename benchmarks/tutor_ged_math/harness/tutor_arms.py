"""
Tutor Arms for Benchmark
Arm A: AIHive Tutor (real implementation)
Arm B: Generic Assistant (same model, generic prompt)
Arm C: Scripted Demo (baseline, not AI)

IMPORTANT LIMITATION:
Arm A follows the real app flow where the Tutor never sees the specific
question text. The app only sends topic + level to the AI. This is a
fundamental design limitation documented in the reports.
"""

import sys
import os
import time
from typing import Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from tutor.tutor_ai import TutorAI
from tutor.models import StudySession
from aihive_coach.config import config

from benchmarks.tutor_ged_math.items.questions import MathQuestion, Misconception
from benchmarks.tutor_ged_math.harness.lesson_protocol import format_question_for_tutor
from benchmarks.tutor_ged_math.harness.token_counter import TokenCountingWrapper


class MockOpenAIClient:
    """
    Mock OpenAI client for testing.
    Returns scripted responses without making API calls.
    Mimics real API response structure including usage tracking.
    """
    
    def __init__(self):
        self.call_count = 0
        
    class ChatCompletions:
        def __init__(self, parent):
            self.parent = parent
            
        def create(self, model, messages, temperature, max_tokens, **kwargs):
            """Mock completion with realistic token counts"""
            self.parent.call_count += 1
            
            # Calculate realistic token estimates
            # Count tokens in messages (rough: 1 token ≈ 0.75 words)
            total_text = ""
            for msg in messages:
                total_text += msg.get("content", "")
            
            word_count = len(total_text.split())
            prompt_tokens = int(word_count * 1.3)  # ~1.3 tokens per word
            
            # Mock response length based on max_tokens
            completion_tokens = min(50 + self.parent.call_count * 10, max_tokens or 500)
            
            # Mock response
            mock_response = f"[MOCK RESPONSE {self.parent.call_count}] This is a scripted response for testing. Temperature={temperature}, max_tokens={max_tokens}."
            
            class MockUsage:
                def __init__(self, prompt_tokens, completion_tokens):
                    self.prompt_tokens = prompt_tokens
                    self.completion_tokens = completion_tokens
            
            class MockMessage:
                def __init__(self, content):
                    self.content = content
            
            class MockChoice:
                def __init__(self, content):
                    self.message = MockMessage(content)
            
            class MockResponse:
                def __init__(self, content, usage):
                    self.choices = [MockChoice(content)]
                    self.usage = usage
            
            return MockResponse(mock_response, MockUsage(prompt_tokens, completion_tokens))
    
    @property
    def chat(self):
        class Chat:
            completions = MockOpenAIClient.ChatCompletions(self)
        return Chat()


class ArmA_AIHiveTutor:
    """
    Arm A: AIHive Tutor using real implementation.
    Follows the actual app flow from tutor/lesson_page.py.
    
    Two variants:
    - A1 (topic_mode="skill"): topic = "GED Math - {skill}" (generic topic)
    - A2 (topic_mode="full"): topic = full question text (specific problem)
    
    Both are real app paths - the topic field accepts any text.
    """
    
    def __init__(self, openai_client=None, force_ai_mode=False, topic_mode="skill"):
        """
        Args:
            openai_client: Optional OpenAI client. If None and OPENAI_API_KEY set,
                          creates real client. If mock client provided, uses it.
            force_ai_mode: If True, force tutor into AI mode even without real key
            topic_mode: "skill" for A1 (generic) or "full" for A2 (full question)
        """
        self.session: Optional[StudySession] = None
        self.topic_mode = topic_mode
        
        # Create tutor
        self.tutor = TutorAI()
        
        # Set up client
        if openai_client:
            # Wrap client for token tracking
            self.wrapped_client = TokenCountingWrapper(openai_client)
            self.tutor.client = self.wrapped_client
            
            # Force AI mode if requested (for testing)
            if force_ai_mode:
                self.tutor.is_demo = False
        else:
            self.wrapped_client = None
    
    def start_new_lesson(self, question: MathQuestion) -> None:
        """
        Start fresh session for question.
        
        A1 (topic_mode="skill"): topic = "GED Math - {skill}"
        A2 (topic_mode="full"): topic = full question text
        """
        if self.topic_mode == "full":
            # A2: Full question text in topic (real app path: learner types problem)
            topic = format_question_for_tutor(question)
        else:
            # A1: Generic skill topic (real app path: learner types subject)
            topic = f"GED Math - {question.skill}"
        
        self.session = StudySession(
            user_id="benchmark_user",
            topic=topic,
            level="Intermediate"  # GED is roughly intermediate
        )
        
        # Reset token tracking for this lesson
        if self.wrapped_client:
            self.wrapped_client.reset_tracking()
    
    def turn_1_solve_and_explain(self, question: MathQuestion) -> Tuple[str, int, int, int]:
        """
        Turn 1: Get tutor explanation.
        
        LIMITATION: Following real app (lesson_page.py:80-85), we just call
        start_lesson(session). The question text is NOT sent to the AI.
        The Tutor only sees topic="GED Math - {skill}" and level="Intermediate".
        """
        # Follow real app flow: lesson_page.py line 83
        response = self.tutor.start_lesson(self.session)
        
        # Record response in session (line 84)
        self.session.add_message("assistant", response, is_question=True)
        
        # Get token counts from wrapper
        if self.wrapped_client:
            input_tokens = self.wrapped_client.total_input_tokens
            output_tokens = self.wrapped_client.total_output_tokens
            latency_ms = self.wrapped_client.total_latency_ms
        else:
            # Demo mode - estimate
            input_tokens = len(f"GED Math - {question.skill}".split()) * 1.3
            output_tokens = len(response.split()) * 1.3
            latency_ms = 0
        
        return response, int(input_tokens), int(output_tokens), latency_ms
    
    def turn_2_diagnose_error(
        self,
        question: MathQuestion,
        misconception: Misconception
    ) -> Tuple[str, int, int, int]:
        """
        Turn 2: Diagnose student error.
        
        Following real app flow (lesson_page.py:160-170):
        1. Call process_answer (line 160)
        2. Call continue_lesson based on correctness (line 170)
        """
        # Reset token tracking for this turn
        if self.wrapped_client:
            self.wrapped_client.reset_tracking()
        
        # Student submits wrong answer
        student_msg = f"I got {misconception.wrong_answer}. {misconception.student_attempt}. What did I do wrong?"
        self.session.add_message("user", student_msg, is_question=True)
        
        # Follow real app: lesson_page.py line 160
        is_correct, feedback = self.tutor.process_answer(self.session, student_msg)
        
        # Record correctness (lines 162-165)
        if is_correct:
            self.session.questions_answered_correctly += 1
        else:
            self.session.questions_answered_incorrectly += 1
        
        # Give feedback (line 168)
        self.session.add_message("assistant", feedback)
        
        # Continue lesson (line 170)
        next_chunk = self.tutor.continue_lesson(self.session, is_correct)
        self.session.add_message("assistant", next_chunk, is_question=True)
        self.session.chunks_completed += 1
        
        # Combine feedback + next_chunk as the full Turn 2 response
        full_response = feedback + "\n\n" + next_chunk
        
        # Get token counts (includes BOTH API calls: process_answer + continue_lesson)
        if self.wrapped_client:
            input_tokens = self.wrapped_client.total_input_tokens
            output_tokens = self.wrapped_client.total_output_tokens
            latency_ms = self.wrapped_client.total_latency_ms
        else:
            input_tokens = len(student_msg.split()) * 1.3
            output_tokens = len(full_response.split()) * 1.3
            latency_ms = 0
        
        return full_response, int(input_tokens), int(output_tokens), latency_ms
    
    def turn_3_practice_questions(self, question: MathQuestion) -> Tuple[str, int, int, int]:
        """
        Turn 3: Get practice questions.
        
        Following real app flow (lesson_page.py:122-125):
        Calls generate_summary which makes TWO API calls internally.
        """
        # Reset token tracking for this turn
        if self.wrapped_client:
            self.wrapped_client.reset_tracking()
        
        # Request practice (simulating user clicking "End Lesson")
        # Mark session ready for summary
        self.session.chunks_completed = 2
        
        # Follow real app: lesson_page.py line 123
        summary, practice_questions = self.tutor.generate_summary(self.session)
        
        # Format response
        response = summary + "\n\n**Practice Questions:**\n"
        for i, q in enumerate(practice_questions[:2], 1):  # Only first 2
            response += f"\n{i}. {q}"
        
        # Get token counts (includes BOTH API calls: summary + practice questions)
        if self.wrapped_client:
            input_tokens = self.wrapped_client.total_input_tokens
            output_tokens = self.wrapped_client.total_output_tokens
            latency_ms = self.wrapped_client.total_latency_ms
        else:
            input_tokens = 50 * 1.3
            output_tokens = len(response.split()) * 1.3
            latency_ms = 0
        
        return response, int(input_tokens), int(output_tokens), latency_ms


class ArmB_GenericAssistant:
    """
    Arm B: Generic AI assistant.
    
    One call per learner turn (natural for a general-purpose assistant).
    Max tokens per turn set to Tutor's total for that turn:
    - Turn 1: 500 tokens (matches Tutor's single call)
    - Turn 2: 700 tokens (Tutor's 200+500 combined)
    - Turn 3: 600 tokens (Tutor's 300+300 combined)
    
    Temperature averaged: 0.7 for all turns (Tutor averages 0.65 per turn)
    
    PROMPT: Same learner-facing text for both arms.
    """
    
    def __init__(self, openai_client=None, model: str = "gpt-4o-mini"):
        """
        Args:
            openai_client: OpenAI client (required, can be mock)
            model: Model to use (default matches Tutor)
        """
        if openai_client is None:
            raise ValueError("OpenAI client required for GenericAssistant")
        
        # Wrap client for token tracking
        self.wrapped_client = TokenCountingWrapper(openai_client)
        self.model = model
        self.messages = []
    
    def start_new_lesson(self, question: MathQuestion) -> None:
        """Start fresh conversation"""
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        self.wrapped_client.reset_tracking()
    
    def turn_1_solve_and_explain(self, question: MathQuestion) -> Tuple[str, int, int, int]:
        """
        Turn 1: Solve and explain.
        One call: temp=0.7, max_tokens=500
        """
        formatted_q = format_question_for_tutor(question)
        # Same learner-facing text
        prompt = f"I want to learn about this problem:\n\n{formatted_q}\n\nCan you help me understand how to solve it?"
        
        self.messages.append({"role": "user", "content": prompt})
        
        response = self.wrapped_client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0.7,
            max_tokens=500
        )
        
        assistant_message = response.choices[0].message.content.strip()
        self.messages.append({"role": "assistant", "content": assistant_message})
        
        return (
            assistant_message,
            self.wrapped_client.total_input_tokens,
            self.wrapped_client.total_output_tokens,
            self.wrapped_client.total_latency_ms
        )
    
    def turn_2_diagnose_error(
        self,
        question: MathQuestion,
        misconception: Misconception
    ) -> Tuple[str, int, int, int]:
        """
        Turn 2: Diagnose error.
        One call: temp=0.7, max_tokens=700 (sum of Tutor's 200+500)
        """
        self.wrapped_client.reset_tracking()
        
        # Same learner-facing text
        prompt = f"I got {misconception.wrong_answer}. {misconception.student_attempt}. What did I do wrong?"
        self.messages.append({"role": "user", "content": prompt})
        
        response = self.wrapped_client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0.7,
            max_tokens=700
        )
        
        assistant_message = response.choices[0].message.content.strip()
        self.messages.append({"role": "assistant", "content": assistant_message})
        
        return (
            assistant_message,
            self.wrapped_client.total_input_tokens,
            self.wrapped_client.total_output_tokens,
            self.wrapped_client.total_latency_ms
        )
    
    def turn_3_practice_questions(self, question: MathQuestion) -> Tuple[str, int, int, int]:
        """
        Turn 3: Summary and practice questions.
        One call: temp=0.7, max_tokens=600 (sum of Tutor's 300+300)
        """
        self.wrapped_client.reset_tracking()
        
        prompt = "Can you summarize what we covered and give me 2 practice questions?"
        self.messages.append({"role": "user", "content": prompt})
        
        response = self.wrapped_client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0.7,
            max_tokens=600
        )
        
        assistant_message = response.choices[0].message.content.strip()
        self.messages.append({"role": "assistant", "content": assistant_message})
        
        return (
            assistant_message,
            self.wrapped_client.total_input_tokens,
            self.wrapped_client.total_output_tokens,
            self.wrapped_client.total_latency_ms
        )


class ArmC_ScriptedDemo:
    """
    Arm C: Scripted demo baseline (not AI).
    Uses AIHive Tutor in demo mode (no API calls).
    Labeled clearly as a scripted baseline.
    """
    
    def __init__(self):
        # Force demo mode
        self.tutor = TutorAI()
        if not self.tutor.is_demo:
            raise ValueError("ArmC must run in demo mode (OPENAI_API_KEY not set)")
        
        self.arm_a = ArmA_AIHiveTutor(openai_client=None)
    
    def start_new_lesson(self, question: MathQuestion) -> None:
        self.arm_a.start_new_lesson(question)
    
    def turn_1_solve_and_explain(self, question: MathQuestion) -> Tuple[str, int, int, int]:
        return self.arm_a.turn_1_solve_and_explain(question)
    
    def turn_2_diagnose_error(
        self,
        question: MathQuestion,
        misconception: Misconception
    ) -> Tuple[str, int, int, int]:
        return self.arm_a.turn_2_diagnose_error(question, misconception)
    
    def turn_3_practice_questions(self, question: MathQuestion) -> Tuple[str, int, int, int]:
        return self.arm_a.turn_3_practice_questions(question)
