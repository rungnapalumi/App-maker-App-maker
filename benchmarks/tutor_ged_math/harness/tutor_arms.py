"""
Tutor Arms for Benchmark
Arm A: AIHive Tutor (real implementation)
Arm B: Generic Assistant (same model, generic prompt)
Arm C: Scripted Demo (baseline, not AI)
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


class MockOpenAIClient:
    """
    Mock OpenAI client for testing.
    Returns scripted responses without making API calls.
    """
    
    def __init__(self):
        self.call_count = 0
        
    class ChatCompletions:
        def __init__(self, parent):
            self.parent = parent
            
        def create(self, model, messages, temperature, max_tokens, **kwargs):
            """Mock completion"""
            self.parent.call_count += 1
            
            # Extract last user message
            user_msg = ""
            for msg in reversed(messages):
                if msg["role"] == "user":
                    user_msg = msg["content"]
                    break
            
            # Mock response
            mock_response = f"[MOCK RESPONSE {self.parent.call_count}] This is a scripted response for testing. User asked: {user_msg[:50]}..."
            
            class MockChoice:
                class MockMessage:
                    content = mock_response
                message = MockMessage()
            
            class MockResponse:
                class MockUsage:
                    prompt_tokens = 100
                    completion_tokens = 50
                usage = MockUsage()
                choices = [MockChoice()]
            
            return MockResponse()
    
    @property
    def chat(self):
        class Chat:
            completions = MockOpenAIClient.ChatCompletions(self)
        return Chat()


class ArmA_AIHiveTutor:
    """
    Arm A: AIHive Tutor using real implementation.
    Calls the actual Tutor functions from the app.
    """
    
    def __init__(self, openai_client=None):
        """
        Args:
            openai_client: Optional OpenAI client. If None and OPENAI_API_KEY set,
                          creates real client. If mock client provided, uses it.
        """
        self.session: Optional[StudySession] = None
        self.client = openai_client
        
        # Create tutor with injected client for testing
        self.tutor = TutorAI()
        if openai_client and not self.tutor.is_demo:
            self.tutor.client = openai_client
    
    def start_new_lesson(self, question: MathQuestion) -> None:
        """Start fresh session for question"""
        self.session = StudySession(
            user_id="benchmark_user",
            topic=f"GED Math - {question.skill}",
            level="Intermediate"  # GED is roughly intermediate
        )
    
    def turn_1_solve_and_explain(self, question: MathQuestion) -> Tuple[str, int, int, int]:
        """Turn 1: Get tutor explanation"""
        start_time = time.time()
        
        # Add question to session
        formatted_q = format_question_for_tutor(question)
        self.session.add_message("user", formatted_q)
        
        # Get tutor response
        response = self.tutor.start_lesson(self.session)
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Record response
        self.session.add_message("assistant", response)
        
        # Token counts (approximate for demo mode, real for API mode)
        if self.tutor.is_demo:
            input_tokens = len(formatted_q.split()) * 1.3  # rough estimate
            output_tokens = len(response.split()) * 1.3
        else:
            # In real mode, we'd track from API response
            # For now, estimate (real harness would capture from API)
            input_tokens = len(formatted_q.split()) * 1.3
            output_tokens = len(response.split()) * 1.3
        
        return response, int(input_tokens), int(output_tokens), latency_ms
    
    def turn_2_diagnose_error(
        self,
        question: MathQuestion,
        misconception: Misconception
    ) -> Tuple[str, int, int, int]:
        """Turn 2: Diagnose student error"""
        start_time = time.time()
        
        # Student submits wrong answer
        student_msg = f"I got {misconception.wrong_answer}. {misconception.student_attempt}. What did I do wrong?"
        self.session.add_message("user", student_msg, is_question=True)
        
        # Get tutor response
        # Tutor doesn't have a specific "diagnose error" method, so we use continue_lesson
        response = self.tutor.continue_lesson(self.session, understood=False)
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Record response
        self.session.add_message("assistant", response)
        
        # Token counts
        if self.tutor.is_demo:
            input_tokens = len(student_msg.split()) * 1.3
            output_tokens = len(response.split()) * 1.3
        else:
            input_tokens = len(student_msg.split()) * 1.3
            output_tokens = len(response.split()) * 1.3
        
        return response, int(input_tokens), int(output_tokens), latency_ms
    
    def turn_3_practice_questions(self, question: MathQuestion) -> Tuple[str, int, int, int]:
        """Turn 3: Get practice questions"""
        start_time = time.time()
        
        # Request practice questions
        request = "Can you give me 2 practice questions to help me understand this better?"
        self.session.add_message("user", request)
        
        # Mark session ready for summary
        self.session.chunks_completed = 2
        
        # Get practice questions from tutor
        summary, practice_questions = self.tutor.generate_summary(self.session)
        
        # Format response
        response = summary + "\n\n**Practice Questions:**\n"
        for i, q in enumerate(practice_questions[:2], 1):  # Only first 2
            response += f"\n{i}. {q}"
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Record response
        self.session.add_message("assistant", response)
        
        # Token counts
        if self.tutor.is_demo:
            input_tokens = len(request.split()) * 1.3
            output_tokens = len(response.split()) * 1.3
        else:
            input_tokens = len(request.split()) * 1.3
            output_tokens = len(response.split()) * 1.3
        
        return response, int(input_tokens), int(output_tokens), latency_ms


class ArmB_GenericAssistant:
    """
    Arm B: Generic AI assistant.
    Same model, same temperature, but generic "helpful assistant" prompt.
    """
    
    def __init__(self, openai_client=None, model: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Args:
            openai_client: OpenAI client (required, can be mock)
            model: Model to use (default matches Tutor)
            temperature: Temperature (default matches Tutor)
        """
        if openai_client is None:
            raise ValueError("OpenAI client required for GenericAssistant")
        
        self.client = openai_client
        self.model = model
        self.temperature = temperature
        self.messages = []
    
    def start_new_lesson(self, question: MathQuestion) -> None:
        """Start fresh conversation"""
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    
    def _make_api_call(self, user_message: str, max_tokens: int = 500) -> Tuple[str, int, int, int]:
        """Make API call and track metrics"""
        start_time = time.time()
        
        self.messages.append({"role": "user", "content": user_message})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=max_tokens
        )
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        assistant_message = response.choices[0].message.content.strip()
        self.messages.append({"role": "assistant", "content": assistant_message})
        
        # Get token counts from response
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
        return assistant_message, input_tokens, output_tokens, latency_ms
    
    def turn_1_solve_and_explain(self, question: MathQuestion) -> Tuple[str, int, int, int]:
        """Turn 1: Solve and explain"""
        formatted_q = format_question_for_tutor(question)
        prompt = f"Please solve this problem and explain your solution:\n\n{formatted_q}"
        
        return self._make_api_call(prompt, max_tokens=500)
    
    def turn_2_diagnose_error(
        self,
        question: MathQuestion,
        misconception: Misconception
    ) -> Tuple[str, int, int, int]:
        """Turn 2: Diagnose error"""
        prompt = f"I got {misconception.wrong_answer}. {misconception.student_attempt}. What did I do wrong?"
        
        return self._make_api_call(prompt, max_tokens=300)
    
    def turn_3_practice_questions(self, question: MathQuestion) -> Tuple[str, int, int, int]:
        """Turn 3: Generate practice"""
        prompt = "Can you give me 2 practice questions to help me understand this better?"
        
        return self._make_api_call(prompt, max_tokens=400)


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
