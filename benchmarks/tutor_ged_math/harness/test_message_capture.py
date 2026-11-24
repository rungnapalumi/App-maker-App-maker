"""
Test to verify what messages each arm sends to the model
"""

import pytest
from benchmarks.tutor_ged_math.items.questions import Q1
from benchmarks.tutor_ged_math.harness.tutor_arms import (
    MockOpenAIClient,
    ArmA_AIHiveTutor,
    ArmB_GenericAssistant
)


class MessageCapturingClient:
    """
    Mock client that captures exactly what messages are sent
    """
    
    def __init__(self):
        self.captured_calls = []
        
    class ChatCompletions:
        def __init__(self, parent):
            self.parent = parent
            
        def create(self, model, messages, temperature, max_tokens, **kwargs):
            """Capture messages and return mock response"""
            # Store this call
            self.parent.captured_calls.append({
                "model": model,
                "messages": messages.copy(),  # Copy to preserve state
                "temperature": temperature,
                "max_tokens": max_tokens
            })
            
            # Mock response
            mock_text = "[CAPTURED] Mock response"
            
            class MockUsage:
                prompt_tokens = 100
                completion_tokens = 50
            
            class MockMessage:
                content = mock_text
            
            class MockChoice:
                message = MockMessage()
            
            class MockResponse:
                choices = [MockChoice()]
                usage = MockUsage()
            
            return MockResponse()
    
    @property
    def chat(self):
        class Chat:
            completions = MessageCapturingClient.ChatCompletions(self)
        return Chat()


def test_arm_a_does_not_see_question_text():
    """
    Test that Arm A (following real app) does NOT send question text to model.
    This is the fundamental limitation documented in the reports.
    
    Forces Tutor into real-AI mode with a mock client so test actually runs.
    """
    client = MessageCapturingClient()
    arm = ArmA_AIHiveTutor(openai_client=client, force_ai_mode=True)
    
    question = Q1  # "Maria bought 3 notebooks for $2.45 each..."
    
    arm.start_new_lesson(question)
    response, _, _, _ = arm.turn_1_solve_and_explain(question)
    
    # Check all captured messages
    all_message_text = ""
    for call in client.captured_calls:
        for msg in call["messages"]:
            all_message_text += msg.get("content", "").lower()
    
    # Question-specific text should NOT appear
    assert "maria" not in all_message_text, "Question text leaked to Arm A!"
    assert "notebooks" not in all_message_text, "Question text leaked to Arm A!"
    assert "2.45" not in all_message_text, "Question text leaked to Arm A!"
    
    # Only topic and level should appear
    assert "ged math" in all_message_text or "decimal" in all_message_text
    assert "intermediate" in all_message_text
    
    print("\n✓ Confirmed: Arm A never sees the question text")
    print(f"  Topic sent: 'GED Math - {question.skill}'")
    print(f"  Level sent: 'Intermediate'")
    print(f"  Question text NOT sent")


def test_arm_b_does_see_question_text():
    """
    Test that Arm B DOES send the full question text to model.
    """
    client = MessageCapturingClient()
    arm = ArmB_GenericAssistant(openai_client=client)
    
    question = Q1  # "Maria bought 3 notebooks for $2.45 each..."
    
    arm.start_new_lesson(question)
    response, _, _, _ = arm.turn_1_solve_and_explain(question)
    
    # Check all captured messages
    all_message_text = ""
    for call in client.captured_calls:
        for msg in call["messages"]:
            all_message_text += msg.get("content", "").lower()
    
    # Question text SHOULD appear
    assert "maria" in all_message_text, "Arm B should see question text!"
    assert "notebooks" in all_message_text, "Arm B should see question text!"
    assert "2.45" in all_message_text, "Arm B should see question text!"
    
    print("\n✓ Confirmed: Arm B sees the full question text")


def test_arm_a_turn_counts():
    """
    Test that Arm A makes the correct number of API calls per turn.
    Forces Tutor into real-AI mode so test actually runs.
    """
    client = MessageCapturingClient()
    arm = ArmA_AIHiveTutor(openai_client=client, force_ai_mode=True)
    
    question = Q1
    misconception = question.misconceptions[0]
    
    # Turn 1: should make 1 call (start_lesson)
    arm.start_new_lesson(question)
    client.captured_calls.clear()
    arm.turn_1_solve_and_explain(question)
    assert len(client.captured_calls) == 1, f"Turn 1 should make 1 call, made {len(client.captured_calls)}"
    
    # Turn 2: should make 2 calls (process_answer + continue_lesson)
    client.captured_calls.clear()
    arm.turn_2_diagnose_error(question, misconception)
    assert len(client.captured_calls) == 2, f"Turn 2 should make 2 calls, made {len(client.captured_calls)}"
    
    # Turn 3: should make 2 calls (summary + practice questions)
    client.captured_calls.clear()
    arm.turn_3_practice_questions(question)
    assert len(client.captured_calls) == 2, f"Turn 3 should make 2 calls, made {len(client.captured_calls)}"
    
    print("\n✓ Confirmed: Arm A API call counts")
    print("  Turn 1: 1 call (start_lesson)")
    print("  Turn 2: 2 calls (process_answer + continue_lesson)")
    print("  Turn 3: 2 calls (summary + practice)")


def test_arm_b_turn_counts():
    """
    Test that Arm B makes one call per turn (general-purpose assistant pattern).
    """
    client = MessageCapturingClient()
    arm = ArmB_GenericAssistant(openai_client=client)
    
    question = Q1
    misconception = question.misconceptions[0]
    
    # Turn 1: should make 1 call
    arm.start_new_lesson(question)
    client.captured_calls.clear()
    arm.turn_1_solve_and_explain(question)
    assert len(client.captured_calls) == 1, f"Turn 1 should make 1 call, made {len(client.captured_calls)}"
    
    # Turn 2: should make 1 call (combined 700 tokens)
    client.captured_calls.clear()
    arm.turn_2_diagnose_error(question, misconception)
    assert len(client.captured_calls) == 1, f"Turn 2 should make 1 call, made {len(client.captured_calls)}"
    
    # Turn 3: should make 1 call (combined 600 tokens)
    client.captured_calls.clear()
    arm.turn_3_practice_questions(question)
    assert len(client.captured_calls) == 1, f"Turn 3 should make 1 call, made {len(client.captured_calls)}"
    
    print("\n✓ Confirmed: Arm B makes one call per turn")
    print("  Turn 1: 1 call (500 tokens)")
    print("  Turn 2: 1 call (700 tokens)")
    print("  Turn 3: 1 call (600 tokens)")


def test_equivalent_capacity():
    """
    Test that both arms have equivalent capacity (total max_tokens per turn).
    
    Arm A makes 5 calls total (1+2+2).
    Arm B makes 3 calls total (1+1+1), with combined max_tokens.
    
    Both arms have same total capacity per turn.
    """
    # Arm A settings
    arm_a_expected = [
        # Turn 1: 1 call
        {"temp": 0.7, "max_tokens": 500},  # Total: 500
        # Turn 2: 2 calls
        {"temp": 0.6, "max_tokens": 200},  # process_answer
        {"temp": 0.7, "max_tokens": 500},  # continue_lesson (Total: 700)
        # Turn 3: 2 calls
        {"temp": 0.6, "max_tokens": 300},  # summary
        {"temp": 0.7, "max_tokens": 300},  # practice (Total: 600)
    ]
    
    # Arm B settings (one call per turn)
    arm_b_expected = [
        {"temp": 0.7, "max_tokens": 500},  # Turn 1
        {"temp": 0.7, "max_tokens": 700},  # Turn 2 (200+500 combined)
        {"temp": 0.7, "max_tokens": 600},  # Turn 3 (300+300 combined)
    ]
    
    question = Q1
    misconception = question.misconceptions[0]
    
    # Test Arm A
    client_a = MessageCapturingClient()
    arm_a = ArmA_AIHiveTutor(openai_client=client_a, force_ai_mode=True)
    
    arm_a.start_new_lesson(question)
    arm_a.turn_1_solve_and_explain(question)
    arm_a.turn_2_diagnose_error(question, misconception)
    arm_a.turn_3_practice_questions(question)
    
    assert len(client_a.captured_calls) == 5, f"Arm A: Should have 5 calls, got {len(client_a.captured_calls)}"
    
    # Verify Arm A settings
    for i, (call, expected) in enumerate(zip(client_a.captured_calls, arm_a_expected)):
        assert call["temperature"] == expected["temp"], \
            f"Arm A Call {i+1}: temperature {call['temperature']} != {expected['temp']}"
        assert call["max_tokens"] == expected["max_tokens"], \
            f"Arm A Call {i+1}: max_tokens {call['max_tokens']} != {expected['max_tokens']}"
    
    # Test Arm B
    client_b = MessageCapturingClient()
    arm_b = ArmB_GenericAssistant(openai_client=client_b)
    
    arm_b.start_new_lesson(question)
    arm_b.turn_1_solve_and_explain(question)
    arm_b.turn_2_diagnose_error(question, misconception)
    arm_b.turn_3_practice_questions(question)
    
    assert len(client_b.captured_calls) == 3, f"Arm B: Should have 3 calls, got {len(client_b.captured_calls)}"
    
    # Verify Arm B settings
    for i, (call, expected) in enumerate(zip(client_b.captured_calls, arm_b_expected)):
        assert call["temperature"] == expected["temp"], \
            f"Arm B Call {i+1}: temperature {call['temperature']} != {expected['temp']}"
        assert call["max_tokens"] == expected["max_tokens"], \
            f"Arm B Call {i+1}: max_tokens {call['max_tokens']} != {expected['max_tokens']}"
    
    # Verify equivalent capacity per turn
    turn_1_capacity = 500  # Arm A: 500, Arm B: 500
    turn_2_capacity = 700  # Arm A: 200+500, Arm B: 700
    turn_3_capacity = 600  # Arm A: 300+300, Arm B: 600
    
    print("\n✓ Confirmed: Equivalent capacity per turn")
    print(f"  Turn 1: {turn_1_capacity} tokens (Arm A: 1 call, Arm B: 1 call)")
    print(f"  Turn 2: {turn_2_capacity} tokens (Arm A: 2 calls, Arm B: 1 call)")
    print(f"  Turn 3: {turn_3_capacity} tokens (Arm A: 2 calls, Arm B: 1 call)")
    print(f"  Total: Arm A makes 5 calls, Arm B makes 3 calls")
