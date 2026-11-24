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
    
    If running in demo mode (no API calls), skip this test.
    """
    client = MessageCapturingClient()
    arm = ArmA_AIHiveTutor(openai_client=client)
    
    # Skip if in demo mode
    if arm.tutor.is_demo:
        pytest.skip("Arm A is in demo mode (no API calls to capture)")
    
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
    """
    client = MessageCapturingClient()
    arm = ArmA_AIHiveTutor(openai_client=client)
    
    # Skip if in demo mode
    if arm.tutor.is_demo:
        pytest.skip("Arm A is in demo mode (no API calls to capture)")
    
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
    Test that Arm B makes the same number of API calls as Arm A.
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
    
    # Turn 2: should make 2 calls (to match Arm A)
    client.captured_calls.clear()
    arm.turn_2_diagnose_error(question, misconception)
    assert len(client.captured_calls) == 2, f"Turn 2 should make 2 calls, made {len(client.captured_calls)}"
    
    # Turn 3: should make 2 calls (to match Arm A)
    client.captured_calls.clear()
    arm.turn_3_practice_questions(question)
    assert len(client.captured_calls) == 2, f"Turn 3 should make 2 calls, made {len(client.captured_calls)}"
    
    print("\n✓ Confirmed: Arm B API call counts match Arm A")
    print("  Turn 1: 1 call")
    print("  Turn 2: 2 calls")
    print("  Turn 3: 2 calls")


def test_temperature_parity():
    """
    Test that both arms use the same temperature settings per call.
    """
    # Arm A settings (from code inspection)
    arm_a_settings = [
        # Turn 1
        {"temp": 0.7, "max_tokens": 500},
        # Turn 2
        {"temp": 0.6, "max_tokens": 200},  # process_answer
        {"temp": 0.7, "max_tokens": 500},  # continue_lesson
        # Turn 3
        {"temp": 0.6, "max_tokens": 300},  # summary
        {"temp": 0.7, "max_tokens": 300},  # practice
    ]
    
    client = MessageCapturingClient()
    arm_b = ArmB_GenericAssistant(openai_client=client)
    
    question = Q1
    misconception = question.misconceptions[0]
    
    # Execute all turns
    arm_b.start_new_lesson(question)
    arm_b.turn_1_solve_and_explain(question)
    arm_b.turn_2_diagnose_error(question, misconception)
    arm_b.turn_3_practice_questions(question)
    
    # Check settings match
    assert len(client.captured_calls) == 5, "Should have 5 total API calls"
    
    for i, (call, expected) in enumerate(zip(client.captured_calls, arm_a_settings)):
        assert call["temperature"] == expected["temp"], \
            f"Call {i+1}: temperature {call['temperature']} != {expected['temp']}"
        assert call["max_tokens"] == expected["max_tokens"], \
            f"Call {i+1}: max_tokens {call['max_tokens']} != {expected['max_tokens']}"
    
    print("\n✓ Confirmed: Temperature and max_tokens parity")
    print("  Call 1: temp=0.7, max_tokens=500")
    print("  Call 2: temp=0.6, max_tokens=200")
    print("  Call 3: temp=0.7, max_tokens=500")
    print("  Call 4: temp=0.6, max_tokens=300")
    print("  Call 5: temp=0.7, max_tokens=300")
