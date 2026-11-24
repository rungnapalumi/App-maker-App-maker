"""
Tests for benchmark harness
"""

import pytest
import tempfile
from pathlib import Path

from benchmarks.tutor_ged_math.items.questions import Q1, Q2
from benchmarks.tutor_ged_math.harness.lesson_protocol import execute_lesson
from benchmarks.tutor_ged_math.harness.tutor_arms import (
    MockOpenAIClient,
    ArmA_AIHiveTutor,
    ArmB_GenericAssistant
)
from benchmarks.tutor_ged_math.harness.runner import BenchmarkRunner


def test_mock_client_works():
    """Test that mock client returns responses"""
    client = MockOpenAIClient()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
        max_tokens=100
    )
    
    assert response.choices[0].message.content.startswith("[MOCK RESPONSE")
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0


def test_arm_a_executes_lesson():
    """Test Arm A can execute a lesson"""
    mock_client = MockOpenAIClient()
    arm = ArmA_AIHiveTutor(openai_client=mock_client, force_ai_mode=True)
    
    question = Q1
    misconception = question.misconceptions[0]
    
    execution = execute_lesson(arm, question, misconception)
    
    assert execution.question_id == question.id
    assert execution.misconception_id == misconception.id
    assert len(execution.turns) == 3
    assert execution.total_input_tokens > 0
    assert execution.total_output_tokens > 0


def test_arm_b_executes_lesson():
    """Test Arm B can execute a lesson"""
    mock_client = MockOpenAIClient()
    arm = ArmB_GenericAssistant(openai_client=mock_client)
    
    question = Q2
    misconception = question.misconceptions[0]
    
    execution = execute_lesson(arm, question, misconception)
    
    assert execution.question_id == question.id
    assert len(execution.turns) == 3
    # Arm B should have responses with MOCK prefix
    assert "[MOCK RESPONSE" in execution.turn_1.assistant_response


def test_benchmark_runner_mock_mode():
    """Test full benchmark run in mock mode"""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = BenchmarkRunner(
            output_dir=tmpdir,
            use_mock=True,
            random_seed=42,
            rubric_commit_hash="TEST_COMMIT"
        )
        
        # Run with just 2 questions (subset for speed)
        from benchmarks.tutor_ged_math.items.questions import ALL_QUESTIONS
        
        # Temporarily reduce question set
        original_questions = ALL_QUESTIONS.copy()
        ALL_QUESTIONS.clear()
        ALL_QUESTIONS.extend([Q1, Q2])
        
        try:
            runner.run(include_arm_c=False)
        finally:
            # Restore original questions
            ALL_QUESTIONS.clear()
            ALL_QUESTIONS.extend(original_questions)
        
        # Check output files exist
        output_path = Path(tmpdir)
        assert (output_path / "raw_results.json").exists()
        assert (output_path / "grading_sheet.csv").exists()
        assert (output_path / "arm_key.json").exists()
        assert (output_path / "cost_breakdown.csv").exists()


def test_cost_estimation():
    """Test cost estimation works"""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = BenchmarkRunner(output_dir=tmpdir)
        estimate = runner.estimate_cost()
        
        assert estimate["model"] == "gpt-4o-mini"
        assert estimate["questions_per_arm"] == 20
        # Note: api_calls varies by arm (A: 5, B: 3), so check repeats instead
        assert estimate["repeats"] == 1
        assert estimate["estimated_cost_both_arms_usd"] > 0
        assert estimate["price_verified"] is True  # Updated: prices are now verified


def test_turns_capture_metrics():
    """Test that turns capture token counts and latency"""
    mock_client = MockOpenAIClient()
    arm = ArmB_GenericAssistant(openai_client=mock_client)
    
    question = Q1
    misconception = question.misconceptions[0]
    
    execution = execute_lesson(arm, question, misconception)
    
    for turn in execution.turns:
        assert turn.input_tokens > 0, f"Turn {turn.turn_number} has no input tokens"
        assert turn.output_tokens > 0, f"Turn {turn.turn_number} has no output tokens"
        assert turn.latency_ms >= 0, f"Turn {turn.turn_number} has negative latency"


def test_blinding_shuffles_order():
    """Test that blinding shuffles the order of results"""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = BenchmarkRunner(output_dir=tmpdir, random_seed=42)
        
        # Create mock results
        results = [
            {"arm": "Arm_A", "question_id": f"Q{i:03d}", "repeat_index": 0, "misconception_id": f"M{i}",
             "turn_1_response": "R1", "turn_2_response": "R2", "turn_3_response": "R3"}
            for i in range(10)
        ]
        
        grading_file = runner.export_blinded_grading_sheet(results)
        
        # Read grading sheet
        import csv
        with open(grading_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Check order is different from input
        original_order = [r["question_id"] for r in results]
        grading_order = [r["lesson_id"] for r in rows]
        
        # Should be shuffled (very unlikely to be identical)
        assert original_order != grading_order
        
        # Check arm labels are anonymized
        arm_labels = set(r["arm_label"] for r in rows)
        assert "Arm_A" not in arm_labels  # Original labels should be hidden
        assert len(arm_labels) > 0  # Should have some labels
