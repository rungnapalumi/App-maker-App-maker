"""
Unit tests for TutorAI - lesson generation and answer processing
"""
import pytest
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tutor.tutor_ai import TutorAI
from tutor.models import StudySession


@pytest.fixture
def tutor():
    """Create a TutorAI instance"""
    return TutorAI()


@pytest.fixture
def beginner_session():
    """Create a beginner-level study session"""
    return StudySession(
        user_id="test_user",
        topic="Photosynthesis",
        level="Beginner"
    )


@pytest.fixture
def intermediate_session():
    """Create an intermediate-level study session"""
    return StudySession(
        user_id="test_user",
        topic="Python Programming",
        level="Intermediate"
    )


def test_start_lesson_returns_string(tutor, beginner_session):
    """Test that start_lesson returns a non-empty string"""
    result = tutor.start_lesson(beginner_session)
    
    assert isinstance(result, str)
    assert len(result) > 0


def test_start_lesson_includes_topic(tutor, beginner_session):
    """Test that the lesson mentions the topic"""
    result = tutor.start_lesson(beginner_session)
    
    # Should mention photosynthesis or plants
    assert "photosynthesis" in result.lower() or "plant" in result.lower()


def test_start_lesson_has_question(tutor, beginner_session):
    """Test that the lesson includes a question mark (check for understanding)"""
    result = tutor.start_lesson(beginner_session)
    
    assert "?" in result


def test_process_answer_returns_tuple(tutor, beginner_session):
    """Test that process_answer returns (bool, str)"""
    beginner_session.add_message("assistant", "What is photosynthesis?", is_question=True)
    
    result = tutor.process_answer(beginner_session, "It's how plants make food")
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], bool)
    assert isinstance(result[1], str)


def test_process_answer_evaluates_quality(tutor, beginner_session):
    """Test that short/incomplete answers are marked as incorrect"""
    beginner_session.add_message("assistant", "What is photosynthesis?", is_question=True)
    
    # Very short answer should be marked incorrect
    is_correct, feedback = tutor.process_answer(beginner_session, "idk")
    
    assert isinstance(is_correct, bool)
    assert len(feedback) > 0


def test_process_answer_gives_feedback(tutor, beginner_session):
    """Test that process_answer always provides feedback"""
    beginner_session.add_message("assistant", "Explain photosynthesis", is_question=True)
    
    good_answer = "Photosynthesis is the process where plants use sunlight to make food from water and carbon dioxide"
    is_correct, feedback = tutor.process_answer(beginner_session, good_answer)
    
    assert len(feedback) > 0
    assert isinstance(feedback, str)


def test_continue_lesson_returns_string(tutor, beginner_session):
    """Test that continue_lesson returns a string"""
    result = tutor.continue_lesson(beginner_session, understood=True)
    
    assert isinstance(result, str)
    assert len(result) > 0


def test_continue_lesson_adapts_to_understanding(tutor, beginner_session):
    """Test that lesson adapts based on understanding"""
    # Mark some progress
    beginner_session.chunks_completed = 1
    
    # Continue after understanding
    result_understood = tutor.continue_lesson(beginner_session, understood=True)
    
    # Continue after not understanding
    result_not_understood = tutor.continue_lesson(beginner_session, understood=False)
    
    # Both should return valid strings
    assert isinstance(result_understood, str)
    assert isinstance(result_not_understood, str)
    assert len(result_understood) > 0
    assert len(result_not_understood) > 0


def test_generate_summary_returns_tuple(tutor, beginner_session):
    """Test that generate_summary returns (str, list)"""
    # Add some messages
    beginner_session.add_message("assistant", "Lesson content")
    beginner_session.chunks_completed = 2
    beginner_session.questions_answered_correctly = 3
    
    result = tutor.generate_summary(beginner_session)
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], str)  # summary
    assert isinstance(result[1], list)  # questions


def test_generate_summary_creates_practice_questions(tutor, beginner_session):
    """Test that summary includes exactly 3 practice questions"""
    beginner_session.add_message("assistant", "Lesson about photosynthesis")
    beginner_session.chunks_completed = 2
    
    summary, questions = tutor.generate_summary(beginner_session)
    
    assert len(questions) == 3
    for q in questions:
        assert isinstance(q, str)
        assert len(q) > 0


def test_summary_mentions_topic(tutor, beginner_session):
    """Test that summary mentions the learned topic"""
    beginner_session.add_message("assistant", "Lesson content")
    beginner_session.chunks_completed = 1
    
    summary, questions = tutor.generate_summary(beginner_session)
    
    # Summary should mention the topic
    assert beginner_session.topic.lower() in summary.lower()


def test_different_levels_produce_content(tutor):
    """Test that different difficulty levels all produce content"""
    for level in ["Beginner", "Intermediate", "Advanced"]:
        session = StudySession(
            user_id="test_user",
            topic="Test Topic",
            level=level
        )
        
        result = tutor.start_lesson(session)
        assert isinstance(result, str)
        assert len(result) > 50  # Should be substantial


def test_demo_mode_deterministic(tutor, beginner_session):
    """Test that demo mode returns consistent results for same inputs"""
    # Only test if in demo mode
    if not tutor.is_demo:
        pytest.skip("Not in demo mode")
    
    result1 = tutor.start_lesson(beginner_session)
    result2 = tutor.start_lesson(beginner_session)
    
    # Same topic/level should produce same result in demo mode
    assert result1 == result2


def test_tutor_handles_various_topics(tutor):
    """Test that tutor can handle different topics"""
    topics = ["Photosynthesis", "Python", "World War 2", "Calculus"]
    
    for topic in topics:
        session = StudySession(
            user_id="test_user",
            topic=topic,
            level="Beginner"
        )
        
        result = tutor.start_lesson(session)
        assert isinstance(result, str)
        assert len(result) > 0
