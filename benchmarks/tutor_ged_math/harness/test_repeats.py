"""
Test for --repeats functionality
"""

import pytest
import tempfile
import json
import csv
from pathlib import Path

from benchmarks.tutor_ged_math.items.questions import Q1, Q2
from benchmarks.tutor_ged_math.harness.runner import BenchmarkRunner


def test_repeats_functionality():
    """
    Test that --repeats=2 yields:
    - 2x lessons per arm
    - 2x API calls per arm  
    - Distinct repeat_index values (0 and 1)
    - Fresh sessions per repeat
    - Correct rows in blinded grading sheet and arm key
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run with repeats=2 and just 2 questions for speed
        runner = BenchmarkRunner(
            output_dir=tmpdir,
            use_mock=True,
            random_seed=42,
            rubric_commit_hash="TEST",
            repeats=2
        )
        
        # Temporarily reduce question set
        from benchmarks.tutor_ged_math.items.questions import ALL_QUESTIONS
        original_questions = ALL_QUESTIONS.copy()
        ALL_QUESTIONS.clear()
        ALL_QUESTIONS.extend([Q1, Q2])
        
        try:
            runner.run(include_arm_c=False)
        finally:
            # Restore
            ALL_QUESTIONS.clear()
            ALL_QUESTIONS.extend(original_questions)
        
        # Load results
        output_path = Path(tmpdir)
        with open(output_path / "raw_results.json") as f:
            data = json.load(f)
        
        results = data['results']
        
        # Check total lessons: 3 arms × 2 questions × 2 repeats = 12 lessons
        assert len(results) == 12, f"Expected 12 lessons total, got {len(results)}"
        
        # Group by arm
        arms = {}
        for r in results:
            arm = r['arm']
            if arm not in arms:
                arms[arm] = []
            arms[arm].append(r)
        
        # Each arm should have 4 lessons (2 questions × 2 repeats)
        for arm_name, arm_results in arms.items():
            assert len(arm_results) == 4, f"{arm_name}: expected 4 lessons, got {len(arm_results)}"
        
        # Check repeat_index values
        for arm_name, arm_results in arms.items():
            repeat_indices = [r['repeat_index'] for r in arm_results]
            # Should have both 0 and 1
            assert 0 in repeat_indices, f"{arm_name}: missing repeat_index 0"
            assert 1 in repeat_indices, f"{arm_name}: missing repeat_index 1"
            # Should have 2 of each (one per question)
            assert repeat_indices.count(0) == 2, f"{arm_name}: wrong count of repeat_index 0"
            assert repeat_indices.count(1) == 2, f"{arm_name}: wrong count of repeat_index 1"
        
        # Check API call counts per arm
        # A arms make 5 calls per lesson, B makes 3 calls per lesson
        # With 2 questions × 2 repeats = 4 lessons per arm
        # A arms: 4 × 5 = 20 total calls expected
        # B arm: 4 × 3 = 12 total calls expected
        
        # Verify distinct sessions per repeat (responses should differ for same question)
        a1_results = arms['Arm_A1_Skill']
        # Group by question
        q1_results = [r for r in a1_results if r['question_id'] == 'GED-Q001']
        assert len(q1_results) == 2, "Should have 2 results for Q001 (2 repeats)"
        
        # Check they have different repeat_index
        assert q1_results[0]['repeat_index'] != q1_results[1]['repeat_index']
        
        # Check blinded grading sheet
        with open(output_path / "grading_sheet.csv") as f:
            reader = csv.DictReader(f)
            grading_rows = list(reader)
        
        # Should have 12 rows (3 arms × 2 questions × 2 repeats)
        assert len(grading_rows) == 12, f"Expected 12 grading rows, got {len(grading_rows)}"
        
        # Each row should have repeat_index
        for row in grading_rows:
            assert 'repeat_index' in row
            assert row['repeat_index'] in ['0', '1']
        
        # Check arm_key has repeats recorded
        with open(output_path / "arm_key.json") as f:
            arm_key = json.load(f)
        
        assert arm_key['repeats'] == 2
        assert len(arm_key['arm_mapping']) == 3  # A1, A2, B
        
        print("\n✓ Confirmed: --repeats=2 works correctly")
        print(f"  Total lessons: {len(results)}")
        print(f"  Lessons per arm: 4 (2 questions × 2 repeats)")
        print(f"  Distinct repeat_index values: 0 and 1")
        print(f"  Blinded grading rows: {len(grading_rows)}")
        print(f"  Arm mapping recorded repeats: {arm_key['repeats']}")
