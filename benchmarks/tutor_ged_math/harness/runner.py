"""
Benchmark Runner
Executes all lessons for each arm with safety controls
"""

import argparse
import json
import random
import csv
import os
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

from benchmarks.tutor_ged_math.items.questions import ALL_QUESTIONS
from benchmarks.tutor_ged_math.harness.lesson_protocol import execute_lesson
from benchmarks.tutor_ged_math.harness.tutor_arms import (
    ArmA_AIHiveTutor,
    ArmB_GenericAssistant,
    ArmC_ScriptedDemo,
    MockOpenAIClient
)
from benchmarks.tutor_ged_math.rubric.rubric import calculate_cost, RUBRIC_VERSION


class BenchmarkRunner:
    """Runs the full benchmark with safety controls"""
    
    def __init__(
        self,
        output_dir: str,
        use_mock: bool = True,
        random_seed: int = 42,
        rubric_commit_hash: str = "UNKNOWN",
        repeats: int = 1
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_mock = use_mock
        self.random_seed = random_seed
        self.rubric_commit_hash = rubric_commit_hash
        self.repeats = repeats
        
        random.seed(random_seed)
    
    def estimate_cost(self, model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """
        Pre-estimate cost for the full benchmark.
        
        NOTE: Turn counts updated to reflect real implementation:
        - Turn 1: 1 API call
        - Turn 2: 2 API calls (process_answer + continue_lesson)
        - Turn 3: 2 API calls (summary + practice questions)
        Total: 5 API calls per question
        
        Accounts for --repeats: each lesson run N times.
        """
        questions_per_arm = 20
        api_calls_per_question = 5  # Updated: 1 + 2 + 2
        
        # Rough token estimates (will be updated after first mock run)
        avg_input_tokens_per_call = 150  # Includes system prompts and history
        avg_output_tokens_per_call = 200
        
        total_calls = questions_per_arm * api_calls_per_question * self.repeats
        total_input_tokens = total_calls * avg_input_tokens_per_call
        total_output_tokens = total_calls * avg_output_tokens_per_call
        
        cost_per_arm, source, verified = calculate_cost(
            model, total_input_tokens, total_output_tokens
        )
        
        return {
            "model": model,
            "questions_per_arm": questions_per_arm,
            "repeats": self.repeats,
            "api_calls_per_question": api_calls_per_question,
            "total_api_calls_per_arm": total_calls,
            "estimated_input_tokens_per_arm": total_input_tokens,
            "estimated_output_tokens_per_arm": total_output_tokens,
            "estimated_cost_per_arm_usd": cost_per_arm,
            "estimated_cost_both_arms_usd": cost_per_arm * 2,
            "price_source": source,
            "price_verified": verified,
            "note": "Estimates are rough; actual costs tracked from API response.usage"
        }
    
    def run_arm(
        self,
        arm_name: str,
        tutor_arm,
        questions: List[Any]
    ) -> List[Dict[str, Any]]:
        """Run all lessons for one arm"""
        results = []
        
        for question in questions:
            # Pick first misconception for each question
            misconception = question.misconceptions[0]
            
            print(f"  Running {arm_name}: {question.id} ({question.skill})...")
            
            # Execute lesson
            execution = execute_lesson(tutor_arm, question, misconception)
            
            # Calculate cost
            cost_usd, price_source, price_verified = calculate_cost(
                "gpt-4o-mini",  # TODO: get from tutor
                execution.total_input_tokens,
                execution.total_output_tokens
            )
            
            result = {
                "arm": arm_name,
                "question_id": question.id,
                "misconception_id": misconception.id,
                "turn_1_response": execution.turn_1.assistant_response,
                "turn_2_response": execution.turn_2.assistant_response,
                "turn_3_response": execution.turn_3.assistant_response,
                "total_input_tokens": execution.total_input_tokens,
                "total_output_tokens": execution.total_output_tokens,
                "total_latency_ms": execution.total_latency_ms,
                "cost_usd": cost_usd,
                "price_source": price_source,
                "price_verified": price_verified
            }
            
            results.append(result)
        
        return results
    
    def export_blinded_grading_sheet(self, all_results: List[Dict[str, Any]]) -> str:
        """
        Export results to CSV with blinding:
        - Random arm labels (A, B, C -> X, Y, Z etc.)
        - Shuffled order
        - Arm mapping saved separately
        """
        # Create arm mapping
        unique_arms = sorted(set(r["arm"] for r in all_results))
        arm_labels = ["X", "Y", "Z", "W"]
        arm_mapping = {arm: arm_labels[i] for i, arm in enumerate(unique_arms)}
        
        # Save arm mapping separately
        mapping_file = self.output_dir / "arm_key.json"
        with open(mapping_file, "w") as f:
            json.dump({
                "arm_mapping": arm_mapping,
                "random_seed": self.random_seed,
                "created_at": datetime.now().isoformat()
            }, f, indent=2)
        
        # Prepare blinded results
        blinded = []
        for r in all_results:
            blinded.append({
                "lesson_id": r["question_id"],
                "arm_label": arm_mapping[r["arm"]],
                "misconception_id": r["misconception_id"],
                "turn_1_response": r["turn_1_response"],
                "turn_2_response": r["turn_2_response"],
                "turn_3_response": r["turn_3_response"]
            })
        
        # Shuffle
        random.shuffle(blinded)
        
        # Export to CSV
        grading_file = self.output_dir / "grading_sheet.csv"
        with open(grading_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "lesson_id", "arm_label", "misconception_id",
                "turn_1_response", "turn_2_response", "turn_3_response"
            ])
            writer.writeheader()
            writer.writerows(blinded)
        
        return str(grading_file)
    
    def export_cost_table(self, all_results: List[Dict[str, Any]]) -> str:
        """Export detailed cost breakdown"""
        cost_file = self.output_dir / "cost_breakdown.csv"
        
        with open(cost_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "arm", "question_id", "input_tokens", "output_tokens",
                "latency_ms", "cost_usd", "price_source", "price_verified"
            ])
            writer.writeheader()
            
            for r in all_results:
                writer.writerow({
                    "arm": r["arm"],
                    "question_id": r["question_id"],
                    "input_tokens": r["total_input_tokens"],
                    "output_tokens": r["total_output_tokens"],
                    "latency_ms": r["total_latency_ms"],
                    "cost_usd": r["cost_usd"],
                    "price_source": r["price_source"],
                    "price_verified": r["price_verified"]
                })
        
        return str(cost_file)
    
    def run(self, include_arm_c: bool = False):
        """
        Run the full benchmark.
        
        Args:
            include_arm_c: Whether to include scripted demo baseline
        """
        print(f"\n{'='*60}")
        print("AIHive Tutor GED Math Benchmark")
        print(f"{'='*60}")
        print(f"Output directory: {self.output_dir}")
        print(f"Random seed: {self.random_seed}")
        print(f"Rubric version: {RUBRIC_VERSION}")
        print(f"Rubric commit: {self.rubric_commit_hash}")
        print(f"Mock mode: {self.use_mock}")
        print(f"{'='*60}\n")
        
        # Create clients
        if self.use_mock:
            print("Using MOCK client (no real API calls)\n")
            openai_client = MockOpenAIClient()
        else:
            print("Using REAL OpenAI client\n")
            import openai
            openai_client = openai.OpenAI()
        
        # Initialize arms
        print("Initializing arms...")
        # Force AI mode in mock runs so token counting works
        arm_a = ArmA_AIHiveTutor(openai_client=openai_client, force_ai_mode=self.use_mock)
        arm_b = ArmB_GenericAssistant(openai_client=openai_client)
        
        arms = [
            ("Arm_A_AIHive", arm_a),
            ("Arm_B_Generic", arm_b)
        ]
        
        if include_arm_c:
            try:
                arm_c = ArmC_ScriptedDemo()
                arms.append(("Arm_C_Scripted", arm_c))
            except ValueError as e:
                print(f"Warning: Could not initialize Arm C: {e}")
        
        print(f"Running {len(arms)} arms on {len(ALL_QUESTIONS)} questions\n")
        
        # Run all arms
        all_results = []
        
        for arm_name, arm_instance in arms:
            print(f"\n{'='*60}")
            print(f"Running {arm_name}")
            print(f"{'='*60}\n")
            
            arm_results = self.run_arm(arm_name, arm_instance, ALL_QUESTIONS)
            all_results.extend(arm_results)
        
        # Export results
        print(f"\n{'='*60}")
        print("Exporting results...")
        print(f"{'='*60}\n")
        
        # Save raw results
        raw_file = self.output_dir / "raw_results.json"
        with open(raw_file, "w") as f:
            json.dump({
                "metadata": {
                    "rubric_version": RUBRIC_VERSION,
                    "rubric_commit_hash": self.rubric_commit_hash,
                    "random_seed": self.random_seed,
                    "run_timestamp": datetime.now().isoformat(),
                    "mock_mode": self.use_mock
                },
                "results": all_results
            }, f, indent=2)
        print(f"✓ Raw results: {raw_file}")
        
        # Export blinded grading sheet
        grading_file = self.export_blinded_grading_sheet(all_results)
        print(f"✓ Blinded grading sheet: {grading_file}")
        print(f"✓ Arm mapping (keep secret): {self.output_dir / 'arm_key.json'}")
        
        # Export cost breakdown
        cost_file = self.export_cost_table(all_results)
        print(f"✓ Cost breakdown: {cost_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}\n")
        
        for arm_name, _ in arms:
            arm_results = [r for r in all_results if r["arm"] == arm_name]
            total_cost = sum(r["cost_usd"] for r in arm_results)
            avg_latency = sum(r["total_latency_ms"] for r in arm_results) / len(arm_results)
            
            print(f"{arm_name}:")
            print(f"  Lessons: {len(arm_results)}")
            print(f"  Total cost: ${total_cost:.4f}")
            print(f"  Avg latency: {avg_latency:.0f}ms")
            print()
        
        print(f"All results saved to: {self.output_dir}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run AIHive Tutor GED Math Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
COST SAFETY:
  By default, runs in MOCK mode (no real API calls).
  To run with real API:
    1. Set OPENAI_API_KEY environment variable
    2. Use --confirm-live flag
    3. Use --max-budget-usd to set spending limit
  
  Example: python runner.py --confirm-live --max-budget-usd 5.0
        """
    )
    
    parser.add_argument(
        "--output-dir",
        default="benchmarks/tutor_ged_math/results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Confirm you want to make REAL API calls (not mock)"
    )
    
    parser.add_argument(
        "--max-budget-usd",
        type=float,
        default=0.0,
        help="Maximum budget in USD for API calls (required with --confirm-live)"
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--include-arm-c",
        action="store_true",
        help="Include Arm C (scripted demo baseline)"
    )
    
    parser.add_argument(
        "--rubric-commit",
        default="PENDING",
        help="Git commit hash of rubric (for auditability)"
    )
    
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to repeat each lesson (default 1). With temperature 0.7, one run is noisy."
    )
    
    args = parser.parse_args()
    
    # Safety check
    use_mock = True
    if args.confirm_live:
        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: --confirm-live requires OPENAI_API_KEY environment variable")
            return 1
        
        # Check budget
        if args.max_budget_usd <= 0:
            print("ERROR: --confirm-live requires --max-budget-usd > 0")
            return 1
        
        # Estimate cost
        runner_temp = BenchmarkRunner(args.output_dir)
        estimate = runner_temp.estimate_cost()
        
        print("\n" + "="*60)
        print("COST ESTIMATE")
        print("="*60)
        print(f"Model: {estimate['model']}")
        print(f"Estimated cost (both arms): ${estimate['estimated_cost_both_arms_usd']:.4f}")
        print(f"Your budget: ${args.max_budget_usd:.2f}")
        print(f"Price source: {estimate['price_source']}")
        print(f"Price verified: {estimate['price_verified']}")
        print("="*60 + "\n")
        
        if estimate['estimated_cost_both_arms_usd'] > args.max_budget_usd:
            print(f"ERROR: Estimated cost (${estimate['estimated_cost_both_arms_usd']:.4f}) exceeds budget (${args.max_budget_usd:.2f})")
            return 1
        
        if not estimate['price_verified']:
            print("WARNING: Prices are UNVERIFIED. Actual cost may differ.")
            response = input("Continue anyway? (yes/no): ")
            if response.lower() != "yes":
                print("Aborted.")
                return 1
        
        use_mock = False
        print("Proceeding with REAL API calls...\n")
    else:
        print("\nRunning in MOCK mode (no real API calls)")
        print("To run with real API, use: --confirm-live --max-budget-usd X.XX\n")
    
    # Run benchmark
    runner = BenchmarkRunner(
        output_dir=args.output_dir,
        use_mock=use_mock,
        random_seed=args.random_seed,
        rubric_commit_hash=args.rubric_commit,
        repeats=args.repeats
    )
    
    if args.repeats > 1:
        print(f"\nNote: Running {args.repeats} repeats per lesson (temperature 0.7 is noisy)\n")
    
    runner.run(include_arm_c=args.include_arm_c)
    
    return 0


if __name__ == "__main__":
    exit(main())
