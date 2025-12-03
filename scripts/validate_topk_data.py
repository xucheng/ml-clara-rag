"""
Validate CLaRa training data format and check top-k compatibility.

Usage:
    python scripts/validate_topk_data.py --input_file example/end_to_end_data.jsonl
"""

import json
import argparse
from collections import Counter
from pathlib import Path

def validate_sample(sample: dict, line_num: int) -> list:
    """Validate a single training sample and return list of issues."""
    issues = []

    # Check required fields
    required_fields = ["question", "docs", "gold_answer"]
    for field in required_fields:
        if field not in sample:
            issues.append(f"Line {line_num}: Missing required field '{field}'")
            return issues  # Can't continue validation without required fields

    # Check field types
    if not isinstance(sample["question"], str):
        issues.append(f"Line {line_num}: 'question' must be a string, got {type(sample['question'])}")

    if not isinstance(sample["docs"], list):
        issues.append(f"Line {line_num}: 'docs' must be a list, got {type(sample['docs'])}")
    else:
        # Check docs content
        if len(sample["docs"]) == 0:
            issues.append(f"Line {line_num}: 'docs' list is empty")
        else:
            for i, doc in enumerate(sample["docs"]):
                if not isinstance(doc, str):
                    issues.append(f"Line {line_num}: docs[{i}] must be a string")
                elif len(doc.strip()) == 0:
                    issues.append(f"Line {line_num}: docs[{i}] is empty or whitespace-only")

    if not isinstance(sample["gold_answer"], str):
        issues.append(f"Line {line_num}: 'gold_answer' must be a string, got {type(sample['gold_answer'])}")
    elif len(sample["gold_answer"].strip()) == 0:
        issues.append(f"Line {line_num}: 'gold_answer' is empty")

    return issues

def main():
    parser = argparse.ArgumentParser(description="Validate CLaRa training data format")
    parser.add_argument("--input_file", type=str, required=True, help="Path to JSONL file to validate")
    parser.add_argument("--expected_top_k", type=int, help="Expected top-k value (optional)")
    parser.add_argument("--verbose", action="store_true", help="Show all issues (default: first 10)")
    args = parser.parse_args()

    if not Path(args.input_file).exists():
        print(f"❌ Error: File not found: {args.input_file}")
        return

    print(f"🔍 Validating: {args.input_file}")
    print("=" * 60)
    print()

    samples = []
    parse_errors = []
    validation_issues = []
    docs_count_distribution = Counter()

    # Read and parse file
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
                samples.append(sample)

                # Validate sample
                issues = validate_sample(sample, line_num)
                validation_issues.extend(issues)

                # Track docs count distribution
                if "docs" in sample and isinstance(sample["docs"], list):
                    docs_count_distribution[len(sample["docs"])] += 1

            except json.JSONDecodeError as e:
                parse_errors.append(f"Line {line_num}: JSON parse error - {e}")

    # Report statistics
    print(f"📊 Statistics:")
    print(f"   Total samples: {len(samples)}")
    print(f"   Parse errors:  {len(parse_errors)}")
    print(f"   Validation issues: {len(validation_issues)}")
    print()

    # Report docs count distribution
    if docs_count_distribution:
        print(f"📈 Documents per sample distribution:")
        for docs_count in sorted(docs_count_distribution.keys()):
            count = docs_count_distribution[docs_count]
            percentage = (count / len(samples)) * 100
            bar = "█" * int(percentage / 2)
            print(f"   {docs_count} docs: {count:4d} samples ({percentage:5.1f}%) {bar}")
        print()

        # Check consistency
        unique_counts = list(docs_count_distribution.keys())
        if len(unique_counts) > 1:
            print(f"⚠️  Warning: Inconsistent document counts detected!")
            print(f"   Found {len(unique_counts)} different document counts: {unique_counts}")
            print(f"   Recommendation: Use consistent top-k value across all samples")
            print()

    # Check against expected top-k
    if args.expected_top_k:
        print(f"🎯 Expected top-k: {args.expected_top_k}")
        matching_samples = docs_count_distribution.get(args.expected_top_k, 0)
        if matching_samples == len(samples):
            print(f"   ✅ All samples match expected top-k")
        else:
            print(f"   ❌ Only {matching_samples}/{len(samples)} samples match expected top-k")
            mismatches = [(k, v) for k, v in docs_count_distribution.items() if k != args.expected_top_k]
            if mismatches:
                print(f"   Mismatched samples:")
                for docs_count, count in mismatches:
                    print(f"      - {count} samples have {docs_count} docs (should be {args.expected_top_k})")
        print()

    # Report errors
    if parse_errors:
        print(f"❌ Parse Errors ({len(parse_errors)}):")
        for error in parse_errors[:10 if not args.verbose else None]:
            print(f"   {error}")
        if len(parse_errors) > 10 and not args.verbose:
            print(f"   ... and {len(parse_errors) - 10} more (use --verbose to see all)")
        print()

    if validation_issues:
        print(f"⚠️  Validation Issues ({len(validation_issues)}):")
        for issue in validation_issues[:10 if not args.verbose else None]:
            print(f"   {issue}")
        if len(validation_issues) > 10 and not args.verbose:
            print(f"   ... and {len(validation_issues) - 10} more (use --verbose to see all)")
        print()

    # Show sample
    if samples:
        print(f"📄 Sample (first entry):")
        print("-" * 60)
        sample = samples[0]
        print(f"Question: {sample.get('question', 'N/A')[:80]}...")
        print(f"Docs: {len(sample.get('docs', []))} documents")
        if sample.get('docs'):
            for i, doc in enumerate(sample['docs'][:3]):
                print(f"  [{i+1}] {doc[:60]}...")
            if len(sample['docs']) > 3:
                print(f"  ... and {len(sample['docs']) - 3} more")
        print(f"Answer: {sample.get('gold_answer', 'N/A')[:80]}...")
        print("-" * 60)
        print()

    # Final verdict
    if parse_errors or validation_issues:
        print("❌ Validation FAILED - Please fix the issues above")
        return 1
    else:
        print("✅ Validation PASSED - Data format is correct!")

        # Additional recommendations
        if len(docs_count_distribution) > 1:
            print("\n💡 Recommendation:")
            print("   Consider regenerating data with consistent top-k value")
            print("   Use: python scripts/synthesize_data_topk.py --target_top_k N")

        return 0

if __name__ == "__main__":
    exit(main())
