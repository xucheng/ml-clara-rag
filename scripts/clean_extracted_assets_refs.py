#!/usr/bin/env python3
"""
Clean image references from legacy synthesized data.

IMPORTANT: This script is for LEGACY data only (before image enrichment feature).
New data synthesis automatically replaces [IMAGE_REF: ...] with image descriptions.

Use this script ONLY if you have old data that:
1. Contains [IMAGE_REF: ...] markers without corresponding image descriptions
2. Was generated before the image enrichment feature was implemented

For new data synthesis, use:
    - synthesize_data.py (automatically enriches with image descriptions)
    - synthesize_data_topk.py (with image enrichment support)

Usage (for legacy data only):
    python scripts/clean_extracted_assets_refs.py \
        --input example/pretrain_data_old.jsonl \
        --output example/pretrain_data_cleaned.jsonl
"""

import json
import re
import argparse
from pathlib import Path


def clean_text(text: str) -> str:
    """
    Clean technical image references from text.

    Removes:
    - [IMAGE_REF: example/extracted_assets/xxx.png] markers
    - "--- Extracted Images ---" sections
    - Technical file path references

    Preserves:
    - Document content and meaning
    - Context around images (replaced with [图片] placeholder)
    """

    # Remove "--- Extracted Images ---" section and everything after
    if "--- Extracted Images ---" in text:
        text = text.split("--- Extracted Images ---")[0]

    # Replace [IMAGE_REF: ...] with simple placeholder
    # This preserves the document flow while removing technical details
    text = re.sub(r'\[IMAGE_REF:.*?\]', '[图片]', text)

    # Remove multiple consecutive [图片] markers (keep max 1)
    text = re.sub(r'(\[图片\]\s*){2,}', '[图片] ', text)

    # Clean up extra whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text


def clean_jsonl_file(input_file: str, output_file: str):
    """Clean a JSONL training data file."""

    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")

    cleaned_count = 0
    total_count = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            total_count += 1

            try:
                data = json.loads(line)

                # Clean different data formats
                if 'docs' in data:
                    # Stage 1/2/3 format with docs list
                    original_docs = data['docs']
                    data['docs'] = [clean_text(doc) for doc in data['docs']]

                    if original_docs != data['docs']:
                        cleaned_count += 1

                if 'question' in data:
                    # Clean question if it's a string
                    if isinstance(data['question'], str):
                        data['question'] = clean_text(data['question'])
                    elif isinstance(data['question'], list):
                        data['question'] = [clean_text(q) for q in data['question']]

                if 'answers' in data:
                    # Clean answers
                    if isinstance(data['answers'], str):
                        data['answers'] = clean_text(data['answers'])
                    elif isinstance(data['answers'], list):
                        data['answers'] = [clean_text(a) for a in data['answers']]

                if 'gold_answer' in data:
                    # Clean gold_answer
                    data['gold_answer'] = clean_text(data['gold_answer'])

                # Write cleaned data
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

            except Exception as e:
                print(f"Error processing line {total_count}: {e}")
                # Write original line if cleaning fails
                f_out.write(line)

    print(f"\n✅ Cleaned {cleaned_count}/{total_count} entries")
    print(f"📁 Output: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Clean image references from training data")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    args = parser.parse_args()

    clean_jsonl_file(args.input, args.output)


if __name__ == "__main__":
    main()
