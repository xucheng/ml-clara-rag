"""
Enhanced data synthesis script for CLaRa with top-k > 1 support.

This script generates training data where each question has multiple candidate documents:
- 1 positive document (contains the answer)
- 2-4 hard negative documents (semantically similar but don't answer the question)
- 0-1 random negative documents (topically related)

Usage:
    python scripts/synthesize_data_topk.py \
        --input_file example/raw_knowledge.jsonl \
        --output_dir example \
        --api_key $OPENAI_API_KEY \
        --target_top_k 5
"""

import json
import os
import argparse
import time
import random
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
import numpy as np

# CLaRa Specific Prompt Template (same as original)
PROMPT_TEMPLATE = """# Role
You are a data synthesis expert for a bilingual (English/Chinese) Retrieval Augmented Generation (RAG) system called CLaRa.
Your task is to read the provided [TEXT CHUNK] and generate high-quality training data that supports CROSS-LINGUAL retrieval.

# Input
[TEXT CHUNK]:
{{TEXT_CHUNK}}

# Task
Generate a JSON object with two fields:
1. "dense_summary": A rewritten paragraph that contains ALL key information from the text chunk. It should be 50-80% of the original length.
2. "qa_pairs": A list of 3-5 question-answer pairs based STRICTLY on the text.
   - **CRITICAL: You must generate questions in BOTH English and Chinese.**
   - If the text is in Chinese, include at least 1-2 questions in English.
   - If the text is in English, include at least 1-2 questions in Chinese.
   - Questions must be self-contained (no "according to the text", "he said", etc.).
   - Include at least one fact-based question and one reasoning question.

# Output Format (JSON)
{
    "dense_summary": "string",
    "qa_pairs": [
        {
            "type": "fact",
            "question": "string (in English or Chinese)",
            "answer": "string (in the same language as the text)"
        },
        {
            "type": "cross_lingual",
            "question": "string (in the OPPOSITE language of the text)",
            "answer": "string (in the same language as the text)"
        }
    ]
}
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Synthesize CLaRa training data with top-k support")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input text file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the split training files")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"), help="API Key")
    parser.add_argument("--base_url", type=str, default=os.getenv("BASE_URL", "https://api.openai.com/v1"), help="API Base URL")
    parser.add_argument("--model", type=str, default="qwen-turbo", help="Model name")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Character count per chunk")
    parser.add_argument("--target_top_k", type=int, default=5, help="Number of candidate documents per question (1-10)")
    parser.add_argument("--use_embeddings", action="store_true", help="Use embeddings for hard negative mining (requires OpenAI API)")
    return parser.parse_args()

def generate_data(client: OpenAI, model: str, chunk: str) -> Optional[Dict]:
    """Generate QA pairs from a text chunk using LLM."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful data synthesis assistant. Always output valid JSON."},
                {"role": "user", "content": PROMPT_TEMPLATE.replace("{{TEXT_CHUNK}}", chunk)}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=2048
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"\nError processing chunk: {e}")
        return None

def get_embedding(client: OpenAI, text: str, model: str = "text-embedding-v3") -> Optional[List[float]]:
    """Get text embedding using OpenAI API."""
    try:
        response = client.embeddings.create(
            model=model,
            input=text[:8000]  # Truncate to avoid token limit
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"\nError getting embedding: {e}")
        return None

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def select_negative_documents(
    positive_chunk: str,
    positive_chunk_idx: int,
    all_chunks: List[str],
    target_top_k: int,
    chunk_embeddings: Optional[List[List[float]]] = None,
    positive_embedding: Optional[List[float]] = None
) -> List[str]:
    """
    Select negative documents for a question.

    Strategy:
    - If embeddings available: select (top_k - 1) semantically similar chunks (hard negatives)
    - If no embeddings: randomly sample (top_k - 1) chunks

    Returns:
        List of document strings (length = target_top_k - 1)
    """
    num_negatives = target_top_k - 1

    if chunk_embeddings and positive_embedding:
        # Hard negative mining using embeddings
        similarities = []
        for i, (chunk, emb) in enumerate(zip(all_chunks, chunk_embeddings)):
            if i != positive_chunk_idx and emb:
                sim = cosine_similarity(positive_embedding, emb)
                similarities.append((i, sim))

        # Sort by similarity (descending) and take top-N
        similarities.sort(key=lambda x: x[1], reverse=True)
        negative_indices = [idx for idx, _ in similarities[:num_negatives]]
        negative_docs = [all_chunks[i] for i in negative_indices]
    else:
        # Random sampling
        available_indices = [i for i in range(len(all_chunks)) if i != positive_chunk_idx]
        if len(available_indices) < num_negatives:
            # Not enough chunks, use what we have
            negative_indices = available_indices
        else:
            negative_indices = random.sample(available_indices, num_negatives)
        negative_docs = [all_chunks[i] for i in negative_indices]

    return negative_docs

def main():
    args = parse_args()

    if not args.api_key:
        raise ValueError("Please provide --api_key or set OPENAI_API_KEY env var.")

    if args.target_top_k < 1 or args.target_top_k > 10:
        raise ValueError("target_top_k must be between 1 and 10")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # Prepare output files
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    f_pretrain = open(output_path / "pretrain_data.jsonl", 'w', encoding='utf-8')
    f_instruct = open(output_path / "instruction_data.jsonl", 'w', encoding='utf-8')
    f_e2e = open(output_path / "end_to_end_data.jsonl", 'w', encoding='utf-8')

    # Read input
    print(f"Reading input from {args.input_file}...")

    all_texts = []

    if args.input_file.endswith('.jsonl'):
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if "content" in entry and entry["content"].strip():
                        all_texts.append((entry.get("filename", "unknown"), entry["content"]))
                except json.JSONDecodeError:
                    pass
    else:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
            all_texts.append((os.path.basename(args.input_file), content))

    # Chunking
    print("Chunking documents...")
    chunks = []
    for filename, text in all_texts:
        current_pos = 0
        text_len = len(text)
        while current_pos < text_len:
            end_pos = min(current_pos + args.chunk_size, text_len)
            if end_pos < text_len:
                search_window = text[end_pos-100:end_pos]
                last_newline = search_window.rfind('\n')
                if last_newline != -1:
                    end_pos = (end_pos - 100) + last_newline + 1

            chunk_text = text[current_pos:end_pos].strip()
            if len(chunk_text) > 50:
                chunks.append(chunk_text)
            current_pos = end_pos

    print(f"Generated {len(chunks)} chunks from {len(all_texts)} documents.")

    # Generate embeddings if needed
    chunk_embeddings = None
    if args.use_embeddings and args.target_top_k > 1:
        print(f"Generating embeddings for {len(chunks)} chunks...")
        chunk_embeddings = []
        for chunk in tqdm(chunks, desc="Embeddings"):
            emb = get_embedding(client, chunk)
            chunk_embeddings.append(emb)
            time.sleep(0.1)  # Rate limiting

    # Process chunks and generate QA data
    print(f"Synthesizing data with top-k={args.target_top_k}...")

    for i, chunk in enumerate(tqdm(chunks, desc="Synthesizing")):
        if len(chunk.strip()) < 50:
            continue

        data = generate_data(client, args.model, chunk)

        if data:
            # 1. Stage 1 Data (SCP/Pretrain) - always single document
            scp_entry = {
                "data_type": "qa",
                "question": [f"Summarize the following text: {chunk[:50]}..."],
                "answers": [data.get("dense_summary", "")],
                "docs": [chunk]
            }
            f_pretrain.write(json.dumps(scp_entry, ensure_ascii=False) + "\n")

            # 2. Stage 2 & 3 Data (Instruction & E2E)
            if "qa_pairs" in data:
                for qa in data["qa_pairs"]:
                    # Construct candidate documents
                    if args.target_top_k == 1:
                        # Single document mode
                        candidate_docs = [chunk]
                    else:
                        # Multi-document mode
                        positive_emb = chunk_embeddings[i] if chunk_embeddings else None
                        negative_docs = select_negative_documents(
                            chunk, i, chunks, args.target_top_k,
                            chunk_embeddings, positive_emb
                        )
                        # Combine positive + negatives and shuffle
                        candidate_docs = [chunk] + negative_docs
                        random.shuffle(candidate_docs)

                    qa_entry = {
                        "question": qa.get("question"),
                        "docs": candidate_docs,  # Multiple documents!
                        "gold_answer": qa.get("answer")
                    }

                    # Write to both datasets
                    f_instruct.write(json.dumps(qa_entry, ensure_ascii=False) + "\n")
                    f_e2e.write(json.dumps(qa_entry, ensure_ascii=False) + "\n")

        time.sleep(0.5)

    # Close files
    f_pretrain.close()
    f_instruct.close()
    f_e2e.close()

    print(f"\n✅ Done! Data split into 3 files in {args.output_dir}")
    print(f"📊 Each Stage 2/3 sample has {args.target_top_k} candidate documents")
    if args.use_embeddings:
        print("🔍 Used embedding-based hard negative mining")
    else:
        print("🎲 Used random negative sampling")

if __name__ == "__main__":
    main()
