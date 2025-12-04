import json
import os
import argparse
import time
from typing import List, Dict, Optional
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path

# CLaRa Specific Prompt Template
PROMPT_TEMPLATE = """# Role
You are a data synthesis expert for a bilingual (English/Chinese) Retrieval Augmented Generation (RAG) system called CLaRa.
Your task is to read the provided [TEXT CHUNK] and generate high-quality training data that supports CROSS-LINGUAL retrieval.

# Input
[TEXT CHUNK]:
{{TEXT_CHUNK}}

# Important Instructions
- **Focus on the actual business content** - features, processes, requirements, architecture, workflows, etc.
- **You can generate questions about images** when they illustrate business concepts (e.g., architecture diagrams, process flows, UI mockups)
- **Do NOT generate questions about technical artifacts** like:
  - File paths or folder names (e.g., "What is in the extracted_assets folder?")
  - Image file names (e.g., "What is xxx.png?")
  - Document structure markers
- Images descriptions are provided inline to help you understand the complete context

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
    parser = argparse.ArgumentParser(description="Synthesize CLaRa training data using LLM API")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input text file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the split training files")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"), help="API Key")
    parser.add_argument("--base_url", type=str, default=os.getenv("BASE_URL", "https://api.openai.com/v1"), help="API Base URL")
    parser.add_argument("--model", type=str, default="qwen-turbo", help="Model name")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Character count per chunk")
    return parser.parse_args()

def load_image_descriptions(input_file: str) -> Dict[str, str]:
    """
    Load all image descriptions from the input file.

    Returns:
        Dict mapping image filename -> description content
    """
    image_descs = {}

    if not os.path.exists(input_file):
        return image_descs

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get('source_type') == 'image' and 'filename' in entry and 'content' in entry:
                    # Keep the full description
                    image_descs[entry['filename']] = entry['content']
            except json.JSONDecodeError:
                continue

    return image_descs

def replace_image_refs_with_descriptions(chunk: str, image_descs: Dict[str, str]) -> str:
    """
    Replace [IMAGE_REF: path] markers with actual image descriptions.

    This enriches the document with semantic image content instead of technical paths.
    """
    import re

    # Remove "--- Extracted Images ---" section and everything after (that's just index)
    if "--- Extracted Images ---" in chunk:
        chunk = chunk.split("--- Extracted Images ---")[0]

    # Find all [IMAGE_REF: ...] markers
    def replace_ref(match):
        full_path = match.group(1)
        # Extract filename from path (e.g., "example/extracted_assets/xxx.png" -> "xxx.png")
        filename = os.path.basename(full_path)

        # Try to find the description
        if filename in image_descs:
            # Return the full description
            return f"\n\n{image_descs[filename]}\n\n"
        else:
            # Fallback: use placeholder if description not found
            return "[图片]"

    # Replace all [IMAGE_REF: path] with descriptions
    chunk = re.sub(r'\[IMAGE_REF:\s*([^\]]+)\]', replace_ref, chunk)

    # Clean up excessive whitespace
    chunk = re.sub(r'\n{3,}', '\n\n', chunk)
    chunk = chunk.strip()

    return chunk

def generate_data(client: OpenAI, model: str, chunk: str, image_descs: Dict[str, str]) -> Optional[Dict]:
    # Replace [IMAGE_REF: ...] with actual image descriptions
    enriched_chunk = replace_image_refs_with_descriptions(chunk, image_descs)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful data synthesis assistant. Always output valid JSON."},
                {"role": "user", "content": PROMPT_TEMPLATE.replace("{{TEXT_CHUNK}}", enriched_chunk)}
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

def main():
    args = parse_args()
    
    if not args.api_key:
        raise ValueError("Please provide --api_key or set OPENAI_API_KEY env var.")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # Prepare output files
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    f_pretrain = open(output_path / "pretrain_data.jsonl", 'w', encoding='utf-8')
    f_instruct = open(output_path / "instruction_data.jsonl", 'w', encoding='utf-8')
    f_e2e = open(output_path / "end_to_end_data.jsonl", 'w', encoding='utf-8')

    # Load image descriptions first (for enriching document chunks)
    print(f"Loading image descriptions from {args.input_file}...")
    image_descriptions = load_image_descriptions(args.input_file)
    print(f"Loaded {len(image_descriptions)} image descriptions")

    # Read input (excluding standalone image entries)
    print(f"Reading documents from {args.input_file}...")

    all_texts = []

    if args.input_file.endswith('.jsonl'):
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # Skip standalone image description entries (they'll be merged into docs via IMAGE_REF)
                    if entry.get("source_type") == "image":
                        continue
                    if "content" in entry and entry["content"].strip():
                        all_texts.append((entry.get("filename", "unknown"), entry["content"]))
                except json.JSONDecodeError:
                    pass
    else:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
            all_texts.append((os.path.basename(args.input_file), content))

    # Chunking
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

    for i, chunk in enumerate(tqdm(chunks, desc="Synthesizing")):
        if len(chunk.strip()) < 50:
            continue

        data = generate_data(client, args.model, chunk, image_descriptions)
        
        if data:
            # 1. Stage 1 Data (SCP/Pretrain)
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
                    qa_entry = {
                        "question": qa.get("question"),
                        "docs": [chunk], 
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

    print(f"Done! Data split into 3 files in {args.output_dir}")

if __name__ == "__main__":
    main()