import os
import json
import base64
import argparse
import time
import io
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from PIL import Image

# Supported Image Extensions
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def validate_and_convert_image(image_path: Path) -> str:
    """
    Validates image using Pillow and converts to standardized JPEG base64.
    Returns base64 string if valid, None otherwise.
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB (removes alpha channel, fixes RGBA issues)
            if img.mode in ('RGBA', 'P', 'LA'):
                img = img.convert('RGB')
            
            # Resize if too large (optional optimization for token/speed)
            # Qwen/GPT-4 usually handle up to 2048x2048 well, but strict limits exist.
            if max(img.size) > 2048:
                img.thumbnail((2048, 2048))

            # Save to memory buffer as JPEG
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Skipping corrupt/unsupported image {image_path.name}: {e}")
        return None

def get_processed_images(output_file: str) -> set:
    """Reads output file to find which images have already been processed."""
    processed = set()
    if not os.path.exists(output_file):
        return processed
    
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # We used filename as key identifier in previous runs
                if "filename" in data:
                    processed.add(data["filename"])
            except json.JSONDecodeError:
                pass
    return processed

def analyze_image(client: OpenAI, model: str, image_path: Path):
    """Sends image to VLM API for semantic extraction."""
    
    # Validating and converting first!
    base64_image = validate_and_convert_image(image_path)
    if not base64_image:
        return None
    
    # Prompt optimized for flowchart/diagram analysis
    prompt = """
    Analyze this image in detail. It appears to be a product manual diagram, flowchart, or UI screenshot.
    
    Please extract the following information:
    1. **Title/Topic**: What is this diagram about?
    2. **Text Content**: Transcribe any visible text accurately.
    3. **Visual Logic**: If it's a flowchart or process, describe the steps in order. Who are the actors? What are the decision points? What is the flow direction?
    4. **Summary**: Provide a concise summary of the business rule or functionality depicted.
    
    Output the result as plain text description.
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            },
                        },
                    ],
                }
            ],
            max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error processing image {image_path.name}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Extract semantic info from images using VLM")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file to append to")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"), help="API Key")
    parser.add_argument("--base_url", type=str, default=os.getenv("BASE_URL", "https://api.openai.com/v1"), help="API Base URL (Must support Vision!)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Vision Model (e.g., gpt-4o, qwen-vl-max)")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Please provide --api_key or set OPENAI_API_KEY env var.")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    input_path = Path(args.input_dir).expanduser().resolve()
    
    # 1. Scan for images
    print(f"Scanning {input_path} for images...")
    all_images = []
    for root, _, files in os.walk(input_path):
        for file in files:
            if Path(file).suffix.lower() in IMG_EXTENSIONS:
                all_images.append(Path(root) / file)
    
    # 2. Check what's already processed (Resume logic)
    processed_filenames = get_processed_images(args.output_file)
    images_to_process = [img for img in all_images if img.name not in processed_filenames]
    
    print(f"Found {len(all_images)} images total.")
    print(f"Skipping {len(processed_filenames)} already processed.")
    print(f"Remaining {len(images_to_process)} images to process.")
    
    processed_count = 0
    
    with open(args.output_file, 'a', encoding='utf-8') as f:
        for img_path in tqdm(images_to_process, desc="Analyzing Images"):
            description = analyze_image(client, args.model, img_path)
            
            if description:
                entry = {
                    "file_path": str(img_path),
                    "filename": img_path.name,
                    "content": f"[IMAGE DESCRIPTION of {img_path.name}]\n{description}",
                    "product_area": img_path.parent.name,
                    "source_type": "image"
                }
                
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                processed_count += 1
            
            time.sleep(1) # Rate limit

    print(f"\nSuccessfully processed {processed_count} new images.")
    print(f"Results appended to {args.output_file}")

if __name__ == "__main__":
    main()