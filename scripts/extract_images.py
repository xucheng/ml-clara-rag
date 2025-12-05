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

def validate_and_convert_image(image_path: Path, min_dimension: int = 10) -> str:
    """
    Validates image using Pillow and converts to standardized JPEG base64.

    Args:
        image_path: Path to image file
        min_dimension: Minimum width/height in pixels (default: 10)

    Returns:
        base64 string if valid, None otherwise.
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size

            # Check minimum dimensions (API requirement: both dimensions must be > 10px)
            if width <= min_dimension or height <= min_dimension:
                print(f"⚠️  Skipping too small image {image_path.name}: {width}x{height} (min: {min_dimension}px)")
                return None

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
        print(f"⚠️  Skipping corrupt/unsupported image {image_path.name}: {e}")
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

def analyze_image(client: OpenAI, model: str, image_path: Path, max_retries: int = 3):
    """
    Sends image to VLM API for semantic extraction with exponential backoff retry.

    Args:
        client: OpenAI client
        model: Vision model name
        image_path: Path to image file
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        str: Image description or None if all retries failed
    """

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

    # Retry with exponential backoff
    for attempt in range(max_retries):
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
                max_tokens=2048,
                timeout=60.0  # 60 second timeout per request
            )
            return response.choices[0].message.content

        except Exception as e:
            error_msg = str(e)
            is_last_attempt = (attempt == max_retries - 1)

            # Check if it's a retryable error
            retryable_errors = ['Connection error', 'timeout', 'Timeout', 'rate limit', 'Rate limit', '429', '503', '500', '502', '504']
            is_retryable = any(err in error_msg for err in retryable_errors)

            if is_retryable and not is_last_attempt:
                # Exponential backoff: 2^attempt seconds (2s, 4s, 8s, ...)
                wait_time = 2 ** attempt
                print(f"⚠️  Retry {attempt + 1}/{max_retries} for {image_path.name}: {error_msg}")
                print(f"   Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                # Non-retryable error or last attempt
                if is_last_attempt:
                    print(f"❌ Failed after {max_retries} attempts: {image_path.name} - {error_msg}")
                else:
                    print(f"❌ Non-retryable error for {image_path.name}: {error_msg}")
                return None

    return None

def main():
    parser = argparse.ArgumentParser(description="Extract semantic info from images using VLM")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file to append to")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"), help="API Key")
    parser.add_argument("--base_url", type=str, default=os.getenv("BASE_URL", "https://api.openai.com/v1"), help="API Base URL (Must support Vision!)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Vision Model (e.g., gpt-4o, qwen-vl-max)")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum retry attempts for failed requests (default: 3)")
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
    failed_count = 0
    skipped_count = 0  # For images that are invalid/too small

    with open(args.output_file, 'a', encoding='utf-8') as f:
        for img_path in tqdm(images_to_process, desc="Analyzing Images"):
            description = analyze_image(client, args.model, img_path, max_retries=args.max_retries)

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
            else:
                # Check if it was skipped due to validation or failed after retries
                # (We can't distinguish perfectly, but assume None from validation is skip)
                failed_count += 1

            time.sleep(1) # Rate limit

    print(f"\n{'='*60}")
    print(f"📊 Image Processing Summary:")
    print(f"   ✅ Successfully processed: {processed_count} images")
    if failed_count > 0:
        print(f"   ⚠️  Skipped/Failed:        {failed_count} images")
        print(f"      (Too small, corrupt, or connection errors)")
        print(f"   ℹ️  Failed images will be retried on next run")
    print(f"   📝 Results appended to: {args.output_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()