import os
import json
from openai import OpenAI
from pathlib import Path
import base64
import argparse
from PIL import Image
import io
from typing import Optional

def encode_image(image_path):
    """Encodes an image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_dummy_image_base64(size=(100, 100)):
    """Dynamically creates a transparent PNG image of specified size and returns its base64 string."""
    img = Image.new('RGBA', size, (255, 0, 0, 255)) # Red square, 100x100
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def find_first_image(directory: Path) -> Optional[Path]:
    """Finds the first .jpg or .png image in a directory recursively."""
    if not directory.is_dir():
        return None
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                return file_path
    return None

def test_api_key(args):
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BASE_URL")
    vision_model = os.getenv("VISION_MODEL")

    print(f"--- API Key Availability Test ---")
    print(f"OPENAI_API_KEY: {'*****' + api_key[-4:] if api_key else 'Not set'}")
    print(f"BASE_URL: {base_url if base_url else 'Not set (default used)'}")
    print(f"VISION_MODEL: {vision_model if vision_model else 'Not set (Text-only test)'}")
    print("-" * 30)

    if not api_key:
        print("❌ Error: OPENAI_API_KEY 环境变量未设置。请先设置您的 API Key。")
        print("例如: export OPENAI_API_KEY=\"sk-...\"")
        return

    try:
        if base_url:
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = OpenAI(api_key=api_key) 
        
        # --- 测试文本功能 ---
        print("🟢 正在测试文本功能...")
        messages = [
            {"role": "user", "content": "Hello, just verify you are working."}
        ]
        text_model_name = "gpt-3.5-turbo" 
        if base_url and "deepseek" in base_url.lower():
            text_model_name = "deepseek-chat"
        elif base_url and "aliyuncs" in base_url.lower():
            text_model_name = "qwen-turbo" 

        response = client.chat.completions.create(
            model=text_model_name,
            messages=messages,
            max_tokens=20
        )
        print("✅ 文本功能测试成功！")
        print(f"   API 回应: {response.choices[0].message.content.strip()}")

        # --- 测试视觉功能 ---
        if vision_model:
            print("\n🟢 正在测试视觉功能...")
            
            # 尝试寻找真实图片
            test_image_path = None
            if args.input_dir:
                search_dir = Path(args.input_dir).expanduser()
                if search_dir.exists():
                    test_image_path = find_first_image(search_dir)
            
            base64_image = ""
            if test_image_path:
                print(f"   📷 使用本地真实图片进行测试: {test_image_path.name}")
                base64_image = encode_image(test_image_path)
            else:
                print(f"   ⚠️ 未找到本地图片 (搜索路径: {args.input_dir})")
                print(f"   🎨 生成 100x100 虚拟图片进行测试...")
                base64_image = create_dummy_image_base64()
            
            vision_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in one sentence."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ]
            response = client.chat.completions.create(
                model=vision_model,
                messages=vision_messages,
                max_tokens=50
            )
            print("✅ 视觉功能测试成功！")
            print(f"   API 回应: {response.choices[0].message.content.strip()}")

    except Exception as e:
        print(f"\n❌ API Key 或配置似乎无效！")
        print(f"   错误详情: {e}")

    print("\n--- 测试结束 ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="~/Downloads/Product Manual",
                        help="Directory to search for a real image.")
    args = parser.parse_args()
    test_api_key(args)