import os
import json
import argparse
import base64
from pathlib import Path
from typing import List, Dict, Tuple
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions


def extract_image_from_uri(uri: str) -> bytes:
    """从 data URI 中提取图片二进制数据"""
    if uri.startswith('data:'):
        # Format: data:image/png;base64,<data>
        header, encoded = uri.split(',', 1)
        return base64.b64decode(encoded)
    return None


def reconstruct_document_with_image_refs(doc_dict: Dict, image_output_dir: Path, filename_prefix: str) -> Tuple[str, List[str]]:
    """
    根据文档字典重建文档内容,在正确位置插入图片引用 (dict模式)

    Returns:
        (markdown_content, list_of_image_paths)
    """

    # 保存所有图片并记录路径
    image_paths = {}  # self_ref -> file_path
    img_output_path = Path(image_output_dir)
    img_output_path.mkdir(parents=True, exist_ok=True)

    for idx, pic_dict in enumerate(doc_dict.get('pictures', [])):
        self_ref = pic_dict.get('self_ref', f'#/pictures/{idx}')

        # 提取图片数据
        if 'image' in pic_dict and 'uri' in pic_dict['image']:
            img_data = extract_image_from_uri(pic_dict['image']['uri'])
            if img_data:
                # 保存图片
                img_filename = f"{filename_prefix}_img_{idx}.png"
                img_path = img_output_path / img_filename
                with open(img_path, 'wb') as f:
                    f.write(img_data)
                image_paths[self_ref] = str(img_path)

    # 递归重建文档内容
    def render_element(ref: str, doc_dict: Dict) -> str:
        """递归渲染文档元素"""

        if not ref.startswith('#/'):
            return ""

        # 解析引用路径 (e.g., "#/texts/0" -> ["texts", "0"])
        path_parts = ref[2:].split('/')

        # 获取元素
        element = doc_dict
        for part in path_parts:
            if part.isdigit():
                element = element[int(part)]
            else:
                element = element.get(part)
                if element is None:
                    return ""

        # 处理不同类型的元素
        if 'text' in element:
            # 文本元素
            return element['text'] + "\n\n"

        elif path_parts[0] == 'pictures':
            # 图片元素 - 插入图片引用
            self_ref = element.get('self_ref')
            if self_ref in image_paths:
                return f"[IMAGE_REF: {image_paths[self_ref]}]\n\n"
            else:
                return ""  # 图片提取失败，跳过

        elif path_parts[0] == 'tables':
            # 表格元素
            if 'text' in element:
                return element['text'] + "\n\n"
            return ""

        elif 'children' in element:
            # 容器元素 - 递归处理子元素
            content = ""
            for child_ref_obj in element['children']:
                if isinstance(child_ref_obj, dict) and '$ref' in child_ref_obj:
                    content += render_element(child_ref_obj['$ref'], doc_dict)
            return content

        return ""

    # 从 body 开始渲染
    markdown_content = render_element('#/body', doc_dict)

    return markdown_content.strip(), list(image_paths.values())


def main():
    parser = argparse.ArgumentParser(
        description="Extract text and images using Docling (supports dict and markdown modes)"
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing documents")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--image_output_dir", type=str, required=True, help="Directory to save extracted images")
    parser.add_argument(
        "--extraction_mode",
        type=str,
        default="dict",
        choices=["dict", "markdown"],
        help="Extraction mode: 'dict' (precise image position, default) or 'markdown' (legacy compatibility)"
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir).expanduser()
    output_file = Path(args.output_file)
    img_out_path = Path(args.image_output_dir)
    img_out_path.mkdir(parents=True, exist_ok=True)

    # Configure Pipeline Options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = True

    # Configure Converter
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    results = []
    files_to_process = []
    supported_exts = {'.pdf', '.docx', '.pptx'}

    for root, _, files in os.walk(input_path):
        for file in files:
            if Path(file).suffix.lower() in supported_exts:
                files_to_process.append(Path(root) / file)

    print(f"Found {len(files_to_process)} documents. Starting Docling conversion...")
    print(f"Extraction mode: {args.extraction_mode}")
    if args.extraction_mode == "dict":
        print("  → Using dict mode: images will be inserted at precise positions")
    else:
        print("  → Using markdown mode: images will be appended at the end (legacy)")
    print(f"Note: First run might be slow as it downloads models.")

    for file_path in files_to_process:
        print(f"Processing: {file_path.name}")
        try:
            conv_res = converter.convert(file_path)
            doc = conv_res.document

            if args.extraction_mode == "dict":
                # Dict-based extraction: precise image positions
                doc_dict = doc.export_to_dict()
                md_content, extracted_img_paths = reconstruct_document_with_image_refs(
                    doc_dict,
                    img_out_path,
                    file_path.stem
                )

            else:
                # Markdown-based extraction: legacy compatibility
                extracted_img_paths = []

                # Save images using legacy method
                counter = 0
                if hasattr(doc, 'pictures'):
                    for picture in doc.pictures:
                        img = picture.get_image(doc)
                        if img:
                            filename = f"{file_path.stem}_img_{counter}.png"
                            save_path = img_out_path / filename
                            img.save(save_path)
                            extracted_img_paths.append(str(save_path))
                            counter += 1

                # Export to markdown with image placeholders disabled
                md_content = doc.export_to_markdown(image_placeholder='')

                # Append image references at the end
                if extracted_img_paths:
                    md_content += "\n\n--- Extracted Images ---\n"
                    for p in extracted_img_paths:
                        md_content += f"[IMAGE_REF: {p}]\n"

            results.append({
                "file_path": str(file_path),
                "filename": file_path.name,
                "content": md_content,
                "product_area": file_path.parent.name,
                "extracted_images": extracted_img_paths
            })

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n✅ Done! Processed {len(results)} files.")
    if args.extraction_mode == "dict":
        print("📍 Images inserted at precise positions in document structure")
    else:
        print("📎 Images appended at end of documents (markdown mode)")

if __name__ == "__main__":
    main()
