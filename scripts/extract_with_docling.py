import os
import json
import argparse
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions

def main():
    parser = argparse.ArgumentParser(description="Extract text and images using Docling")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--image_output_dir", type=str, required=True)
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
    # Docling supports many formats. Let's focus on PDF and DOCX for now.
    supported_exts = {'.pdf', '.docx', '.pptx'} 
    
    for root, _, files in os.walk(input_path):
        for file in files:
            if Path(file).suffix.lower() in supported_exts:
                files_to_process.append(Path(root) / file)
    
    print(f"Found {len(files_to_process)} documents. Starting Docling conversion...")
    print(f"Note: First run might be slow as it downloads models.")

    for file_path in files_to_process:
        print(f"Processing: {file_path.name}")
        try:
            conv_res = converter.convert(file_path)
            doc = conv_res.document
            
            extracted_img_paths = []
            
            # Save images
            # Iterate through document elements to find pictures
            # Note: doc.pictures is usually a list of PictureItem
            counter = 0
            # The logic to iterate pictures depends on Docling version structure.
            # Modern Docling attaches pictures to the document model.
            if hasattr(doc, 'pictures'):
                for picture in doc.pictures:
                    img = picture.get_image(doc)
                    if img:
                        filename = f"{file_path.stem}_img_{counter}.png"
                        save_path = img_out_path / filename
                        img.save(save_path)
                        extracted_img_paths.append(str(save_path))
                        counter += 1
            
            # Export content to Markdown
            # This provides a very clean, structured representation of the document
            md_content = doc.export_to_markdown()
            
            # Append Image Placeholders if they aren't in the markdown
            # (Docling's markdown might use internal refs, let's ensure we have paths)
            # A simple strategy: Just append the list of extracted images at the end of text
            # to ensure the VLM processing step picks them up.
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
            
    print(f"Done. Processed {len(results)} files.")

if __name__ == "__main__":
    main()
