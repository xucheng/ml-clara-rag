import unittest
from unittest.mock import patch, MagicMock, mock_open, ANY
import os
import json
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

try:
    import extract_raw_data
    import synthesize_data
    import extract_images
except ImportError as e:
    pass

class TestDataPipeline(unittest.TestCase):

    def setUp(self):
        self.env_patcher = patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test-key",
            "BASE_URL": "https://test.api.com/v1",
            "VISION_MODEL": "test-vision-model"
        })
        self.env_patcher.start()
        self.img_out_dir = Path("./test_assets")

    def tearDown(self):
        self.env_patcher.stop()

    # --- Test extract_raw_data.py (Enhanced) ---

    @patch('extract_raw_data.docx.Document')
    @patch('extract_raw_data.save_image', return_value="/path/to/img.jpg")
    def test_extract_docx_with_images(self, mock_save_image, mock_document):
        # Mock DOCX Structure
        mock_doc = MagicMock()
        
        # Mock Text
        mock_para = MagicMock()
        mock_para.text = "Text Content"
        mock_doc.paragraphs = [mock_para]
        mock_doc.tables = []
        
        # Mock Image Relationship
        mock_rel = MagicMock()
        mock_rel.target_ref = "media/image1.png"
        mock_rel.target_part.blob = b'fake_image_bytes'
        mock_rel.target_part.content_type = "image/png"
        
        # doc.part.rels is a dict
        mock_doc.part.rels = {"rId1": mock_rel}
        
        mock_document.return_value = mock_doc
        
        text, images = extract_raw_data.extract_from_docx(Path("test.docx"), self.img_out_dir)
        
        # Verification
        self.assertIn("Text Content", text)
        self.assertIn("[IMAGE_REF: /path/to/img.jpg]", text) # Check placeholder insertion
        self.assertIn("/path/to/img.jpg", images)
        mock_save_image.assert_called_once()

    @patch('extract_raw_data.PdfReader')
    @patch('extract_raw_data.save_image', return_value="/path/to/pdf_img.jpg")
    def test_extract_pdf_with_images(self, mock_save_image, mock_pdf_reader):
        # Mock PDF Structure
        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PDF Text"
        
        # Mock XObject (Images)
        mock_xobject_obj = MagicMock() # Represents the XObject dictionary itself
        mock_image_data_obj = MagicMock() # Represents a single image object within XObject
        mock_image_data_obj.get_data.return_value = b'fake_image_bytes_large_enough' * 100 # ensure > 1024 bytes
        mock_image_data_obj.__getitem__.side_effect = lambda key: {"/Subtype": "/Image", "/Filter": "/DCTDecode"}[key] # Mimic dictionary access
        
        mock_xobject_obj.__getitem__.side_effect = lambda key: mock_image_data_obj if key in ["/Img1"] else None
        mock_xobject_obj.__iter__.return_value = iter(["/Img1"]) # Make it iterable
        
        # Mock Resources dict
        resources = {
            "/XObject": MagicMock()
        }
        resources["/XObject"].get_object.return_value = mock_xobject_obj
        
        mock_page.__getitem__.side_effect = lambda key: resources if key == "/Resources" else None
        
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader
        
        text, images = extract_raw_data.extract_from_pdf(Path("test.pdf"), self.img_out_dir)
        
        self.assertIn("PDF Text", text)
        self.assertIn("[IMAGE_REF: /path/to/pdf_img.jpg]", text)
        self.assertIn("/path/to/pdf_img.jpg", images)
        mock_save_image.assert_called_once()

    @patch('extract_raw_data.Presentation')
    @patch('extract_raw_data.save_image', return_value="/path/to/ppt_img.jpg")
    def test_extract_pptx_with_images(self, mock_save_image, mock_presentation):
        # Mock PPTX
        mock_prs = MagicMock()
        mock_slide = MagicMock()
        
        # Shape 1: Text
        shape_text = MagicMock()
        shape_text.text = "Slide Text"
        shape_text.shape_type = 1 # Not Picture
        
        # Shape 2: Picture
        shape_pic = MagicMock()
        shape_pic.shape_type = extract_raw_data.MSO_SHAPE_TYPE.PICTURE
        shape_pic.image.blob = b'fake_bytes'
        shape_pic.image.ext = "jpg"
        # Ensure it doesn't have 'text' attr to avoid text extraction logic confusion
        del shape_pic.text 
        
        mock_slide.shapes = [shape_text, shape_pic]
        mock_prs.slides = [mock_slide]
        mock_presentation.return_value = mock_prs
        
        text, images = extract_raw_data.extract_from_pptx(Path("test.pptx"), self.img_out_dir)
        
        self.assertIn("Slide Text", text)
        self.assertIn("[IMAGE_REF: /path/to/ppt_img.jpg]", text)
        self.assertIn("/path/to/ppt_img.jpg", images)
        mock_save_image.assert_called_once()


    # --- Test synthesize_data.py (Existing) ---

    @patch('synthesize_data.OpenAI')
    def test_synthesize_data_args(self, mock_openai):
        test_args = [
            'synthesize_data.py', 
            '--input_file', 'dummy.jsonl', 
            '--output_dir', 'out_dir',  # Updated argument
            '--model', 'qwen-plus'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('builtins.open', mock_open(read_data='{"filename": "test", "content": "This is a test content that is definitely longer than fifty characters for the mock API call."}')):
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.choices[0].message.content = '{"dense_summary": "sum", "qa_pairs": []}'
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client
                
                # Mock Path.mkdir to avoid filesystem errors
                with patch('pathlib.Path.mkdir'):
                    synthesize_data.main()
                
                mock_openai.assert_called_with(
                    api_key="sk-test-key", 
                    base_url="https://test.api.com/v1" 
                )
                call_args = mock_client.chat.completions.create.call_args
                self.assertEqual(call_args.kwargs['model'], 'qwen-plus')

    # --- Test extract_images.py (Existing) ---

    @patch('extract_images.OpenAI')
    @patch('extract_images.validate_and_convert_image', return_value="base64_string") # Updated function name
    @patch('extract_images.get_processed_images', return_value=set()) # Mock resume logic
    def test_extract_images_args(self, mock_get_processed, mock_validate, mock_openai):
        test_args = [
            'extract_images.py',
            '--input_dir', './images',
            '--output_file', 'images.jsonl'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('os.walk') as mock_walk:
                mock_walk.return_value = [('./images', [], ['test.jpg'])]
                
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.choices[0].message.content = "Image Description"
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client
                
                with patch('builtins.open', mock_open()) as mock_file:
                    extract_images.main()
                
                mock_openai.assert_called_with(
                    api_key="sk-test-key",
                    base_url="https://test.api.com/v1"
                )

if __name__ == '__main__':
    unittest.main()