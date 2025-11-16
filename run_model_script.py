from deepseek_ocr.model import DeepSeekOCRModel


# prompt = "Free OCR. "
prompt = "<|grounding|>Convert the document to markdown. "
image_file = './images/invoice-example-1.png'
output_path = './output_dir'

model = DeepSeekOCRModel()
model.run_inference(
    prompt=prompt,
    image_path=image_file,
    model_type="gundam",
    output_dir_path=output_path
)
