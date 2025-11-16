from deepseek_ocr.model import DeepSeekOCRModel

# prompt = "Free OCR. "
prompt = "<|grounding|>Convert the document to markdown. "
image_path = "./images/invoice-example-1.png"
model_type = "gundam"
output_path = "./output_dir"

model = DeepSeekOCRModel()
model.run_inference(
    prompt=prompt,
    image_path=image_path,
    model_type=model_type,
    output_dir_path=output_path,
)
