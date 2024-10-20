from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from PIL import Image
import cv2

def load_model():
    # Load your fine-tuned model and processor
    model = AutoModelForCausalLM.from_pretrained("florence_model_ckpt", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("florence_processor_ckpt", trust_remote_code=True)
    return model, processor

model, processor = load_model()
model.eval()

def run_example(task_prompt, text_input, image):
    prompt = task_prompt + text_input

    # Ensure the image is in RGB mode (PIL Image)
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image.width, image.height)
    )
    return parsed_answer

# def process_frame(frame):
#     global fin_result
#     # Convert OpenCV image (BGR) to PIL image (RGB)

frame = cv2.imread("test.jpg")
image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(image)

#     # Run the model inference
result = run_example(
    "DocVQA",
    "What is the MRP, Brand Name and Exp date of the product if visible in the picture?",
    pil_image
)
fin_result = result  # Update the global variable
print(result)
    # return frame  