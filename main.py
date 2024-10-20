from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import threading
import time
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from PIL import Image
import numpy as np

# Initialize the FastAPI app
app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables
video_stream = None
lock = threading.Lock()
fin_result = ""

# Replace with your IP webcam URL or use 0 for the default webcam
# IP_WEBCAM_URL = "https://192.168.172.134:8080/video"
IP_WEBCAM_URL = 0

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

def process_frame(frame):
    global fin_result
    # Convert OpenCV image (BGR) to PIL image (RGB)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)

    # Run the model inference
    result = run_example(
        "DocVQA",
        "What is the MRP, Brand Name and Exp date of the product if visible in the picture?",
        pil_image
    )
    fin_result = result  # Update the global variable
    print(result)
    return frame  # Return the original frame for video streaming

def generate_frames():
    global video_stream, lock
    video_stream = cv2.VideoCapture(IP_WEBCAM_URL)
    c = 0
    while True:
        success, frame = video_stream.read()
        if not success:
            break
        else:
            # Process every 30th frame to reduce load
            if c % 10 == 0:
                threading.Thread(target=process_frame, args=(frame.copy(),)).start()
            c += 1
            if c > 6000:
                c = 0

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Use a lock to ensure thread safety
            with lock:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        # Small sleep to prevent overutilization of CPU
        time.sleep(0.01)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Render the frontend page
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
def video_feed():
    # Return the video feed
    return StreamingResponse(
        generate_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@app.get("/get_result")
def get_result():
    # Return the latest result as JSON
    global fin_result
    return JSONResponse(content={"result": fin_result})
