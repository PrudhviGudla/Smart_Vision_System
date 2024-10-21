from fastapi import FastAPI, Request
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
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the FastAPI app
app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables
video_stream = None
video_stream_lock = threading.Lock()
fin_result = ""
fin_result_lock = threading.Lock()

# Replace with your IP webcam URL or use 0 for the default webcam
IP_WEBCAM_URL = 0  # Use 0 for default webcam

def load_model():
    try:
        logger.info("Loading model and processor...")
        model = AutoModelForCausalLM.from_pretrained("florence_model_ckpt", trust_remote_code=True)
        processor = AutoProcessor.from_pretrained("florence_processor_ckpt", trust_remote_code=True)
        model.eval()
        logger.info("Model and processor loaded successfully.")
        return model, processor
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

# Load the model at startup
model, processor = load_model()

def run_example(task_prompt, text_input, image):
    try:
        logger.info("Running model inference...")
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
        logger.info("Model inference completed.")
        return parsed_answer
    except Exception as e:
        logger.error(f"Error during model inference: {e}")
        return "Error during model inference."

def capture_frame():
    global video_stream
    try:
        with video_stream_lock:
            if video_stream is None or not video_stream.isOpened():
                logger.info("Opening video stream...")
                # Initialize the video stream
                video_stream = cv2.VideoCapture(IP_WEBCAM_URL)
            else:
                video_stream.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset frame position if necessary

            success, frame = video_stream.read()
            if not success:
                logger.error("Failed to read frame from video stream.")
                return None
            else:
                logger.info("Frame captured successfully.")
                return frame
    except Exception as e:
        logger.error(f"Error capturing frame: {e}")
        return None

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

def generate_frames():
    global video_stream
    try:
        with video_stream_lock:
            if video_stream is None or not video_stream.isOpened():
                logger.info("Initializing video stream for video feed...")
                video_stream = cv2.VideoCapture(IP_WEBCAM_URL)
    except Exception as e:
        logger.error(f"Error initializing video stream: {e}")
        return

    while True:
        with video_stream_lock:
            success, frame = video_stream.read()
        if not success:
            logger.error("Failed to read frame from video stream.")
            break
        else:
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Yield the frame in the appropriate format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)  # Adjust sleep time as needed

@app.post("/process_frame")
def process_frame_endpoint():
    global fin_result
    try:
        # Capture a frame from the video stream
        frame = capture_frame()
        if frame is None:
            logger.error("No frame available for processing.")
            return JSONResponse(content={"result": "No frame available."})

        # Convert OpenCV image (BGR) to PIL image (RGB)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)

        # Run the model inference
        result = run_example(
            "DocVQA",
            "What is the MRP, Brand Name and Exp date of the product if visible in the picture?",
            pil_image
        )
        with fin_result_lock:
            fin_result = result  # Update the global variable
        logger.info(f"Model Output: {result}, {result['DocVQA']}")

        return JSONResponse(content={"result": result['DocVQA']})
    except Exception as e:
        logger.error(f"Error in process_frame_endpoint: {e}")
        return JSONResponse(content={"result": f"Error: {e}"})

@app.get("/get_result")
def get_result():
    # Return the latest result as JSON
    global fin_result
    with fin_result_lock:
        result = fin_result
    return JSONResponse(content={"result": result['DocVQA']})
