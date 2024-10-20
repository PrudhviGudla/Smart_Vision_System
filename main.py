from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import threading
import time

# Initialize the FastAPI app
app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables
video_stream = None
lock = threading.Lock()

# Replace with your IP webcam URL
IP_WEBCAM_URL = 'http://<IP_ADDRESS>/video'

# Load your florence2 model
def load_model():
    # Placeholder for loading your model
    # Replace with actual model loading code
    model = None
    return model

model = load_model()

def process_frame(frame):
    # Placeholder for running your model on the frame
    # Replace with actual model inference code
    # Example: result = model.predict(frame)
    result = frame  # Dummy processing
    return result

def generate_frames():
    global video_stream, lock
    # Capture video from IP webcam
    video_stream = cv2.VideoCapture(IP_WEBCAM_URL)
    while True:
        success, frame = video_stream.read()
        if not success:
            break
        else:
            # Run your model on the frame
            processed_frame = process_frame(frame)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()

            # Use a lock to ensure thread safety
            with lock:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # Add a small sleep to prevent overutilization of CPU
        time.sleep(0.01)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Render the frontend page
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
def video_feed():
    # Return the response generated along with the specific media type (mime type)
    return StreamingResponse(generate_frames(),
                             media_type='multipart/x-mixed-replace; boundary=frame')
