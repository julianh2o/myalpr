import cv2
import base64
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL")


def read_plate(frame):
    """
    Read license plate text from an image frame using Ollama vision model.

    Args:
        frame: OpenCV image frame containing a license plate

    Returns:
        str: Cleaned license plate number (alphanumeric only), or None if failed
    """
    if not OLLAMA_URL or not OLLAMA_VISION_MODEL:
        print("Error: OLLAMA_URL or OLLAMA_VISION_MODEL not configured in .env")
        return None

    # Encode frame as JPEG
    retval, buffer = cv2.imencode('.jpg', frame)
    b64_bytes = base64.b64encode(buffer)
    image_base64 = b64_bytes.decode('utf-8')

    # Prepare the request to Ollama
    payload = {
        "model": OLLAMA_VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": "This is a license plate image. Please read the license plate number/text. Return ONLY the alphanumeric characters you see on the plate, with no spaces, punctuation, or explanation. Just the plate number.",
                "images": [image_base64]
            }
        ],
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        plate_number = result['message']['content'].strip()

        # Clean up the response - remove any extra text
        # Extract only alphanumeric characters
        plate_number = ''.join(c for c in plate_number if c.isalnum()).upper()

        return plate_number

    except Exception as e:
        print(f"Error reading plate: {e}")
        return None
