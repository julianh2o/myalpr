"""
Image utility functions for encoding and processing frames.
"""

import cv2
import base64
from typing import Tuple, Optional
from baml_py import Image


def frame_to_base64(frame, format: str = '.jpg') -> Optional[Tuple[str, str]]:
    """
    Encode an OpenCV frame to base64.

    Args:
        frame: OpenCV image frame (numpy array)
        format: Image format ('.jpg' or '.png')

    Returns:
        Tuple of (media_type, base64_string) or None if encoding fails
        Example: ("image/jpeg", "iVBORw0KGgo...")
    """
    try:
        # Encode frame to the specified format
        retval, buffer = cv2.imencode(format, frame)
        if not retval:
            print(f"Error: Failed to encode frame as {format}")
            return None

        # Convert to base64
        b64_bytes = base64.b64encode(buffer)
        image_base64 = b64_bytes.decode('utf-8')

        # Determine media type
        media_type = "image/jpeg" if format == '.jpg' else "image/png"

        return media_type, image_base64

    except Exception as e:
        print(f"Error encoding frame to base64: {e}")
        return None


def frame_to_baml_image(frame, format: str = '.jpg') -> Optional[Image]:
    """
    Convert an OpenCV frame to a BAML Image object.

    Args:
        frame: OpenCV image frame (numpy array)
        format: Image format ('.jpg' or '.png')

    Returns:
        BAML Image object or None if encoding fails
    """
    result = frame_to_base64(frame, format)
    if result is None:
        return None

    media_type, image_base64 = result
    return Image.from_base64(media_type, image_base64)
