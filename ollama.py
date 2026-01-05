"""
BAML-based license plate reader.

To use this module:
1. Install BAML: pip install baml-py
2. Generate the Python client: baml-cli generate (run from project root)
3. Use read_plate() function

The generated BAML client code will be in the baml_client/ directory.
"""

from typing import Optional, List
from image_utils import frame_to_baml_image

# Import the generated BAML function
from baml_client import b
from baml_client.async_client import b as async_b


def read_plate(frame, known_plates: Optional[List[str]] = None) -> Optional[str]:
    """
    Read license plate text from an image frame using Ollama vision model via BAML.

    Args:
        frame: OpenCV image frame containing a license plate
        known_plates: Optional list of known plate numbers to guide recognition

    Returns:
        str: Cleaned license plate number (alphanumeric only), or None if failed
    """
    try:
        # Convert frame to BAML Image object
        img = frame_to_baml_image(frame)
        if img is None:
            return None

        # Call the BAML-generated function
        plate_number = b.ReadPlate(
            frame=img,
            known_plates=known_plates
        )

        # Clean up the response - extract only alphanumeric characters
        plate_number = ''.join(c for c in plate_number if c.isalnum()).upper()

        return plate_number if plate_number else None

    except Exception as e:
        print(f"Error reading plate: {e}")
        return None


async def read_plate_async(frame, known_plates: Optional[List[str]] = None) -> Optional[str]:
    """
    Async version of read_plate for non-blocking plate reading.

    Args:
        frame: OpenCV image frame containing a license plate
        known_plates: Optional list of known plate numbers to guide recognition

    Returns:
        str: Cleaned license plate number (alphanumeric only), or None if failed
    """
    try:
        # Convert frame to BAML Image object
        img = frame_to_baml_image(frame)
        if img is None:
            return None

        # Call the async BAML-generated function
        plate_number = await async_b.ReadPlate(
            frame=img,
            known_plates=known_plates
        )

        # Clean up the response - extract only alphanumeric characters
        plate_number = ''.join(c for c in plate_number if c.isalnum()).upper()

        return plate_number if plate_number else None

    except Exception as e:
        print(f"Error reading plate: {e}")
        return None
