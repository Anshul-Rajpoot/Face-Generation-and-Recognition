from deepface import DeepFace
import numpy as np

def get_embedding(image_input):
    """
    Generates a 128-d embedding using Facenet.
    image_input can be a path or a numpy array from Streamlit.
    """
    try:
        # Use Facenet for 128-d vectors (matches the MongoDB index)
        results = DeepFace.represent(
            img_path=image_input, 
            model_name="Facenet", 
            enforce_detection=True,
            detector_backend="opencv"
        )
        return results[0]["embedding"]
    except Exception as e:
        print(f"Error in face detection: {e}")
        return None