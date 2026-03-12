import numpy as np
from insightface.app import FaceAnalysis

# Load model once (important for Streamlit performance)
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)   # ctx_id=0 for CPU


def get_embedding(image_input):
    """
    Generates a face embedding using InsightFace.
    image_input should be a numpy array (from Streamlit/PIL).
    """

    try:
        # Detect faces
        faces = app.get(image_input)

        if len(faces) == 0:
            return None

        # Get embedding of first detected face
        embedding = faces[0].embedding

        # Convert to list for MongoDB compatibility
        return embedding.tolist()

    except Exception as e:
        print(f"Face detection error: {e}")
        return None
