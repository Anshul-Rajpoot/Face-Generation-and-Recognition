import numpy as np
import streamlit as st
from insightface.app import FaceAnalysis

# ---------------------------------------------------
# Load model only once (important for Streamlit)
# ---------------------------------------------------

@st.cache_resource
def load_model():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0)   # CPU mode
    return app

app = load_model()


# ---------------------------------------------------
# Generate face embedding
# ---------------------------------------------------

def get_embedding(image_input):
    """
    Generate a face embedding using InsightFace.

    image_input: numpy array image
    returns: 512-d embedding list
    """

    try:
        faces = app.get(image_input)

        if len(faces) == 0:
            return None

        embedding = faces[0].embedding

        return embedding.tolist()

    except Exception as e:
        print(f"Face detection error: {e}")
        return None
