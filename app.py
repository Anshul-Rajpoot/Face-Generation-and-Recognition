import streamlit as st
from PIL import Image
import numpy as np
import cloudinary
import cloudinary.uploader
import os
from dotenv import load_dotenv

from face_utils import get_embedding
from database import search_faces, enroll_face

load_dotenv()

# ---------------- Cloudinary ----------------

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# ---------------- UI ----------------

st.set_page_config(page_title="Face Matcher AI", layout="centered")
st.title("👤 Face Recognition Search")


# ======================================================
# SIDEBAR : ENROLL FACE
# ======================================================

with st.sidebar:

    st.header("Admin: Add to DB")

    new_name = st.text_input("Person Name")

    new_img = st.file_uploader(
        "Upload Image to Save",
        type=['jpg','png','jpeg'],
        key="enroll"
    )

    if st.button("Add to Database"):

        if new_name and new_img:

            with st.spinner(f"Enrolling {new_name}..."):

                img = Image.open(new_img)
                img_array = np.array(img)

                vec = get_embedding(img_array)

                if vec is not None:

                    new_img.seek(0)

                    upload_result = cloudinary.uploader.upload(new_img)
                    image_url = upload_result["secure_url"]

                    if enroll_face(new_name, vec, image_url):

                        st.success(f"{new_name} added successfully!")
                        st.image(image_url, width=200)

                    else:
                        st.error("Database save failed")

                else:
                    st.error("No face detected")

        else:
            st.warning("Provide name and image")


# ======================================================
# MODE SELECTOR
# ======================================================

mode = st.radio(
    "Select Mode",
    ["Upload Image Search"]
)


# ======================================================
# IMAGE SEARCH
# ======================================================

if mode == "Upload Image Search":

    uploaded_file = st.file_uploader(
        "Choose a face image to search...",
        type=['jpg','png','jpeg']
    )

    # ---------------- Threshold Slider ----------------

    threshold = st.slider(
        "Match Threshold",
        min_value=0.4,
        max_value=0.9,
        value=0.65,
        step=0.05
    )

    if uploaded_file:

        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", width=200)

        if st.button("Search for Matches"):

            with st.spinner("Searching..."):

                query_vec = get_embedding(np.array(img))

                if query_vec is not None:

                    results = search_faces(query_vec, limit=5)

                    # ---------------- Apply Threshold ----------------

                    filtered_results = [
                        r for r in results if r.get("score", 0) >= threshold
                    ]

                    if filtered_results:

                        st.success(f"Found {len(filtered_results)} matches!")

                        cols = st.columns(len(filtered_results))

                        for idx, match in enumerate(filtered_results):

                            with cols[idx]:

                                score = match.get("score", 0)
                                name = match.get("name", "Unknown")

                                st.metric(
                                    "Match Score",
                                    f"{round(score * 100,1)}%"
                                )

                                image_url = match.get("image_url") or match.get("image_path")

                                if image_url:
                                    st.image(image_url, width=200)

                                st.subheader(name)

                    else:
                        st.warning("No matches above threshold")

                else:
                    st.error("Face not detected")
