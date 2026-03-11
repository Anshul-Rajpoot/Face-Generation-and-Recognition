import streamlit as st
from PIL import Image
import numpy as np
import os
from face_utils import get_embedding
from database import search_faces, enroll_face

# Create a folder to store enrolled images locally
IMAGE_DIR = "captured_images"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

st.set_page_config(page_title="Face Matcher AI", layout="centered")
st.title("👤 Face Recognition Search")

# --- Sidebar: Enrollment ---
with st.sidebar:
    st.header("Admin: Add to DB")
    new_name = st.text_input("Person Name")
    new_img = st.file_uploader("Upload Image to Save", type=['jpg', 'png', 'jpeg'], key="enroll")
    
    if st.button("Add to Database"):
        if new_name and new_img:
            with st.spinner(f"Enrolling {new_name}..."):
                # 1. Process Image
                img = Image.open(new_img)
                img_array = np.array(img)
                
                # 2. Get Embedding
                vec = get_embedding(img_array)
                
                if vec:
                    # 3. Save Image Locally
                    file_path = os.path.join(IMAGE_DIR, f"{new_name.replace(' ', '_')}.jpg")
                    img.save(file_path)
                    
                    # 4. Save to MongoDB
                    if enroll_face(new_name, vec, file_path):
                        st.success(f"Successfully added {new_name}!")
                    else:
                        st.error("Database save failed.")
                else:
                    st.error("No face detected in enrollment photo.")
        else:
            st.warning("Please provide name and image.")

# --- Main Search Interface ---
uploaded_file = st.file_uploader("Choose a face image to search...", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=250)
    
    if st.button("Search for Matches"):
        with st.spinner("Searching..."):
            query_vec = get_embedding(np.array(img))
            
            if query_vec:
                results = search_faces(query_vec, limit=3)
                
                if results:
                    st.success(f"Found {len(results)} matches!")
                    cols = st.columns(len(results))
                    for idx, match in enumerate(results):
                        with cols[idx]:
                            st.metric("Match Score", f"{round(match['score'] * 100, 1)}%")
                            # Display the saved image with fixed dimensions
                            if os.path.exists(match['image_path']):
                                match_img = Image.open(match['image_path'])
                                match_img = match_img.resize((300,300))
                                st.image(match_img, use_column_width=True)
                            st.subheader(match['name'])
                else:
                    st.warning("No matches found.")
            else:
                st.error("Face not detected in the uploaded photo.")