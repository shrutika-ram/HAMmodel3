import os
import time
import subprocess
import streamlit as st
from PIL import Image

venv_python = os.path.abspath("venv/Scripts/python")

image_path = "output_drawing.jpg"
prediction_path = "prediction.txt"

if st.button("🖌️ Start Painting"):
    st.write("Camera is running... Move your hand to draw.")
    subprocess.run([venv_python, "airpainter.py"])

while not os.path.exists(image_path):
    time.sleep(1)

subprocess.run([venv_python, "image_recognition.py"])

while not os.path.exists(prediction_path):
    time.sleep(1)

if os.path.exists(image_path):
    st.image(Image.open(image_path), caption="🖼️ Your Drawing", use_column_width=True)

    with open(prediction_path, "r") as f:  
        prediction = f.read().strip()
    st.subheader("🔍 Prediction:")
    st.write(prediction)

try:
    os.remove(image_path)
    print("Deleted:", image_path)
except Exception as e:
    print(f"Error deleting file: {e}")

try:
    os.remove(prediction_path)
    print("Deleted:", prediction_path)
except Exception as e:
    print(f"Error deleting file: {e}")
