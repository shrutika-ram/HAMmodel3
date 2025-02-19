import openai
import os
import base64
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=API_KEY)

def get_image_description(image_path):
    """
    Sends an image to GPT-4V and gets the text prediction.
    """
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")


    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI that recognizes and describes images."},
            {"role": "user", "content": [
                {"type": "text", "text": "Identify the object drawn in the image. It shouldn't be more than 5 words."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
            ]}
        ],
        max_tokens=100  # Limits response length
    )

    return response.choices[0].message.content

# Example usage
image_path = "output_drawing.jpg"  # Replace with your image path
prediction = get_image_description(image_path)
# print("Prediction:", prediction)
with open("prediction.txt", "w") as f:
    f.write(prediction)
