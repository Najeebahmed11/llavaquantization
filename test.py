import tensorflow as tf
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import base64
import requests
from io import BytesIO
from PIL import Image
import torch
from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok

# Presumably these imports are part of the llava-torch package
from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import process_images
from llava.model.builder import load_pretrained_model

model = None
class_names = ["real", "screen"]
model_path = "/path/to/your/local/model.h5"  # Replace with the path to your model

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(model_path)

def predict_image_class(image_data):
    load_model()  # Ensure the model is loaded
    
    # Decode the image from base64 and prepare it for prediction
    image_bytes = BytesIO(base64.b64decode(image_data))
    image = np.array(Image.open(image_bytes).convert("RGB").resize((256, 256)))

    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)

    print("Predictions:", predictions)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    return predicted_class, confidence
# In your describe_image_function before processing the image with llava model:

def describe_image_function(image_data, prompt):
    # Classify the image as real or screen
    predicted_class, confidence = predict_image_class(image_data)
    if predicted_class == "screen":
        return f"This is a screen image with a confidence of {confidence}%."

    # Continue with your existing code if the image is classified as "real"
    ...
    # Determine if image_data is a URL or Base64
    if image_data.startswith('http'):
        image = load_image_from_url(image_data)
    else: # Assume Base64
        image = load_image_from_base64(image_data)
    
    if image is None:
        return "Unable to process the image."

    processed_image = process_image(image)
    
    conv = conv_templates[CONV_MODE].copy()
    roles = conv.roles
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv.append_message(roles[0], prompt)
    conv.append_message(roles[1], None)

    # Convert conversation to model inputs
    tokenized_prompt = tokenizer.encode(conv.get_prompt(), return_tensors='pt')

    # Using the model to generate the description
    with torch.no_grad():
        outputs = model.generate(input_ids=tokenized_prompt)
    full_description = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extracting only the assistant's response
    description = full_description.split('###Assistant:')[-1].strip()

    return description
# Function to extract model name from path (assuming this is part of your llava-torch package)
def get_model_name_from_path(model_path):
    return model_path.split('/')[-1]

# Load the pretrained model
MODEL = "4bit/llava-v1.5-13b-3GB"
model_name = get_model_name_from_path(MODEL)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=MODEL, model_base=None, model_name=model_name, load_4bit=True
)

# Define the mode
CONV_MODE = "llava_v0"

# Function to load an image from a URL
def load_image_from_url(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        try:
            image = Image.open(BytesIO(response.content)).convert("RGB")
            return image
        except Exception as e:
            print(f"Error while opening the image: {e}")
            return None
    else:
        print(f"Error loading image, status code: {response.status_code}")
        return None

# Function to load an image from a Base64 string
def load_image_from_base64(image_b64):
    try:
        image_data = BytesIO(base64.b64decode(image_b64))
        image = Image.open(image_data).convert("RGB")
        return image
    except Exception as e:
        print(f"Error while decoding the base64 image: {e}")
        return None

# Function to process the image
def process_image(image):
    args = {"image_aspect_ratio": "pad"}
    image_tensor = process_images([image], image_processor, args)
    return image_tensor.to(model.device, dtype=torch.float16)

# Initialize Flask and flask-ngrok
app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when the app is run

# Define the Flask route
@app.route('/describe', methods=['POST'])
def describe_image():
    image_data = request.json['image_data']  # Can be a URL or Base64
    prompt = request.json['prompt']
    description = describe_image_function(image_data, prompt)
    return jsonify({"description": description})

# Run the Flask app
if __name__ == '__main__':
    app.run()

