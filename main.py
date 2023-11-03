import base64
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import torch
from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok

# TensorFlow Model
tf_model_path = "real-vs-screen_reav_screen.h5"
tf_model = tf.keras.models.load_model(tf_model_path)
class_names = ["real", "screen"]

# llava-torch Package
from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import process_images
from llava.model.builder import load_pretrained_model
# Initialize llava-torch model variables to None
tokenizer = None
llava_model = None
image_processor = None
context_len = None
# Load the pretrained llava-torch model
def load_llava_model():
    global tokenizer, llava_model, image_processor, context_len
    # Check if the llava-torch model is already loaded
    if llava_model is None:
        # Load the pretrained llava-torch model
        MODEL = "4bit/llava-v1.5-13b-3GB"
        model_name = MODEL.split('/')[-1]
        tokenizer, llava_model, image_processor, context_len = load_pretrained_model(
            model_path=MODEL, model_base=None, model_name=model_name, load_4bit=True
        )
# Flask App
app = Flask(__name__)
run_with_ngrok(app)

# TensorFlow Helper Function
def predict_image_class(image_data):
    image_bytes = BytesIO(base64.b64decode(image_data))
    image = Image.open(image_bytes).convert("RGB").resize((256, 256))
    image_array = np.expand_dims(np.array(image), axis=0)
    predictions = tf_model.predict(image_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return predicted_class, confidence

# llava-torch Helper Functions
def load_image(image_data):
    if image_data.startswith('http'):
        response = requests.get(image_data)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            return None
    else:
        image_data = BytesIO(base64.b64decode(image_data))
        image = Image.open(image_data).convert("RGB")
    return image

def process_image_for_llava(image):
    args = {"image_aspect_ratio": "pad"}
    image_tensor = process_images([image], image_processor, args)
    return image_tensor.to(llava_model.device, dtype=torch.float16)

# Flask Route for Image Description
# Flask Route for Image Description
@app.route('/describe', methods=['POST'])
def describe_image():
    data = request.json
    image_data = data['image_data']
    prompt = data['prompt']

    # First, check if the image is real or a screen capture
    predicted_class, confidence = predict_image_class(image_data)
    if predicted_class == "screen":
        return jsonify(description=f"This is a screen image with a confidence of {confidence:.2%}.")

    # If the image is real, process it for llava description
    image = load_image(image_data)
    if image is None:
        return jsonify(description="Could not load the image."), 400

    processed_image = process_image_for_llava(image)

    # Prepare prompt for llava model
    conv = conv_templates['default'].copy()
    roles = conv.roles
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv.append_message(roles[0], prompt)
    conv.append_message(roles[1], None)

    # Convert conversation to model inputs
    tokenized_prompt = tokenizer.encode(conv.get_prompt(), return_tensors='pt')

    # Generate description with the llava model
    llava_model.eval()
    with torch.no_grad():
        outputs = llava_model.generate(input_ids=tokenized_prompt)
    full_description = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extracting only the assistant's response
    description = full_description.split('###Assistant:')[-1].strip()

    return jsonify(description=description)


# Run the Flask App
if __name__ == '__main__':
    load_llava_model()
    app.run(port=5000)  
