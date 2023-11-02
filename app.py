from flask import Flask, request, jsonify
import requests
import torch
from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from PIL import Image

app = Flask(__name__)

MODEL = "4bit/llava-v1.5-13b-3GB"
model_name = get_model_name_from_path(MODEL)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=MODEL, model_base=None, model_name=model_name, load_4bit=True
)
CONV_MODE = "llava_v0"

def load_image(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image

def process_image(image):
    args = {"image_aspect_ratio": "pad"}
    image_tensor = process_images([image], image_processor, args)
    return image_tensor.to(model.device, dtype=torch.float16)

@app.route('/describe', methods=['POST'])
def describe_image():
    image_url = request.json['image_url']
    prompt = request.json['prompt']

    image = load_image(image_url)
    processed_image = process_image(image)
    
    conv = conv_templates[CONV_MODE].copy()
    roles = conv.roles
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv.append_message(roles[0], prompt)
    conv.append_message(roles[1], None)
    tokenized_prompt = conv.get_prompt()

    # Using your model to generate the description
    with torch.no_grad():
        output = model.generate(tokenized_prompt)
    description = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({"description": description})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
