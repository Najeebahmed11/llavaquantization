import torch
from llava.model.builder import load_pretrained_model

def model_warm_up():
    # Assuming you have a function to load your model similar to app.py
    MODEL = "4bit/llava-v1.5-13b-3GB"
    model_name = MODEL.split('/')[-1]
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=MODEL, model_base=None, model_name=model_name, load_4bit=True
    )
def run_pre_startup_tasks():
    print("Warming up the model...")
    model_warm_up()
if __name__ == "__main__":
    run_pre_startup_tasks()    