FROM python:3.10.12-slim

WORKDIR /app

COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install requests torch torchvision transformers accelerate bitsandbytes llava-torch Pillow flask-ngrok tensorflow Pillow flask runpod



# COPY entrypoint.sh /app/
# RUN chmod +x /app/entrypoint.sh

EXPOSE 8080

# # Use the entrypoint script to start the app
# RUN ["/app/entrypoint.sh"]

CMD ["python", "app.py"]
