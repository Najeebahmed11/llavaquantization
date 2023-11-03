FROM python:3.10.12-slim

WORKDIR /app

COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir flask requests torch torchvision transformers accelerate bitsandbytes llava-torch Pillow


EXPOSE 8080

CMD ["python", "app.py"]
