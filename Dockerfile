# Start from the latest Ubuntu base image
FROM ubuntu:22.04

# Set the non-interactive frontend (to avoid user prompts during installation)
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, pip, and other necessary tools
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip3 install --upgrade pip setuptools

# Install PyTorch for CPU usage
RUN pip3 install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install uvicorn
RUN pip3 install uvicorn

# Copy the requirements.txt first to leverage Docker's caching
COPY requirements.txt /app/requirements.txt

# Install the rest of the Python dependencies
WORKDIR /app
RUN pip3 install -r requirements.txt

# Copy the rest of the application files to the container
COPY . /app

# Expose the port that the Gradio app will run on
EXPOSE 8070

# Set the default command to run your Python script with uvicorn when the container starts
CMD ["uvicorn", "gradio_test:demo", "--host", "0.0.0.0", "--port", "8070"]
