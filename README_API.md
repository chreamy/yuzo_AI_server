# Language Model API Server

This project provides an API server that allows you to train, fine-tune, and use GPT-style language models through a simple HTTP interface. It leverages the GPT implementation in this repository to provide an easy-to-use API for text generation.

## Features

- Load and use pretrained models (GPT-2, GPT-2 Medium, GPT-2 Large, GPT-2 XL)
- Train custom language models from scratch or fine-tune existing models
- Generate text with control over parameters like temperature and top-k sampling
- Simple RESTful API interface
- Command-line client for interacting with the API

## Installation

1. Clone this repository (if you haven't already)
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Starting the API Server

You can start the API server with default settings:

```bash
python api_server.py
```

Or with custom settings:

```bash
# Start server on a different port
python api_server.py --port 8000

# Start server with a pre-loaded model
python api_server.py --model_type gpt2

# Start server with a previously trained model checkpoint
python api_server.py --model_path out/my_model/ckpt.pt
```

The server will start on the specified port (default: 5000) and be accessible at `http://localhost:5000`.

## Using the API Client

The API client provides a convenient way to interact with the API server from the command line.

### Check Server Status

```bash
python api_client.py status
```

### List Available Models

```bash
python api_client.py list
```

### Load a Model

```bash
# Load a pretrained model
python api_client.py load --model_type gpt2

# Load a custom trained model
python api_client.py load --model_path out/my_model/ckpt.pt
```

### Generate Text

```bash
python api_client.py generate --prompt "Once upon a time"

# Customize generation parameters
python api_client.py generate --prompt "Once upon a time" --max_tokens 200 --temperature 0.9 --top_k 50
```

### Train a Model

```bash
# Fine-tune GPT-2 on the Shakespeare dataset
python api_client.py train --dataset shakesphere --output_dir out/shakespeare_model --model_type gpt2

# Customize training parameters
python api_client.py train --dataset openwebtext --output_dir out/my_model --model_type gpt2 \
  --batch_size 8 --learning_rate 5e-5 --max_iters 100
```

## API Endpoints

The following endpoints are available:

- `GET /api/status` - Get server status and information
- `GET /api/models` - List available models (pretrained and custom trained)
- `POST /api/load_model` - Load a model for inference
- `POST /api/generate` - Generate text using the loaded model
- `POST /api/train` - Train or fine-tune a model

For detailed API documentation, see below.

## Detailed API Documentation

### GET /api/status

Returns the current status of the API server.

**Response:**
```json
{
  "status": "online",
  "model_loaded": true,
  "device": "cuda",
  "timestamp": 1679441234.567
}
```

### GET /api/models

Lists all available models, both pretrained and custom trained.

**Response:**
```json
{
  "pretrained_models": ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
  "trained_models": [
    {
      "name": "my_model",
      "path": "out/my_model/ckpt.pt",
      "config": {
        "dataset": "openwebtext",
        "model_type": "gpt2",
        "batch_size": 4
      }
    }
  ]
}
```

### POST /api/load_model

Loads a model for inference. You must specify either `model_path` or `model_type`.

**Request:**
```json
{
  "model_path": "out/my_model/ckpt.pt"
}
```

Or:

```json
{
  "model_type": "gpt2-medium"
}
```

**Response:**
```json
{
  "message": "Model loaded successfully"
}
```

### POST /api/generate

Generates text based on a prompt using the currently loaded model.

**Request:**
```json
{
  "prompt": "Once upon a time",
  "max_new_tokens": 150,
  "temperature": 0.8,
  "top_k": 40
}
```

**Response:**
```json
{
  "prompt": "Once upon a time",
  "generated_text": "Once upon a time there was a kingdom far away...",
  "parameters": {
    "max_new_tokens": 150,
    "temperature": 0.8,
    "top_k": 40
  }
}
```

### POST /api/train

Starts a training job for a model.

**Request:**
```json
{
  "dataset": "shakesphere",
  "output_dir": "out/shakespeare_model",
  "model_type": "gpt2",
  "batch_size": 4,
  "learning_rate": 3e-5,
  "max_iters": 20,
  "eval_interval": 5
}
```

**Response:**
```json
{
  "message": "Training started",
  "config": {
    "dataset": "shakesphere",
    "output_dir": "out/shakespeare_model",
    "model_type": "gpt2",
    "batch_size": 4,
    "learning_rate": 3e-5,
    "max_iters": 20,
    "eval_interval": 5
  },
  "config_path": "out/shakespeare_model/train_config.json",
  "process_id": 12345
}
```

## Using with Programming Languages

You can interact with the API server from any programming language that can make HTTP requests. Here's an example in Python:

```python
import requests

# Define the server URL
api_url = "http://localhost:5000"

# Load a model
response = requests.post(f"{api_url}/api/load_model", json={"model_type": "gpt2"})
print(response.json())

# Generate text
response = requests.post(f"{api_url}/api/generate", json={
    "prompt": "Once upon a time",
    "max_new_tokens": 150,
    "temperature": 0.8,
    "top_k": 40
})
print(response.json()["generated_text"])
```

## Limitations

- The API server is designed for demonstration and local use, not for production environments
- Training large models requires significant GPU resources
- Token limits apply depending on the model's maximum context size (default: 1024 tokens)

## Troubleshooting

If you encounter issues:

1. Ensure all dependencies are installed correctly
2. Check that your GPU has enough memory for the chosen model
3. Make sure the API server is running before trying to use the client
4. For training issues, check the log files in the output directory

## Advanced Usage

### Using a Custom Dataset

To train on a custom dataset, you'll need to prepare it in the format required by the training script. See the documentation in the main repository for details on dataset preparation.

### Deploying to Production

For production use, consider:
- Using a WSGI server like Gunicorn
- Adding authentication to the API endpoints
- Implementing rate limiting
- Setting up proper monitoring and logging 