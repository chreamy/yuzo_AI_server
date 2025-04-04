import os
import json
import time
import torch
import argparse
import importlib.util
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import GPTConfig, GPT
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
CORS(app)

# Global variables to store model and tokenizer
model = None
encode = None
decode = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Global variables for Hugging Face model
hf_model = None
hf_tokenizer = None

def load_model(model_path=None, model_type=None):
    """
    Load a model either from a checkpoint or use a pretrained GPT-2 model
    """
    global model, encode, decode
    
    if model_path and os.path.exists(model_path):
        print(f"Loading model from checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # Fix keys in state dict if needed
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        
        # Try to load the encoding/decoding functions from meta.pkl if available
        if 'config' in checkpoint and 'dataset' in checkpoint['config']:
            meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
            if os.path.exists(meta_path):
                print(f"Loading meta from {meta_path}")
                import pickle
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                stoi, itos = meta['stoi'], meta['itos']
                encode = lambda s: [stoi[c] for c in s]
                decode = lambda l: ''.join([itos[i] for i in l])
            else:
                # Use tiktoken as fallback
                import tiktoken
                enc = tiktoken.get_encoding("gpt2")
                encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
                decode = lambda l: enc.decode(l)
        else:
            # Use tiktoken as fallback
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            decode = lambda l: enc.decode(l)
    
    elif model_type and model_type.startswith('gpt2'):
        print(f"Loading pretrained model: {model_type}")
        model = GPT.from_pretrained(model_type, dict(dropout=0.0))
        
        # Use tiktoken for GPT-2 models
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
    
    else:
        raise ValueError("Either model_path or model_type must be provided")
    
    model.eval()
    model.to(device)
    print("Model loaded successfully")

def load_falcon_model(quantize=True):
    """
    Load Falcon model that doesn't require authentication
    """
    global hf_model, hf_tokenizer
    
    model_name = "tiiuae/falcon-7b-instruct"
    print(f"Loading Falcon model: {model_name}")
    
    # Load tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with quantization
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=True if quantize else False,
        torch_dtype=torch.float16,
    )
    
    print(f"{model_name} model loaded successfully")
    return True

@app.route('/api/generate', methods=['POST'])
def generate_text():
    """
    API endpoint to generate text using the loaded model
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Please load a model first."}), 500
    
    data = request.json
    prompt = data.get('prompt', '')
    max_new_tokens = data.get('max_new_tokens', 100)
    temperature = data.get('temperature', 0.8)
    top_k = data.get('top_k', 40)
    
    try:
        with torch.no_grad():
            # Encode the prompt
            start_ids = encode(prompt)
            x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
            
            # Generate text
            context = torch.amp.autocast(device_type='cuda' if 'cuda' in device else 'cpu')
            with context:
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            
            # Decode the generated text
            generated_text = decode(y[0].tolist())
            
            return jsonify({
                "prompt": prompt,
                "generated_text": generated_text,
                "parameters": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_k": top_k
                }
            })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def load_config_file(config_file_path):
    """
    Load configuration from a Python file using importlib
    """
    spec = importlib.util.spec_from_file_location("config_module", config_file_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Extract all variables that don't start with underscore
    config = {k: v for k, v in vars(config_module).items() 
              if not k.startswith('_') and not callable(v)}
    
    return config

@app.route('/api/train', methods=['POST'])
def train_model():
    """
    API endpoint to train or fine-tune a model
    """
    data = request.json
    
    # Training configuration
    dataset = data.get('dataset', 'openwebtext')
    output_dir = data.get('output_dir', 'out')
    model_type = data.get('model_type', 'gpt2')  # Which model to start from
    config_file = data.get('config_file', None)  # Optional config file
    
    # Process config_file if provided
    config_args = []
    if config_file:
        if config_file.startswith('config/'):
            # Use a predefined config
            config_args.append(config_file)
        else:
            # Error if config file doesn't exist
            if not os.path.exists(config_file):
                return jsonify({"error": f"Config file {config_file} not found"}), 400
            config_args.append(config_file)
    
    # Override config with specific parameters from request
    for key, value in data.items():
        if key not in ['config_file']:
            config_args.append(f'--{key}={value}')
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the training config
    config_path = os.path.join(output_dir, 'train_config.json')
    with open(config_path, 'w') as f:
        json.dump(data, f)
    
    # Build the command
    cmd = ['python', 'train.py'] + config_args
    
    print(f"Running training command: {' '.join(cmd)}")
    
    # Call the training function in a background process
    import subprocess
    try:
        # Start the training process
        process = subprocess.Popen(cmd)
        
        return jsonify({
            "message": "Training started",
            "config": data,
            "config_path": config_path,
            "process_id": process.pid,
            "command": ' '.join(cmd)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """
    Get the status of the API and model
    """
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "device": device,
        "timestamp": time.time()
    })

@app.route('/api/configs', methods=['GET'])
def list_configs():
    """
    List available configuration files
    """
    configs = []
    
    if os.path.exists('config'):
        for item in os.listdir('config'):
            if item.endswith('.py'):
                config_path = os.path.join('config', item)
                try:
                    # Try to load the config to get a summary
                    config_data = load_config_file(config_path)
                    description = {
                        "file": item,
                        "path": config_path,
                        "dataset": config_data.get('dataset', 'unknown'),
                        "init_from": config_data.get('init_from', 'unknown'),
                        "settings": {
                            "batch_size": config_data.get('batch_size', 'default'),
                            "learning_rate": config_data.get('learning_rate', 'default'),
                            "max_iters": config_data.get('max_iters', 'default')
                        }
                    }
                    configs.append(description)
                except Exception as e:
                    configs.append({
                        "file": item,
                        "path": config_path,
                        "error": str(e)
                    })
    
    return jsonify({
        "configs": configs
    })

@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """
    List available datasets
    """
    datasets = []
    
    if os.path.exists('data'):
        for item in os.listdir('data'):
            dataset_dir = os.path.join('data', item)
            if os.path.isdir(dataset_dir):
                # Check if this directory contains the expected dataset files
                has_train = os.path.exists(os.path.join(dataset_dir, 'train.bin'))
                has_val = os.path.exists(os.path.join(dataset_dir, 'val.bin'))
                has_meta = os.path.exists(os.path.join(dataset_dir, 'meta.pkl'))
                has_prepare = os.path.exists(os.path.join(dataset_dir, 'prepare.py'))
                
                datasets.append({
                    "name": item,
                    "path": dataset_dir,
                    "files": {
                        "train.bin": has_train,
                        "val.bin": has_val,
                        "meta.pkl": has_meta,
                        "prepare.py": has_prepare
                    },
                    "prepared": has_train and has_val
                })
    
    return jsonify({
        "datasets": datasets
    })

@app.route('/api/models', methods=['GET'])
def list_models():
    """
    List available trained models
    """
    models = []
    
    # Check for models in the out directory
    if os.path.exists('out'):
        for item in os.listdir('out'):
            model_dir = os.path.join('out', item)
            if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, 'ckpt.pt')):
                # Try to read the config to get more info
                config_path = os.path.join(model_dir, 'train_config.json')
                config = {}
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        try:
                            config = json.load(f)
                        except:
                            pass
                
                models.append({
                    "name": item,
                    "path": os.path.join(model_dir, 'ckpt.pt'),
                    "config": config
                })
    
    # Also list available pretrained models
    pretrained = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
    
    return jsonify({
        "trained_models": models,
        "pretrained_models": pretrained
    })

@app.route('/api/prepare_dataset', methods=['POST'])
def prepare_dataset():
    """
    Prepare a dataset for training by running its prepare.py script
    """
    data = request.json
    dataset_name = data.get('dataset')
    
    if not dataset_name:
        return jsonify({"error": "Dataset name is required"}), 400
    
    dataset_dir = os.path.join('data', dataset_name)
    prepare_script = os.path.join(dataset_dir, 'prepare.py')
    
    if not os.path.exists(prepare_script):
        return jsonify({"error": f"Prepare script not found for dataset {dataset_name}"}), 404
    
    try:
        import subprocess
        process = subprocess.Popen(['python', prepare_script], cwd=dataset_dir)
        
        return jsonify({
            "message": f"Dataset preparation started for {dataset_name}",
            "process_id": process.pid
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/load_model', methods=['POST'])
def api_load_model():
    """
    API endpoint to load a specific model
    """
    data = request.json
    model_path = data.get('model_path')
    model_type = data.get('model_type')
    
    try:
        load_model(model_path, model_type)
        return jsonify({"message": "Model loaded successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Store conversation history
conversations = {}

@app.route('/api/mixtral/chat', methods=['POST'])
def mixtral_chat():
    """
    Chatbot endpoint using Falcon
    """
    global hf_model, hf_tokenizer
    
    # Load model if not already loaded
    if hf_model is None or hf_tokenizer is None:
        try:
            load_falcon_model()
        except Exception as e:
            return jsonify({"error": f"Failed to load model: {str(e)}"}), 500
    
    data = request.json
    user_message = data.get('message', '')
    conversation_id = data.get('conversation_id', None)
    max_tokens = data.get('max_tokens', 1024)
    temperature = data.get('temperature', 0.7)
    
    # Get or create conversation history
    if conversation_id and conversation_id in conversations:
        history = conversations[conversation_id]
    else:
        import uuid
        conversation_id = str(uuid.uuid4())
        history = []
        conversations[conversation_id] = history
    
    # Format conversation in Falcon's expected format
    prompt = "You are a helpful assistant.\n\n"
    for entry in history:
        prompt += f"User: {entry['user']}\nAssistant: {entry['assistant']}\n\n"
    
    prompt += f"User: {user_message}\nAssistant:"
    
    try:
        # Generate response
        inputs = hf_tokenizer(prompt, return_tensors="pt").to(hf_model.device)
        outputs = hf_model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
        )
        
        # Extract just the assistant's response
        full_response = hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = full_response.replace(prompt, "").strip()
        
        # Update history
        history.append({"user": user_message, "assistant": assistant_response})
        
        return jsonify({
            "response": assistant_response,
            "conversation_id": conversation_id
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/mixtral/load', methods=['POST'])
def load_mixtral():
    """
    API endpoint to explicitly load the model
    """
    try:
        quantize = request.json.get('quantize', True)
        # Use Falcon instead of Mixtral
        success = load_falcon_model(quantize=quantize)
        return jsonify({"message": "Falcon model loaded successfully", "quantized": quantize})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the LLM API server")
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    parser.add_argument('--model_path', type=str, help='Path to a trained model checkpoint')
    parser.add_argument('--model_type', type=str, help='Pretrained model type (e.g., gpt2, gpt2-medium)')
    parser.add_argument('--config_file', type=str, help='Configuration file for model settings')
    args = parser.parse_args()
    
    # Load a model if specified
    if args.model_path or args.model_type:
        try:
            load_model(args.model_path, args.model_type)
        except Exception as e:
            print(f"Error loading model: {e}")
    
    # Run the server
    app.run(host='0.0.0.0', port=args.port) 