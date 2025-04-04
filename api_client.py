import requests
import json
import argparse
import time

class LLMClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
    
    def get_status(self):
        """Check if the API server is running and get its status"""
        response = requests.get(f"{self.base_url}/api/status")
        return response.json()
    
    def list_models(self):
        """List available models (both pretrained and custom trained)"""
        response = requests.get(f"{self.base_url}/api/models")
        return response.json()
    
    def load_model(self, model_path=None, model_type=None):
        """Load a specific model into the server"""
        data = {}
        if model_path:
            data["model_path"] = model_path
        if model_type:
            data["model_type"] = model_type
        
        response = requests.post(f"{self.base_url}/api/load_model", json=data)
        return response.json()
    
    def generate_text(self, prompt, max_new_tokens=100, temperature=0.8, top_k=40):
        """Generate text using the loaded model"""
        data = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k
        }
        
        response = requests.post(f"{self.base_url}/api/generate", json=data)
        return response.json()
    
    def train_model(self, config=None):
        """Start a training job with the specified configuration"""
        if config is None:
            config = {}
        
        # Set default configuration values if not provided
        default_config = {
            "dataset": "openwebtext",
            "output_dir": "out/custom_model",
            "model_type": "gpt2",
            "batch_size": 4,
            "learning_rate": 3e-5,
            "max_iters": 20,
            "eval_interval": 5
        }
        
        # Update default config with provided config
        for key, value in config.items():
            default_config[key] = value
        
        response = requests.post(f"{self.base_url}/api/train", json=default_config)
        return response.json()

def main():
    parser = argparse.ArgumentParser(description="LLM API Client")
    parser.add_argument("--server", default="http://localhost:5000", help="API server URL")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Status parser
    status_parser = subparsers.add_parser("status", help="Check server status")
    
    # List models parser
    list_parser = subparsers.add_parser("list", help="List available models")
    
    # Load model parser
    load_parser = subparsers.add_parser("load", help="Load a model")
    load_parser.add_argument("--model_path", help="Path to a trained model checkpoint")
    load_parser.add_argument("--model_type", help="Pretrained model type (e.g., gpt2, gpt2-medium)")
    
    # Generate text parser
    generate_parser = subparsers.add_parser("generate", help="Generate text with the loaded model")
    generate_parser.add_argument("--prompt", required=True, help="Text prompt to generate from")
    generate_parser.add_argument("--max_tokens", type=int, default=100, help="Maximum new tokens to generate")
    generate_parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    generate_parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling parameter")
    
    # Train model parser
    train_parser = subparsers.add_parser("train", help="Train a custom model")
    train_parser.add_argument("--config", help="JSON file with training configuration")
    train_parser.add_argument("--dataset", default="openwebtext", help="Dataset to train on")
    train_parser.add_argument("--output_dir", default="out/custom_model", help="Output directory for the model")
    train_parser.add_argument("--model_type", default="gpt2", help="Base model to start from")
    train_parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    train_parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    train_parser.add_argument("--max_iters", type=int, default=20, help="Maximum training iterations")
    
    args = parser.parse_args()
    client = LLMClient(args.server)
    
    if args.command == "status":
        status = client.get_status()
        print(json.dumps(status, indent=2))
    
    elif args.command == "list":
        models = client.list_models()
        print("Available models:")
        print("Pretrained models:")
        for model in models.get("pretrained_models", []):
            print(f"  - {model}")
        print("\nTrained models:")
        for model in models.get("trained_models", []):
            print(f"  - {model['name']} (path: {model['path']})")
            if model['config']:
                print(f"    Config: {json.dumps(model['config'], indent=2)}")
    
    elif args.command == "load":
        if not args.model_path and not args.model_type:
            print("Error: Either --model_path or --model_type must be provided")
            return
        
        print(f"Loading model: {args.model_path or args.model_type}")
        result = client.load_model(args.model_path, args.model_type)
        print(json.dumps(result, indent=2))
    
    elif args.command == "generate":
        print(f"Generating text from prompt: {args.prompt}")
        result = client.generate_text(
            args.prompt, 
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print("\nGenerated text:")
            print("--------------")
            print(result["generated_text"])
            print("--------------")
            print(f"Parameters: {json.dumps(result['parameters'], indent=2)}")
    
    elif args.command == "train":
        config = {}
        
        # Load config from file if provided
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
        
        # Override with command line arguments
        if args.dataset:
            config["dataset"] = args.dataset
        if args.output_dir:
            config["output_dir"] = args.output_dir
        if args.model_type:
            config["model_type"] = args.model_type
        if args.batch_size:
            config["batch_size"] = args.batch_size
        if args.learning_rate:
            config["learning_rate"] = args.learning_rate
        if args.max_iters:
            config["max_iters"] = args.max_iters
        
        print(f"Starting training with config: {json.dumps(config, indent=2)}")
        result = client.train_model(config)
        print(json.dumps(result, indent=2))
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 