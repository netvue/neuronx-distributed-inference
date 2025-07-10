import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import sys
import os
import argparse

# Add the src directory to the Python path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from neuronx_distributed_inference.models.florence2.modeling_florence2 import NeuronFlorence2ForCausalLM

def run_inference(model_path, image_path, prompt):
    print(f"Loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading model from {model_path}...")
    # Load the HuggingFace model first to get the state_dict
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    hf_state_dict = hf_model.state_dict()

    # Initialize NeuronX model and load converted state_dict
    neuron_config = NeuronFlorence2ForCausalLM.get_config_cls().from_pretrained(model_path)
    neuron_model = NeuronFlorence2ForCausalLM(model_path, config=neuron_config)
    
    # Convert HuggingFace state_dict to NeuronX compatible state_dict
    neuron_state_dict = NeuronFlorence2ForCausalLM.convert_hf_to_neuron_state_dict(hf_state_dict, neuron_config)
    
    # Load the converted state_dict into the NeuronX model
    neuron_model.load_state_dict(neuron_state_dict, strict=False) # strict=False because some keys might not match exactly

    print(f"Loading image from {image_path}...")
    image = Image.open(image_path).convert("RGB")

    print("Preparing inputs...")
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    print("Running inference...")
    # Move inputs to CPU for now, as NeuronX model handles device placement internally
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]

    # Perform generation
    outputs = neuron_model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False,
        early_stopping=True
    )

    print("Decoding outputs...")
    decoded_text = processor.batch_decode(outputs, skip_special_tokens=False)[0]

    print("\n--- Inference Result ---")
    print(decoded_text)
    print("------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Florence-2 inference on NeuronX.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Florence-2 model (e.g., microsoft/Florence-2-base-ft).")
    parser.add_argument("--image_path", type=str, default="examples/dog.jpg", help="Path to the input image (default: examples/dog.jpg).")
    parser.add_argument("--prompt", type=str, default="<OD>", help="Prompt for the model (default: <OD> for Object Detection).")

    args = parser.parse_args()

    run_inference(args.model_path, args.image_path, args.prompt)