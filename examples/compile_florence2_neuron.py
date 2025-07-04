"""
A script to compile Florence-2 model for NeuronX.

This script demonstrates how to load a pre-trained Florence-2 model from Hugging Face,
convert its state dict to a NeuronX-compatible format, and compile it using
NeuronFlorence2ForConditionalGenerationApp.

Example usage:
 python compile_florence2_neuron.py --model_id microsoft/Florence-2-base-ft --output_dir ./florence2_neuron_compiled
"""

import argparse
import os
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

from neuronx_distributed_inference.models.florence2.configuration_florence2_neuron import Florence2InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.florence2.modeling_florence2 import NeuronFlorence2ForConditionalGenerationApp

def main():
    parser = argparse.ArgumentParser(description="Compile Florence-2 model for NeuronX")
    parser.add_argument(
        "--model_id",
        type=str,
        default="microsoft/Florence-2-base-ft",
        help="Hugging Face model ID for Florence-2.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the compiled NeuronX model.",
    )
    parser.add_argument(
        "--tp_degree",
        type=int,
        default=2,
        help="Tensor parallelism degree for compilation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for compilation.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=1024,
        help="Image size for compilation (e.g., 1024 for 1024x1024).",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=128,
        help="Sequence length for compilation (max tokens for text).",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading Hugging Face model '{args.model_id}'...")
    # Load the original Hugging Face model and processor
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True)
    hf_processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    print("Creating NeuronX configuration...")
    # Create NeuronConfig and Florence2InferenceConfig
    neuron_config = NeuronConfig(
        tp_degree=args.tp_degree,
        batch_size=args.batch_size,
        torch_dtype=torch.float16, # Use float16 for better performance on Neuron
    )

    # Use the original model's config to create Florence2InferenceConfig
    # This ensures consistency with the pre-trained model's architecture
    florence2_config = Florence2InferenceConfig.from_hf_config(
        hf_model.config,
        neuron_config=neuron_config,
        image_size=args.image_size,
        sequence_length=args.sequence_length,
    )

    print("Converting Hugging Face state dict to NeuronX format...")
    # Convert HF state dict to NeuronX compatible state dict
    neuron_state_dict = NeuronFlorence2ForConditionalGenerationApp.convert_hf_to_neuron_state_dict(
        hf_model.state_dict(), florence2_config
    )

    print("Initializing NeuronFlorence2ForConditionalGenerationApp...")
    # Initialize the NeuronX application
    neuron_app = NeuronFlorence2ForConditionalGenerationApp(args.model_id, florence2_config)

    print("Compiling model for NeuronX...")
    # Compile the model
    neuron_app.compile(args.output_dir)

    print(f"Saving compiled model to '{args.output_dir}'...")
    # Save the compiled model
    neuron_app.save(args.output_dir)

    print("Compilation complete!")


if __name__ == "__main__":
    main()
