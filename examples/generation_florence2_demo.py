
"""
A demo script for running Florence-2 inference on NeuronX.

This script demonstrates how to load a NeuronX-compiled Florence-2 model
and use it for various vision-language tasks like captioning or object detection.

Example usage:
 python generation_florence2_demo.py --model /path/to/neuron_model --prompt "<CAPTION>"
 python generation_florence2_demo.py --model /path/to/neuron_model --prompt "<OD>" --image_path /path/to/your/image.jpg
"""

import argparse
from PIL import Image
import torch
from transformers import AutoProcessor

from neuronx_distributed_inference.models.florence2.modeling_florence2 import Florence2ForConditionalGenerationNeuron

def main():
    """
    Main function to run the Florence-2 demo.
    """
    parser = argparse.ArgumentParser(description="Florence-2 NeuronX Inference Demo")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The path to the NeuronX compiled Florence-2 model directory.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="dog.jpg",
        help="The path to the input image.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The task prompt, e.g., '<CAPTION>', '<OD>', '<MORE_DETAILED_CAPTION>'.",
    )
    parser.add_argument(
        "--tp_degree",
        type=int,
        default=2,
        help="Tensor parallelism degree for the model.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate.",
    )
    args = parser.parse_args()

    print(f"Loading processor from 'microsoft/Florence-2-base-ft'...")
    # Load the processor from the official Hugging Face model hub
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading image from '{args.image_path}'...")
    # Load the input image
    image = Image.open(args.image_path).convert("RGB")

    print(f"Loading NeuronX model from '{args.model}' with tp_degree={args.tp_degree}...")
    # Load the NeuronX-compatible Florence-2 model
    model = Florence2ForConditionalGenerationNeuron.from_pretrained(
        args.model, tp_degree=args.tp_degree
    )
    print("Model loaded successfully.")

    # Prepare the inputs for the model
    # The prompt guides the model on which task to perform
    text = args.prompt
    inputs = processor(text=text, images=image, return_tensors="pt")

    print(f"Running inference with prompt: '{args.prompt}'...")
    # Generate output IDs
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=args.max_new_tokens,
            num_beams=3,
        )

    print("Decoding results...")
    # Decode the generated IDs and post-process the output
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # The post_process_generation function is crucial for parsing the model's raw output
    # into a human-readable format for the specific task.
    image_size = (image.width, image.height)
    parsed_answer = processor.post_process_generation(
        generated_text, task=args.prompt, image_size=image_size
    )

    print("\n--- Inference Result ---")
    print(f"Task Prompt: {args.prompt}")
    print(f"Generated Answer: {parsed_answer}")
    print("------------------------\n")


if __name__ == "__main__":
    main()
