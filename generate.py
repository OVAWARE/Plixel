import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer, BertModel
import argparse
import numpy as np

# Import the model architecture from train.py
from train import CVAE, TextEncoder, LATENT_DIM, HIDDEN_DIM

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def clean_image(image, threshold=0.75):
    """
    Clean up the image by setting pixels with opacity <= threshold to 0% opacity
    and pixels above the threshold to 100% visibility.
    """
    np_image = np.array(image)
    alpha_channel = np_image[:, :, 3]
    alpha_channel[alpha_channel <= int(threshold * 255)] = 0
    alpha_channel[alpha_channel > int(threshold * 255)] = 255  # Set to 100% visibility
    return Image.fromarray(np_image)

def generate_image(model, text_prompt, device):
    # Encode text prompt using BERT tokenizer
    encoded_input = tokenizer(text_prompt, padding=True, truncation=True, return_tensors="pt")
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)
    
    # Generate text encoding
    with torch.no_grad():
        text_encoding = model.text_encoder(input_ids, attention_mask)
    
    # Sample from the latent space
    z = torch.randn(1, LATENT_DIM).to(device)
    
    # Generate image
    with torch.no_grad():
        generated_image = model.decode(z, text_encoding)
    
    # Convert the generated tensor to a PIL Image
    generated_image = generated_image.squeeze(0).cpu()
    generated_image = (generated_image + 1) / 2  # Rescale from [-1, 1] to [0, 1]
    generated_image = generated_image.clamp(0, 1)
    generated_image = transforms.ToPILImage()(generated_image)
    
    return generated_image

def main():
    parser = argparse.ArgumentParser(description="Generate an image from a text prompt using the trained CVAE model.")
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    parser.add_argument("--output", type=str, default="generated_image.png", help="Output image file name")
    parser.add_argument("--model_path", type=str, default="cvae_text2image_model.pth", help="Path to the trained model")
    parser.add_argument("--clean", action="store_true", help="Clean up the image by removing low opacity pixels")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    text_encoder = TextEncoder(hidden_size=HIDDEN_DIM, output_size=HIDDEN_DIM)
    model = CVAE(text_encoder).to(device)
    
    # Load the trained model
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Generate image from prompt
    generated_image = generate_image(model, args.prompt, device)
    
    # Clean up the image if the flag is set
    if args.clean:
        generated_image = clean_image(generated_image)
    
    # Save the generated image
    generated_image.save(args.output)
    print(f"Generated image saved as {args.output}")

if __name__ == "__main__":
    main()