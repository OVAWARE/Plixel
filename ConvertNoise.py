import argparse
from PIL import Image

def resize_image(input_path, output_path):
    # Open the input image
    with Image.open(input_path) as img:
        # Convert to RGB mode if it's not already (this handles PNG with alpha channel, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize the image to 16x16 pixels
        resized_img = img.resize((16, 16), Image.LANCZOS)
        
        # Save the resized image
        resized_img.save(output_path)
        print(f"Resized image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize any image to 16x16 pixels")
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument("output_image", help="Path to save the output 16x16 image")
    
    args = parser.parse_args()
    
    resize_image(args.input_image, args.output_image)