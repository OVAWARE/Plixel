import torch
from tqdm import tqdm
from parameters import *
from noise import get_beta, noise, alphat
from util import sharp_scale, draw_list
import random
from text_embed import get_text_embedding
import clip
from PIL import Image
import numpy as np
from parameters import NUM_CHANNELS

def get_alpha(t):
    return get_beta(t)

def sample_v1(model, N, prompt, display_count=4, noise_mul=6, text_scale=1.0, initial_noise=None):
    with torch.no_grad():
        size = (N, NUM_CHANNELS, ART_SIZE, ART_SIZE)
        if initial_noise is not None:
            h = torch.nn.functional.interpolate(initial_noise, size=(ART_SIZE, ART_SIZE), mode='bilinear', align_corners=False)
            h = h.repeat(N, 1, 1, 1)
            if noise_mul != 0:
                random_noise = torch.randn_like(h)
                h = h + random_noise * noise_mul
        else:
            h = torch.randn(size).to(device) * noise_mul

        text_embed = get_text_embedding(prompt).to(device)
        text_embed = text_embed.repeat(N, 1) * text_scale

        s = ACTUAL_STEPS // display_count if display_count != 0 else ACTUAL_STEPS*5

        for t in tqdm(range(ACTUAL_STEPS, 0, -1)):
            ts = torch.full((N,), t, device=device, dtype=torch.long)
            h = sample_step_v1(model, h, ts, text_embed, noise_mul)

            if t % s == (s//2):
                print("ITERATION", t)
                draw_list((h+1)/2)
        
        return (h+1)/2

def sample_v2(model, N, prompt, display_count=4, noise_mul=0.5, text_scale=1.0, initial_noise=None):
    with torch.no_grad():
        size = (N, NUM_CHANNELS, ART_SIZE, ART_SIZE)
        h = torch.randn(size).to(device)
        
        if initial_noise is not None:
            initial_noise = torch.nn.functional.interpolate(initial_noise, size=(ART_SIZE, ART_SIZE), mode='bilinear', align_corners=False)
            initial_noise = initial_noise.repeat(N, 1, 1, 1)
            h = h * (1 - noise_mul) + initial_noise * noise_mul

        text_embed = get_text_embedding(prompt).to(device)
        text_embed = text_embed.repeat(N, 1) * text_scale

        s = ACTUAL_STEPS // display_count if display_count != 0 else ACTUAL_STEPS*5

        for t in tqdm(range(ACTUAL_STEPS, 0, -1)):
            ts = torch.full((N,), t, device=device, dtype=torch.long)
            h = sample_step_v2(model, h, ts, text_embed)

            if t % s == (s//2):
                print("ITERATION", t)
                draw_list((h+1)/2)
        
        return (h+1)/2

def sample_step_v1(model, im, ts, text_embed, noise_mul=8):
    with torch.no_grad():
        N, C, H, W = im.shape
        z = torch.randn_like(im)
        if ts[0] == 1:
            z *= 0
        noise = model(im, ts, text_embed)
        alpha = 1 - get_beta(ts)
        alpha = alpha.view(-1, 1, 1, 1)
        alphat_ts = alphat[ts].view(-1, 1, 1, 1)
        new_mean = alpha**-0.5 * (im - (1-alpha)/(1-alphat_ts)**0.5 * noise)
        add_noise = get_alpha(ts).view(-1, 1, 1, 1) * z * noise_mul
        return new_mean + add_noise

def sample_step_v2(model, im, ts, text_embed):
    with torch.no_grad():
        N, C, H, W = im.shape
        z = torch.randn_like(im)
        if ts[0] == 1:
            z *= 0
        noise = model(im, ts, text_embed)
        alpha = 1 - get_beta(ts)
        alpha = alpha.view(-1, 1, 1, 1)
        alphat_ts = alphat[ts].view(-1, 1, 1, 1)
        new_mean = alpha**-0.5 * (im - (1-alpha)/(1-alphat_ts)**0.5 * noise)
        add_noise = get_alpha(ts).view(-1, 1, 1, 1) * z
        return new_mean + add_noise

if __name__=="__main__":
    import argparse
    import util
    import model
    from matplotlib import image

    parser = argparse.ArgumentParser("sample.py")

    parser.add_argument("model_path", help="Path to the model.")
    parser.add_argument("num_samples", help="Number of samples.", type=int)
    parser.add_argument("-o", help="Path to save output to. The generated image will be saved as a single spritesheet .png here. If empty, does not save.", default="", nargs='?')
    parser.add_argument("-noise_mul", help="Noise multiplier. For v1: standard deviation during sampling. For v2: initial noise image influence (0: pure random, 1: only initial image). Default: 6.0 for v1, 0.5 for v2", type=float)
    parser.add_argument("-prompt", help="Text prompt for image generation.", default="", nargs='?', type=str)
    parser.add_argument("-text_scale", help="Scaling factor for text embedding. Higher values increase text influence. Default: 1.0", default=1.0, type=float)
    parser.add_argument("-noise_image", help="Path to a noise image to use as initial noise. Can be any size.", default=None, nargs='?', type=str)
    parser.add_argument("-noise_version", help="Version of noise handling to use. 'v1' for previous version, 'v2' for current version. Default: v2", default="v2", choices=["v1", "v2"])

    args = parser.parse_args()

    model = model.UNet().to(device).eval()
    epoch = util.load_model(model, args.model_path)

    if args.prompt == "":
        prompt = input("Enter a text prompt for image generation: ")
    else:
        prompt = args.prompt

    initial_noise = None
    if args.noise_image:
        noise_img = Image.open(args.noise_image).convert('RGBA')
        print(f"Loaded noise image of size {noise_img.size}")
        noise_array = np.array(noise_img) / 127.5 - 1  # Convert to -1 to 1 range
        initial_noise = torch.from_numpy(noise_array).permute(2, 0, 1).float().unsqueeze(0).to(device)

    # Set default noise_mul based on version if not provided
    if args.noise_mul is None:
        args.noise_mul = 6.0 if args.noise_version == "v1" else 0.5

    if args.noise_version == "v1":
        xs = sample_v1(model, args.num_samples, prompt, display_count=0, noise_mul=args.noise_mul, 
                    text_scale=args.text_scale, initial_noise=initial_noise)
    else:
        xs = sample_v2(model, args.num_samples, prompt, display_count=0, noise_mul=args.noise_mul, 
                    text_scale=args.text_scale, initial_noise=initial_noise)

    sheet = util.to_drawable(xs)

    if args.o != "":
        rgba_sheet = np.zeros((sheet.shape[0], sheet.shape[1], 4), dtype=np.float32)
        rgba_sheet[:,:,:4] = sheet
        image.imsave(args.o, (rgba_sheet * 255).astype(np.uint8), format='png')
    else:
        util.draw_im(sheet)
        input("Press enter...")