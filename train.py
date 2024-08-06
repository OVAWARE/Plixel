# FILE: train.py

import torch
from torch import nn
import numpy as np
import torch.utils.data as dutils
import wandb
import subprocess

from parameters import ACTUAL_STEPS, device, NUM_CHANNELS
import noise
import util
import sample
import dset
import model as pixmodel
from text_embed import get_text_embedding

learning_rate = 1e-4
batch_size = 128

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    iters = max(size // batch_size, 1)  # Ensure at least 1 iteration
    print_on = iters // 3  # Print every third iteration

    model.train()
    tot_loss = 0

    for batch, (X, prompts) in enumerate(dataloader):
        B, _, _, _ = X.shape
        X = (X * 2 - 1).to(device)  # Normalize to [-1, 1]

        t = (torch.rand((B,)) * ACTUAL_STEPS).long().to(device)
        err, Y = noise.noise(X, t)

        # Get text embeddings
        text_embeds = get_text_embedding(prompts).to(device)
        pred = model(Y, t, text_embeds)

        loss = loss_fn(pred, err, t)
        tot_loss += float(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return tot_loss / iters

def run():
    # Initialize wandb
    wandb.init(project="pixart-diffusion", config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "architecture": "UNet"
    })

    print("Loading dataset...")
    dataset = dset.PixDataset(
        "./train/*.png",
        "./metadata.json"
    )
    print("Loaded dataset of size", len(dataset))
    train_loader = dutils.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = pixmodel.UNet().to(device)
    epoch = 1
    if args.load_path:
        epoch = util.load_model(model, args.load_path)

    SIMPLE_LOSS = True
    mse = nn.MSELoss()

    def loss_fn(ims, xs, ts):
        if SIMPLE_LOSS:
            return mse(ims, xs)
        else:
            loss = mse(ims, xs)
            alpha = 1 - noise.get_beta(ts)
            mul = noise.get_beta(ts)**2 / (2 * sample.get_alpha(ts) ** 2 * alpha * (1 - noise.alphat[ts]))
            return torch.sum(mul * loss)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    while True:
        print(f"Epoch {epoch}")
        print("LR =", scheduler.get_last_lr())
        loss = train(train_loader, model, loss_fn, optimizer)
        print("Average loss:", loss)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "loss": loss,
            "learning_rate": scheduler.get_last_lr()[0]
        })

        if args.print_on > 0 and epoch % args.print_on == 0:
            print("EPOCH", epoch)

        if epoch % args.save_on == 0:
            print("SAVING MODEL")
            save_path = f"{args.save_path}{epoch}.pt"
            util.save_model(model, epoch, save_path)
            
            # Generate and save image
            prompt = "diamond chestplate"
            output_image = f"{args.save_path}{epoch}_epochs.png"
            subprocess.run([
                "python", "sample.py", save_path, "16",
                "-prompt", prompt,
                "-o", output_image,
                "-noise_mul", "4"
            ])
            
            # Log the generated image to wandb
            wandb.log({
                "generated_image": wandb.Image(output_image, caption=f"Generated at epoch {epoch} with prompt {prompt}")
            })
        
        if scheduler.get_last_lr()[0] > 1e-5:
            scheduler.step()
        
        epoch += 1

if __name__ == "__main__":
    # Handle command line arguments
    import argparse

    parser = argparse.ArgumentParser("train.py")
    parser.add_argument("-load_path", help="Path to load the model from.", default="", nargs='?')
    parser.add_argument("-save_path", help="Path to save the model to.", default="", nargs='?')
    parser.add_argument("-save_on", help="The model is saved every 'save_on' epochs.", default=5, type=int)
    parser.add_argument("-print_on", help="Updates the loss graph every 'print_on' epochs.", default=25, type=int)

    args = parser.parse_args()
    assert args.save_on > 0

    if not args.save_path:
        args.save_path = args.load_path if args.load_path else "pix_model.pt"
        print("Saving to", args.save_path)

    run()