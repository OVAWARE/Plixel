# FILE: dset.py

import torch
from torch.utils.data import Dataset
from glob import glob
import random
import json
from PIL import Image
import numpy as np
from parameters import NUM_CHANNELS  # Ensure NUM_CHANNELS is imported

from util import load_im, flip_hor, shift

# Performs a random flip and a random shift on a numpy array image
def transform(im):
    if im.shape[0] > 16 and im.shape[1] > 16:
        im = Image.fromarray(im).convert('RGBA')
        im = np.array(im)
    # Add a return statement
    return im  # Ensure to return the image

class PixDataset(Dataset):
    def __init__(self, data_path, json_path):
        """
        data_path: path to the directory containing the images
        json_path: path to the JSON file containing image labels
        """
        self.ims = []
        self.prompts = []
        
        print("Reading images and prompts... ", end="")
        
        # Load JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Create a dictionary mapping file names to prompts
        prompt_dict = {item['file_name']: item['prompt'] for item in data}
        
        for path in glob(data_path):
            im = load_im(path)
            self.ims.append(im)
            
            # Get the file name from the path
            file_name = path.split('/')[-1]
            
            # Get the corresponding prompt
            prompt = prompt_dict.get(file_name, "")
            self.prompts.append(prompt)
        
        print("Done")
    
    def __len__(self):
        return len(self.ims)
    
    def __getitem__(self, idx):
        im = transform(self.ims[idx])
        im = torch.Tensor(im)
        prompt = self.prompts[idx]
        return im, prompt