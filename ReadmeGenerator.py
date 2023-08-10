import os
from PIL import Image

# Path to the directory
path = "./Dataset/"
TotalCount = 0

# Open readme.md file for writing
with open("readme.md", "w") as file:

    # Write the header for the table
    file.write("| Mod | Data |\n")
    file.write("| --- | --- |\n")

    # Loop through all the directories in the main directory
    for dirpath, dirnames, filenames in os.walk(path):
        dirpath=dirpath.split("/")[-1]
        # Count the number of png files in each directory with dimensions of 16x16 pixels
        png_count = 0
        for filename in filenames:
            if filename.endswith(".png"):
                filepath = os.path.join(dirpath, filename)
                im = Image.open(path+filepath)
                if im.size == (16, 16):
                    png_count += 1
                    TotalCount += 1
        
        if(png_count>=1):
            file.write(f"| {dirpath} | {png_count} |\n")
    
    file.write(f"Total: {TotalCount}")

