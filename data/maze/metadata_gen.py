# Generate metadata.csv
# Each video corresponds to one line, including video filename, prompt, input image filename

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(root_dir)
from prompts import MAZE_PROMPT as prompt

with open("metadata.csv", "w") as f:
    f.write("video,prompt,input_image\n")

for grid_n in [3, 4, 5, 6]:
    for i in range(1, 1001):
        filename = f"maze{grid_n}_{i:04d}.mp4"
        framename = f"maze{grid_n}_{i:04d}_00.png"
        with open("dataset/maze_train/metadata.csv", "a") as f:
            f.write(f"{filename},\"{prompt}\",{framename}\n")
