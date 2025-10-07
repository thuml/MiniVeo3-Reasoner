# Generate metadata.csv
# Each video corresponds to one line, including video filename, prompt, input image filename

from prompts import MAZE_PROMPT as prompt

with open("metadata.csv", "w") as f:
    f.write("video,prompt,input_image\n")

for grid_n in [3, 4, 5, 6]:
    for i in range(1, 1001):
        filename = f"maze{grid_n}_{i:04d}.mp4"
        framename = f"maze{grid_n}_{i:04d}_00.png"
        with open("./dataset/train/metadata.csv", "a") as f:
            f.write(f"{filename},\"{prompt}\",{framename}\n")
