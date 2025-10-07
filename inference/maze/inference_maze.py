# We provide a simple fast inference script. You need to download the model to ./model and install diffsynth first.

import torch
from PIL import Image
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download

import sys
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
    ],
)


prompt="Create a 2D animation based on the provided image of a maze.\
    The blue star slides smoothly along the white path, stopping perfectly on the red flag and then acquiring a trophy.\
    The blue star never slides or crosses into the black segments of the maze.\
    The camera is a static, top-down view showing the entire maze.\
    Maze:\
    * The maze paths are white, the walls are black.\
    * The blue star depart from origin, represented by a green circle.\
    * The blue star slides smoothly along the white path.\
    * The blue star never slides or crosses into the black segments of the maze.\
    * The blue star stops perfectly on the red flag, acquiring a trophy thereafter.\
    Scene:\
    * No change in scene composition.\
    * No change in the layout of the maze.\
    * The blue star travels along the path without speeding up or slowing down.\
    Camera:\
    * Static camera.\
    * No zoom.\
    * No pan.\
    * No glitches, noise, or artifacts."

pipe.load_lora(pipe.dit, f"model/miniveo3_reasoner_maze.safetensors", alpha=1)

pipe.enable_vram_management()

import argparse
import os
import sys
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Inference')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('file', nargs='?', help='Input data file (PNG)')
    group.add_argument('-r', metavar='DIR', dest='dir', help='Process all PNG files in the directory')
    
    parser.add_argument('--quiet','-q', action='store_true', help='Suppress output messages')
    
    args = parser.parse_args()
    
    if args.file:
        if not args.file.endswith('.png'):
            print(f"Error: {args.file} is not a PNG file.")
            sys.exit(1)
        if not os.path.isfile(args.file):
            print(f"Error: {args.file} does not exist.")
            sys.exit(1)
        run(args.file)
    
    elif args.dir:
        if not os.path.isdir(args.dir):
            print(f"Error: {args.dir} does not exist.")
            sys.exit(1)
        
        data=[]
        for item in os.listdir(args.dir):
            if item.endswith('.png'):
                file_path = os.path.join(args.dir, item)
                if os.path.isfile(file_path):
                    data.append(file_path)
        
        data.sort()
        for item in tqdm(data, desc=f"Processing {args.dir}", unit="file", disable=args.quiet):
            run(item)

def run(file_path):
    input_image = Image.open(file_path)
    video = pipe(
        prompt=prompt,
        negative_prompt="",
        input_image=input_image,
        seed=0, tiled=True,
    )
    output_path = file_path[:-4] + "_inference.mp4"
    save_video(video, output_path, fps=15, quality=5)
    if not args.quiet:
        tqdm.write(f"Saved: {output_path}")
    

if __name__ == '__main__':
    main()