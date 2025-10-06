# 运行推理
# 可调参数：
# * NO_LORA: 是否 不 使用LoRA
#   - True: 不使用LoRA，直接使用基础模型进行推理
#   - False: 使用LoRA进行推理
# * epoch_name: LoRA的epoch名称
#   - 如果NO_LORA=True，则该参数无效
#   - 如果NO_LORA=False，则该参数指定使用哪个epoch的LoRA进行
# * grids: 需要测试的迷宫规模列表
#   - 如果不指定，则默认测试3-8的迷宫规模
#   - 如果指定，则只测试指定规模的迷宫，例如 python inference5b.py 3 只测试3x3的迷宫
# * i的范围：每个迷宫规模测试的迷宫数量
#   - 3-6的迷宫从1001到1250号迷宫
#   - 7-8的迷宫从1到250号迷宫
#   - 可根据需要修改
# ood测试为grid=6
# 推理结果保存在与输入图片相同的目录下，文件名格式为 maze{grid_n}_{i}_{epoch_name}_inference.mp4


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

import argparse

default_epoch_name="base"
default_inference_dir="3by3"
default_lowerbound=1
default_upperbound=250

parser = argparse.ArgumentParser(description='evaluator')
parser.add_argument('--dir', type=str, default=default_inference_dir, help='输入待测文件夹')
parser.add_argument('--epoch_name', type=str, default=default_epoch_name, help='epoch名称')
parser.add_argument('--lowerbound', '-l', type=int, default=default_lowerbound, help='数据下界')
parser.add_argument('--upperbound', '-r', type=int, default=default_upperbound, help='数据上界')

args = parser.parse_args()

test_dir = args.dir
epoch_name = args.epoch_name
lowerbound = args.lowerbound
upperbound = args.upperbound

if epoch_name != "base":
    pipe.load_lora(pipe.dit, f"models/train/Wan2.2-TI2V-5B_lora/{epoch_name}.safetensors", alpha=1)


pipe.enable_vram_management()

test6ood=False if test_dir != "inference_data/6by6_ood" else True

grid_n=int(test_dir.split("by")[0][-1])

ids=range(1000+lowerbound,1001+upperbound) if (grid_n != 7 and grid_n!=8 and not test6ood) else range(lowerbound,upperbound+1)

for i in ids:
    if not test6ood:
        file_name=f"maze{grid_n}_{i:04d}"
    else:
        file_name=f"maze6ood_{i:04d}"
    print(f"Processing {file_name}...")
    input_image = Image.open(f"{test_dir}/{file_name}_00.png")
    video = pipe(
        prompt=prompt,
        negative_prompt="",
        input_image=input_image,
        seed=0, tiled=True,
    )
    save_video(video, f"{test_dir}/{file_name}_{epoch_name}_inference.mp4", fps=15, quality=5)
