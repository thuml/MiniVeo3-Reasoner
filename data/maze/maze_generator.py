# Generate mp4 and first frame png for mazes
# Parameters: --output-dir, --maze-size, --n-mazes, --min-path-length, --max-path-length, --create-algo, --quiet
# Output files are saved in the specified output directory
# The distribution of maze lengths is uniform from min-path-length to max-path-length, cycling through the lengths
# The generated png is in RGB format, dpi 96; the generated video is 832*480*81, fps 15


from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators, get_maze_with_solution

import sys
import os
import argparse

import matplotlib.pyplot as plt

from maze_dataset.plotting import MazePlot
import numpy as np
import imageio
import cv2
import random

default_output_dir = "dataset/maze3x3"
default_maze_size = 3
default_n_mazes = 10
default_maze_ctor = 'gen_kruskal'  # generation algorithm, kruskal is good

parser = argparse.ArgumentParser(description="Generate mazes and export to mp4 and png")
parser.add_argument("--output-dir", "-o", type=str, default=default_output_dir,
                    help="Directory to save output files")
parser.add_argument("--grid-n", "-s", type=int, default=default_maze_size,
                    help="Size of the maze (grid_n by grid_n)")
parser.add_argument("--n-mazes", "-n", type=int, default=default_n_mazes,
                    help="Number of mazes to generate, default=10")
parser.add_argument("--min-path-length", "-l", type=int, help="(Optional) Minimum path length")
parser.add_argument("--max-path-length", "-r", type=int, help="(Optional) Maximum path length")
parser.add_argument("--file-prefix", "-p", type=str, default="maze", help="(Optional) File prefix for output files")
parser.add_argument("--create-algo", "-c", type=str, default=default_maze_ctor,
                    help="(Optional) Maze generation algorithm, default='gen_kruskal'")
parser.add_argument("--quiet", "-q", help="Quiet mode, no output", action="store_true")
parser.add_argument("--const-speed", help="Use constant speed for animation", action="store_true")
parser.add_argument("--fig-size-pixel", type=int, default=None,
                    help="(Optional) Output image/video size in pixels; defaults by grid size (4x4->128, 8x8->256)")


args = parser.parse_args()

output_dir = args.output_dir
grid_n = args.grid_n
n_mazes = args.n_mazes
min_path_length = args.min_path_length if args.min_path_length else 2
max_path_length = args.max_path_length if args.max_path_length else 2 * grid_n
file_prefix = args.file_prefix if args.file_prefix else f"maze{grid_n}"

default_fig_size_by_grid = {4: 128, 8: 256}
fig_size_pixel = args.fig_size_pixel if args.fig_size_pixel else default_fig_size_by_grid.get(grid_n, 256)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


if max_path_length > grid_n * grid_n:
    print(f"max_path_length must be <= grid_n*grid_n={grid_n * grid_n}")
    exit(0)
if min_path_length < 2 or min_path_length >= max_path_length:
    print(f"min_path_length must be >=2 and < max_path_length={max_path_length}")
    exit(0)


maze_ctor = args.create_algo

target_lengths = [(i % (max_path_length - min_path_length + 1)) + min_path_length for i in range(n_mazes)]
need_lengths = {length: target_lengths.count(length) for length in set(target_lengths)}
mazes_with_length = [list() for _ in range(max_path_length + 1)]

mazes_got = 0

random.seed(0)
np.random.seed(0)

while mazes_got < n_mazes:
    newmaze = get_maze_with_solution(maze_ctor, (grid_n, grid_n))
    now_length = len(newmaze.solution)
    # Skip samples outside the requested path-length range.
    if now_length < min_path_length or now_length > max_path_length:
        continue
    if len(mazes_with_length[now_length]) < need_lengths[now_length]:
        mazes_with_length[now_length].append(newmaze)
        mazes_got += 1

dataset = []

count_lengths = [0 for i in range(max_path_length + 1)]
for i in range(n_mazes):
    target = target_lengths[i]
    dataset.append(mazes_with_length[target][count_lengths[target]])
    count_lengths[target] += 1



def export_mp4_const_speed(m, output_file, fps=15, fig_size_pixel=(256,256)):
    frames = MazePlot(m).plot_continuous_const_speed(fig_size_pixel=fig_size_pixel, plain=True)

    if not args.quiet:
        print(f"Exporting {output_file} ...")

    frames[0].canvas.draw()
    img = np.array(frames[0].canvas.renderer.buffer_rgba())[:, :, :3]  # take RGB
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'{output_dir}/{output_file}_00.png', img_bgr)

    plt.close(frames[0])

    def create_video(frames, output_file='output.mp4', fps=15):
        images = []
        for fig in frames:
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]  # take RGB
            images.append(img)
            plt.close(fig)

        imageio.mimsave(output_file, images, fps=fps)

    create_video(frames, f'{output_dir}/{output_file}.mp4', fps=fps)

def export_mp4(m, output_file, fps=15, fig_size_pixel=(256,256)):
    frames = MazePlot(m).plot_continuous(fig_size_pixel=fig_size_pixel, frames_num=16, plain=True)

    if not args.quiet:
        print(f"Exporting {output_file} ...")

    frames[0].canvas.draw()
    img = np.array(frames[0].canvas.renderer.buffer_rgba())[:, :, :3]  # take RGB
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'{output_dir}/{output_file}_00.png', img_bgr)

    plt.close(frames[0])

    def create_video(frames, output_file='output.mp4', fps=15):
        images = []
        for fig in frames:
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]  # take RGB
            images.append(img)
            plt.close(fig)

        imageio.mimsave(output_file, images, fps=fps)

    create_video(frames, f'{output_dir}/{output_file}.mp4', fps=fps)


if not args.quiet:
    print(f"Making {n_mazes} {grid_n} by {grid_n} mazes to {output_dir}/ ...")
    for i, maze in enumerate(dataset):
        if args.const_speed:
            export_mp4_const_speed(maze, f'{file_prefix}_{i + 1:04d}', fps=15, fig_size_pixel=(fig_size_pixel, fig_size_pixel))
        else:
            export_mp4(maze, f'{file_prefix}_{i + 1:04d}', fps=15, fig_size_pixel=(fig_size_pixel, fig_size_pixel))
else:
    from tqdm import tqdm
    for i, maze in enumerate(tqdm(dataset, desc=f"Generating mazes {grid_n}x{grid_n}")):
        if args.const_speed:
            export_mp4_const_speed(maze, f'{file_prefix}_{i + 1:04d}', fps=15, fig_size_pixel=(fig_size_pixel, fig_size_pixel))
        else:
            export_mp4(maze, f'{file_prefix}_{i + 1:04d}', fps=15, fig_size_pixel=(fig_size_pixel, fig_size_pixel))
