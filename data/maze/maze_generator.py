# Generate mp4 and first frame png for mazes.
# Parameters: --output-dir, --grid-n, --n-mazes, --min-path-length,
# --max-path-length, --create-algo, --quiet.
# Output files are saved in the specified output directory.
# The distribution of maze lengths is uniform from min-path-length to
# max-path-length, cycling through the lengths.

import argparse
import os
import random

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from maze_dataset.generation import get_maze_with_solution
from maze_dataset.plotting import MazePlot


DEFAULT_OUTPUT_DIR = "dataset/maze3x3"
DEFAULT_MAZE_SIZE = 3
DEFAULT_NUM_MAZES = 10
DEFAULT_MAZE_CTOR = "gen_kruskal"
DEFAULT_FIG_SIZE_BY_GRID = {4: 128, 8: 256}
DEFAULT_FPS = 15
DEFAULT_FRAME_COUNT = 16


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate mazes and export paired mp4/png samples."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--grid-n",
        "-s",
        type=int,
        default=DEFAULT_MAZE_SIZE,
        help="Maze size as grid_n by grid_n.",
    )
    parser.add_argument(
        "--n-mazes",
        "-n",
        type=int,
        default=DEFAULT_NUM_MAZES,
        help="Number of mazes to generate.",
    )
    parser.add_argument(
        "--min-path-length",
        "-l",
        type=int,
        help="Minimum path length. Default: 2.",
    )
    parser.add_argument(
        "--max-path-length",
        "-r",
        type=int,
        help="Maximum path length. Default: 2 * grid_n.",
    )
    parser.add_argument(
        "--file-prefix",
        "-p",
        type=str,
        default="maze",
        help="Output file prefix.",
    )
    parser.add_argument(
        "--create-algo",
        "-c",
        type=str,
        default=DEFAULT_MAZE_CTOR,
        help="Maze generation algorithm.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Quiet mode.",
    )
    parser.add_argument(
        "--const-speed",
        action="store_true",
        help="Use constant-speed animation export.",
    )
    parser.add_argument(
        "--fig-size-pixel",
        type=int,
        default=None,
        help="Output image/video size in pixels. Defaults by grid size.",
    )
    return parser


def resolve_config(args):
    min_path_length = args.min_path_length if args.min_path_length else 2
    max_path_length = args.max_path_length if args.max_path_length else 2 * args.grid_n
    file_prefix = args.file_prefix if args.file_prefix else f"maze{args.grid_n}"
    fig_size_pixel = (
        args.fig_size_pixel
        if args.fig_size_pixel
        else DEFAULT_FIG_SIZE_BY_GRID.get(args.grid_n, 256)
    )

    if max_path_length > args.grid_n * args.grid_n:
        raise ValueError(
            f"max_path_length must be <= grid_n*grid_n={args.grid_n * args.grid_n}"
        )
    if min_path_length < 2 or min_path_length >= max_path_length:
        raise ValueError(
            f"min_path_length must be >=2 and < max_path_length={max_path_length}"
        )

    return {
        "output_dir": args.output_dir,
        "grid_n": args.grid_n,
        "n_mazes": args.n_mazes,
        "min_path_length": min_path_length,
        "max_path_length": max_path_length,
        "file_prefix": file_prefix,
        "fig_size_pixel": (fig_size_pixel, fig_size_pixel),
        "maze_ctor": args.create_algo,
    }


def build_target_lengths(n_mazes, min_path_length, max_path_length):
    span = max_path_length - min_path_length + 1
    return [(idx % span) + min_path_length for idx in range(n_mazes)]


def generate_dataset(config):
    target_lengths = build_target_lengths(
        config["n_mazes"],
        config["min_path_length"],
        config["max_path_length"],
    )
    required_counts = {
        length: target_lengths.count(length) for length in set(target_lengths)
    }
    mazes_by_length = {length: [] for length in required_counts}

    random.seed(0)
    np.random.seed(0)

    while sum(len(mazes) for mazes in mazes_by_length.values()) < config["n_mazes"]:
        maze = get_maze_with_solution(
            config["maze_ctor"], (config["grid_n"], config["grid_n"])
        )
        path_length = len(maze.solution)
        if (
            path_length < config["min_path_length"]
            or path_length > config["max_path_length"]
            or path_length not in mazes_by_length
        ):
            continue
        if len(mazes_by_length[path_length]) < required_counts[path_length]:
            mazes_by_length[path_length].append(maze)

    counts = {length: 0 for length in required_counts}
    dataset = []
    for target_length in target_lengths:
        dataset.append(mazes_by_length[target_length][counts[target_length]])
        counts[target_length] += 1
    return dataset


def render_frames(maze, const_speed, fig_size_pixel):
    plotter = MazePlot(maze)
    if const_speed:
        return plotter.plot_continuous_const_speed(
            fig_size_pixel=fig_size_pixel,
            plain=True,
        )
    return plotter.plot_continuous(
        fig_size_pixel=fig_size_pixel,
        frames_num=DEFAULT_FRAME_COUNT,
        plain=True,
    )


def fig_to_rgb(fig):
    fig.canvas.draw()
    return np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]


def save_png(rgb_image, output_path):
    image_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image_bgr)


def save_video(frames, output_path, fps):
    images = []
    for fig in frames:
        images.append(fig_to_rgb(fig))
        plt.close(fig)
    imageio.mimsave(output_path, images, fps=fps)


def export_sample(maze, sample_name, config, quiet, const_speed):
    frames = render_frames(maze, const_speed, config["fig_size_pixel"])
    if not quiet:
        print(f"Exporting {sample_name} ...")

    first_frame = fig_to_rgb(frames[0])
    save_png(
        first_frame,
        os.path.join(config["output_dir"], f"{sample_name}_00.png"),
    )
    save_video(
        frames,
        os.path.join(config["output_dir"], f"{sample_name}.mp4"),
        fps=DEFAULT_FPS,
    )


def iterate_dataset(dataset, grid_n, quiet):
    if quiet:
        from tqdm import tqdm

        return enumerate(tqdm(dataset, desc=f"Generating mazes {grid_n}x{grid_n}"))
    return enumerate(dataset)


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = resolve_config(args)
    except ValueError as exc:
        print(exc)
        raise SystemExit(1)

    os.makedirs(config["output_dir"], exist_ok=True)
    dataset = generate_dataset(config)

    if not args.quiet:
        print(
            f"Making {config['n_mazes']} {config['grid_n']} by {config['grid_n']} "
            f"mazes to {config['output_dir']}/ ..."
        )

    for index, maze in iterate_dataset(dataset, config["grid_n"], args.quiet):
        sample_name = f"{config['file_prefix']}_{index + 1:04d}"
        export_sample(maze, sample_name, config, args.quiet, args.const_speed)


if __name__ == "__main__":
    main()
