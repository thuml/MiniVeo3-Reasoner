# This generate our maze dataset.
# There are 1250 3x3, 4x4, 5x5, 6x6 mazes with path length from 2 to 2*n,
#   and 250 7x7, 8x8 mazes with path length from 2 to 2*n
#   and 250 6x6 mazes with path length from 13 to 18.
# File names are in the format of maze{grid_n}_{i:04d}.mp4 and maze{grid_n}_{i:04d}_00.png

# The dataset will be saved in ./dataset/
# Make sure maze-dataset of our customized version is installed.

python data/maze/maze_generator.py --grid-n 3 --n-mazes 1250 --min-path-length 2 --max-path-length 6 --output-dir ./dataset/maze3x3 -q
python data/maze/maze_generator.py --grid-n 4 --n-mazes 1250 --min-path-length 2 --max-path-length 8 --output-dir ./dataset/maze4x4 -q
python data/maze/maze_generator.py --grid-n 5 --n-mazes 1250 --min-path-length 2 --max-path-length 10 --output-dir ./dataset/maze5x5 -q 
python data/maze/maze_generator.py --grid-n 6 --n-mazes 1250 --min-path-length 2 --max-path-length 12 --output-dir ./dataset/maze6x6 -q
python data/maze/maze_generator.py --grid-n 7 --n-mazes 250 --min-path-length 2 --max-path-length 14 --output-dir ./dataset/maze7x7 -q
python data/maze/maze_generator.py --grid-n 8 --n-mazes 250 --min-path-length 2 --max-path-length 16 --output-dir ./dataset/maze8x8 -q
python data/maze/maze_generator.py --grid-n 6 --n-mazes 250 --min-path-length 13 --max-path-length 18 --output-dir ./dataset/maze6x6_ood -q