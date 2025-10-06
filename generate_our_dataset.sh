# This generate the same dataset as ours.
# That is 1250 3x3, 4x4, 5x5, 6x6 mazes with path length from 2 to 2*n,
#   and 250 7x7, 8x8 mazes with path length from 2 to 2*n
#   and 250 6x6 mazes with path length from 13 to 18.
# File names are in the format of maze{grid_n}_{i:04d}.mp4 and maze{grid_n}_{i:04d}_00.png

# If you think it's too slow, you can generate them in parallel.
# The dataset will be saved in ./default_dataset/
# Make sure maze_generator.py is in the same folder as this script, and maze-dataset of given version is installed.

cd ./generate

python maze_generator.py --grid-n 3 --n-mazes 1250 --min-path-length 2 --max-path-length 6 --output-dir ./default_dataset/3by3 -q
python maze_generator.py --grid-n 4 --n-mazes 1250 --min-path-length 2 --max-path-length 8 --output-dir ./default_dataset/4by4 -q
python maze_generator.py --grid-n 5 --n-mazes 1250 --min-path-length 2 --max-path-length 10 --output-dir ./default_dataset/5by5 -q 
python maze_generator.py --grid-n 6 --n-mazes 1250 --min-path-length 2 --max-path-length 12 --output-dir ./default_dataset/6by6 -q
python maze_generator.py --grid-n 7 --n-mazes 250 --min-path-length 2 --max-path-length 14 --output-dir ./default_dataset/7by7 -q
python maze_generator.py --grid-n 8 --n-mazes 250 --min-path-length 2 --max-path-length 16 --output-dir ./default_dataset/8by8 -q
python maze_generator.py --grid-n 6 --n-mazes 250 --min-path-length 13 --max-path-length 18 --output-dir ./default_dataset/6by6_ood -q

cd ..