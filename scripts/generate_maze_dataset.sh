# This generate our maze dataset.
# There are 1250 3x3, 4x4, 5x5, 6x6 mazes with path length from 2 to 2*n,
#   and 250 7x7, 8x8 mazes with path length from 2 to 2*n
#   and 250 6x6 mazes with path length from 13 to 18.
# File names are in the format of maze{grid_n}_{i:04d}.mp4 and maze{grid_n}_{i:04d}_00.png

# The dataset will be saved in ./dataset/
# Make sure maze-dataset of our customized version is installed.

python data/maze/maze_generator.py --grid-n 3 --n-mazes 1250 --min-path-length 2 --max-path-length 6 --output-dir ./dataset/maze3x3 --file-prefix "maze3" -q
python data/maze/maze_generator.py --grid-n 4 --n-mazes 1250 --min-path-length 2 --max-path-length 8 --output-dir ./dataset/maze4x4 --file-prefix "maze4" -q
python data/maze/maze_generator.py --grid-n 5 --n-mazes 1250 --min-path-length 2 --max-path-length 10 --output-dir ./dataset/maze5x5 --file-prefix "maze5" -q 
python data/maze/maze_generator.py --grid-n 6 --n-mazes 1250 --min-path-length 2 --max-path-length 12 --output-dir ./dataset/maze6x6 --file-prefix "maze6" -q
python data/maze/maze_generator.py --grid-n 7 --n-mazes 250 --min-path-length 2 --max-path-length 14 --output-dir ./dataset/maze7x7 --file-prefix "maze7" -q
python data/maze/maze_generator.py --grid-n 8 --n-mazes 250 --min-path-length 2 --max-path-length 16 --output-dir ./dataset/maze8x8 --file-prefix "maze8" -q
python data/maze/maze_generator.py --grid-n 6 --n-mazes 250 --min-path-length 13 --max-path-length 18 --output-dir ./dataset/maze6x6_ood --file-prefix "maze6ood" -q

# Split train and test datasets

# Create train dataset at dataset/maze_train
for ((i=3; i<=6; i++)); do
    source_dir="dataset/maze${i}x${i}"
    target_dir="dataset/maze_train"

    mkdir -p "$target_dir"

    for ((x=1; x<=1000; x++)); do
        num=$(printf "%04d" $x)

        mp4_file="maze${i}_${num}.mp4"

        if [ -f "$source_dir/$mp4_file" ]; then
            mv "$source_dir/$mp4_file" "$target_dir/"
        fi
        
        png_file="maze${i}_${num}_00.png"
        if [ -f "$source_dir/$png_file" ]; then
            mv "$source_dir/$png_file" "$target_dir/"
        fi
    done
done

python data/maze/metadata_gen.py

echo "Train dataset created at dataset/maze_train"

# Create test dataset at dataset/test
mkdir -p dataset/maze_test
mv dataset/{maze3x3,maze4x4,maze5x5,maze6x6,maze7x7,maze8x8,maze6x6_ood} dataset/maze_test

echo "Test dataset created at dataset/maze_test"