# This inference script is for the Maze Reasoning dataset, automatically evaluating all sub-categories.

dir="dataset/maze_test"

for sub in "maze3x3" "maze4x4" "maze5x5" "maze6x6" "maze7x7" "maze8x8" "maze6x6_ood"; do
    python inference/maze/inference_maze.py -r "${dir}/${sub}/" --quiet
done