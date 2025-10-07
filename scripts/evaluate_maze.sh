# This is a Default settings.
# You can change the dir_rt and output_rt as you wish.
# Otherwise, just put this script in the same folder as inference_data/ and run it with an epoch name as the first argument.
# If you don't provide an epoch name (use our model), your results should be named as mazeN_IIII_inference.mp4
# If you provide an epoch name (use your own model), your results should be named as mazeN_IIII_{epochname}_inference.mp4

input_rt="dataset/maze_test"

epochname=$1

if [ -z "$epochname" ]; then
    echo "Take default parameter epochname=Empty"
    output_rt="eval_result/miniveo3_reasoner_maze"

    for sub in "maze3x3" "maze4x4" "maze5x5" "maze6x6" "maze7x7" "maze8x8" "maze6x6_ood"; do
        dir="${input_rt}/${sub}/"
        output="${output_rt}/eval_${sub}/"
        mkdir -p $output
        python evaluation/maze/evaluate.py --input-dir $dir --output-dir $output --quiet
    done
    exit 0
fi

output_rt="eval_result/${epochname}"

for sub in "maze3x3" "maze4x4" "maze5x5" "maze6x6" "maze7x7" "maze8x8" "maze6x6_ood"; do
    dir="${input_rt}/${sub}/"
    output="${output_rt}/eval_${sub}/"
    mkdir -p $output
    python evaluation/maze/evaluate.py --input-dir $dir --output-dir $output --epoch-name $epochname --quiet
done
