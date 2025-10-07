# This is a Default settings.
# You can change the dir_rt and output_rt as you wish.
# Otherwise, just put this script in the same folder as inference_data/ and run it with an epoch name as the first argument.
# In that case, your epoch's output should be named as xxx_inference.mp4 with the generated answer .mp4 file.
# The output will be saved in eval_result/eval{sub}/, default checking all cases as ours.

input_rt="dataset/test"

epochname=$1

if [ -z "$epochname" ]; then
    echo "Please provide an epoch name as the first argument."
    exit 1
fi

output_rt="eval_result/${epochname}"

for sub in "maze3x3" "maze4x4" "maze5x5" "maze6x6" "maze7x7" "maze8x8" "maze6x6_ood"; do
    dir="${input_rt}/${sub}/"
    output="${output_rt}/eval_${sub}/"
    mkdir -p $output
    python evaluation/maze/evaluate.py --input-dir $dir --output-dir $output --epoch-name $epochname --quiet
done
