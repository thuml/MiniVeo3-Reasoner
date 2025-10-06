# This is a Default settings.
# You can change the dir_rt and output_rt as you wish.
# Otherwise, just put this script in the same folder as inference_data/ and run it with an epoch name as the first argument.
# In that case, your epoch's output should be named as xxx_{epoch_name}_inference.mp4 with the generated answer .mp4 file.
# The output will be saved in eval_result/{epoch_name}/eval{sub}/, default checking all cases as ours.

input_rt="inference_data"
epoch=$1
if [ -z "$epoch" ]; then
    echo "Please provide an epoch name as the first argument."
    exit 1
fi
output_rt="eval_result/${epoch}"

for sub in "3by3" "4by4" "5by5" "6by6" "7by7" "8by8" "6by6_ood"; do
    dir="${input_rt}/${sub}/"
    output="${output_rt}/eval${sub}/"
    python evaluate/evaluator.py --dir $dir --output-dir $output --epoch-name $epoch --quiet
done
