# Split train and test datasets

# Create train dataset at ./dataset/train
for ((i=3; i<=6; i++)); do
    source_dir="./dataset/maze${i}x${i}"
    target_dir="./dataset/train"

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

echo "Train dataset created at ./dataset/train"

# Create test dataset at ./dataset/test
mkdir -p ./dataset/test
mv ./dataset/{maze3x3,maze4x4,maze5x5,maze6x6,maze7x7,maze8x8,maze6x6_ood} ./dataset/test/

echo "Test dataset created at ./dataset/test"