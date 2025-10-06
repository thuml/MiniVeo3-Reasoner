# 生成metadata.csv
# 每个视频对应一行，包含视频文件名、prompt、输入图像文件名



prompt="Create a 2D animation based on the provided image of a maze.\
    The blue star slides smoothly along the white path, stopping perfectly on the red flag and then acquiring a trophy.\
    The blue star never slides or crosses into the black segments of the maze.\
    The camera is a static, top-down view showing the entire maze.\
    Maze:\
    * The maze paths are white, the walls are black.\
    * The blue star depart from origin, represented by a green circle.\
    * The blue star slides smoothly along the white path.\
    * The blue star never slides or crosses into the black segments of the maze.\
    * The blue star stops perfectly on the red flag, acquiring a trophy thereafter.\
    Scene:\
    * No change in scene composition.\
    * No change in the layout of the maze.\
    * The blue star travels along the path without speeding up or slowing down.\
    Camera:\
    * Static camera.\
    * No zoom.\
    * No pan.\
    * No glitches, noise, or artifacts."

with open("metadata.csv","w") as f:
    f.write("video,prompt,input_image\n")

for grid_n in [3,4,5,6]:
    for i in range(1,1001):
        filename=f"maze{grid_n}_{i:04d}.mp4"
        framename=f"maze{grid_n}_{i:04d}_00.png"
        with open("data/example_video_dataset/metadata.csv","a") as f:
            f.write(f"{filename},\"{prompt}\",{framename}\n")
