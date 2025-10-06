<h1 align="center"> MiniVeo3-Reasoner: Thinking with Videos from Open-Source Priors </h1>

<!-- Overview -->



## 🔥 News

- 🚩 **2025.10**: We release MiniVeo3-Reasoner using mazes as testbed!

## 🤗 Models

| Models                    | Download Links                                               | Description                                                  |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| MiniVeo3-Reasoner-Maze-5B | 🤗 [HuggingFace](https://huggingface.co/thuml/MiniVeo3-Reasoner-Maze-5B) | Finetuned for [Maze](https://github.com/understanding-search/maze-dataset) (3x3 to 6x6 sizes) from [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) |

## ✨ Examples

### Maze

| Problem Setup | Examples |
| ------------- | -------- |
| Maze 3x3      | <video width="208" height="120" controls><source src="./example_videos/maze3_1004_epoch-2_inference.mp4" type="video/mp4"></video> <video width="208" height="120" controls><source src="./example_videos/maze3_1005_epoch-2_inference.mp4" type="video/mp4"></video> |
| Maze 4x4      | <video width="208" height="120" controls><source src="./example_videos/maze4_1007_epoch-2_inference.mp4" type="video/mp4"></video> <video width="208" height="120" controls><source src="./example_videos/maze4_1008_epoch-2_inference.mp4" type="video/mp4"></video> |
| Maze 5x5      | <video width="208" height="120" controls><source src="./example_videos/maze5_1006_epoch-2_inference.mp4" type="video/mp4"></video> <video width="208" height="120" controls><source src="./example_videos/maze5_1008_epoch-2_inference.mp4" type="video/mp4"></video> |
| Maze 6x6      | <video width="208" height="120" controls><source src="./example_videos/maze6_1001_epoch-2_inference.mp4" type="video/mp4"></video> <video width="208" height="120" controls><source src="./example_videos/maze6_1012_epoch-2_inference.mp4" type="video/mp4"></video> |

### OOD Generalization

OOD Solution Lengths: 

| Problem Setup | Examples |
| ------------- | -------- |
| Maze 6x6      | <video width="208" height="120" controls><source src="./example_videos/maze6ood_0004_epoch-2_inference.mp4" type="video/mp4"></video> <video width="208" height="120" controls><source src="./example_videos/maze6ood_0010_epoch-2_inference.mp4" type="video/mp4"></video> |

OOD Maze Sizes:

| Problem Setup | Examples |
| ------------- | -------- |
| Maze 7x7      | <video width="208" height="120" controls><source src="./example_videos/maze7_0011_epoch-2_inference.mp4" type="video/mp4"></video> <video width="208" height="120" controls><source src="./example_videos/maze7_0012_epoch-2_inference.mp4" type="video/mp4"></video> |
| Maze 8x8      | <video width="208" height="120" controls><source src="./example_videos/maze8_0024_epoch-2_inference.mp4" type="video/mp4"></video> <video width="208" height="120" controls><source src="./example_videos/maze8_0029_epoch-2_inference.mp4" type="video/mp4"></video> |

## 📊 Performance

| Maze                      | MiniVeo3-Reasoner-Maze-5B |
| ------------------------- | ------------------------- |
| 3x3                       | 100                       |
| 4x4                       | 100                       |
| 5x5                       | 100                       |
| 6x6                       | 98.4                      |
| 6x6 (OOD solution length) | 53.6                      |
| 7x7 (OOD size)            | 86.8                      |
| 8x8 (OOD size)            | 59.6                      |

### Comparisons

We include performance reported from [Visual Planning: Let's Think Only with Images](https://arxiv.org/abs/2505.11409)

| Model                            | Thinking Modality | Maze Overall |
| -------------------------------- | ----------------- | ------------ |
| Gemini 2.0 Flash - Direct        | Text              | 8.3          |
| Gemini 2.0 Flash - CoT           | Text              | 6.9          |
| Gemini 2.0 Pro (think)           | Text              | 21.5         |
| Qwen 2.5-VL-Instruct-3B - Direct | Text              | 0.5          |
| Qwen 2.5-VL-Instruct-3B - CoT    | Text              | 0.8          |
| Qwen 2.5-VL-Instruct-3B - SFT    | Text              | 33.3         |
| LVM-3B - VPFT                    | Image             | 59.0         |
| LVM-3B - VPRL                    | Image             | 74.5         |
| MiniVeo3-Reasoner-Maze-5B        | Video             | **99.6**     |

## 🚀 Get Started

### Environment Setup

```bash

```

### Data Preparation

Our data generator generates a series of mazes of certain size, path length and amount, outputting a .mp4 file and a .png file (the first frame of .mp4 file).

We use adapted maze-dataset to generate mazes. You can install it as follows:

```bash
cd maze-dataset-1.4.0
pip install -e .
cd ..
```

Thereafter, use `generate/maze_generator.py` to generate mazes with certain conditions. You can check the arguments in this Python file.

If you want to generate the same distribution dataset as ours, simply run `./generate_our_dataset.sh` and the output will be in `./generate/default_dataset`.

### Inference

Our inference is based on [Wan](https://github.com/Wan-Video/Wan2.2) model. By the time of writing this MD, the provided `inference5b.py` works for our inferencing. We will soon provide a generalized version.

You can check `./inference/inference5b.py`.

### Success Evaluation 

We use serveral metrics to evaluate the result.

Our evaluator compares the inferenced version and the answer version of one single output, then give the max distance of these two trajectories.

The results are divided into several categories based on the distances, and then determine whether it's correct, or it's imperfect.

We provide max-distance and PR (see [Visual Planning](https://github.com/yix8/VisualPlanning)) metrics output. You can add your own metrics at will.

If you follow the same dataset setting as ours, we provide you a shell script to easily evaluate.

```bash
./evaluate_script.sh
```

### Training Models

We train [Wan](https://github.com/Wan-Video/Wan2.2) model, which is well-instructed. You can easily fine-tune your own models.

For your convenience if you follow ours, we provide you a metadata generator, `./train/metadata_gen.py`, which generates the metadata for our training. You can easily read it and modify it. 

Notice that the prompt in `metadata_gen.py` is actually the prompt we use in our training.

## 🤝 Contributors

[Jialong Wu](https://manchery.github.io/)\*, [Tianhao Huang](https://github.com/MrH2T)\*, [Changjing He](https://github.com/hcjqwq)\*, [Mingsheng Long](https://ise.thss.tsinghua.edu.cn/~mlong/). (\* Equal Contribution)

We welcome contributions! Feel free to create [GitHub issues](https://github.com/thuml/MiniVeo3-Reasoner/issues) to track bugs and feature requests.

## 💡 Acknowledgements

- [Veo 3](https://video-zero-shot.github.io/): This project is inspired by the impressive zero-shot performance of Veo 3!
- [Wan](https://github.com/Wan-Video/Wan2.2): The strong open-sourced video diffusion models.
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/wanvideo): Video diffusion model training.
- [maze-dataset](https://github.com/understanding-search/maze-dataset): Dataset generation.
- [Visual Planning](https://github.com/yix8/VisualPlanning): Benchmark performance.

## 📜 Citation 

If you find MiniVeo3-Reasoner useful, we would appreciate it if you consider citing our work:

```
@misc{miniveo3reasoner,
    title = {MiniVeo3-Reasoner: Thinking with Videos from Open-Source Priors},
    author = {Jialong Wu, Tianhao Huang, Changjing He, Mingsheng Long},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/thuml/MiniVeo3-Reasoner}},
}
```