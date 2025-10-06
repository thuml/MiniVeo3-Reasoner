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
| Maze 3x3      |          |
| Maze 4x4      |          |
| Maze 5x5      |          |
| Maze 6x6      |          |

### OOD Generalization

OOD Solution Lengths: 

| Problem Setup | Examples |
| ------------- | -------- |
| Maze 6x6      |          |

OOD Maze Sizes:

| Problem Setup | Examples |
| ------------- | -------- |
| Maze 7x7      |          |
| Maze 8x8      |          |

## 📊 Performance

| Maze                      | MiniVeo3-Reasoner-Maze-5B |
| ------------------------- | ------------------------- |
| 3x3                       | 100                       |
| 4x4                       | 100                       |
| 5x5                       | 100                       |
| 6x6                       | 98.4                      |
| 6x6 (OOD solution length) | 58.3                      |
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

```bash

```

### Inference

```bash

```

### Success Evaluation 



### Training Models



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