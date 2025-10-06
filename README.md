<p align="center"><img src="assets/miniveo3-reasoner-logo-pure.png" width="200px" alt="MiniVeo3-Reasoner icon" /></p>
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

<table style="width: 100%; text-align: center; margin-top: 20px;">
    <tr>
        <td> <b>Problem Setup</b></td>
        <td colspan=2> <b>Examples</b></td>
    </tr>
    <tr>
        <td>Maze 3x3</td>
      <td >
          <video src="https://github.com/user-attachments/assets/661a4248-bb95-4625-89a9-43d4d67d0b13" width="100%" controls autoplay loop></video>
      </td>
        <td>
          <video src="https://github.com/user-attachments/assets/5eb477c3-00f2-41f8-952b-c6f8f04b4c5f" width="100%" controls autoplay loop></video>
      </td>
  </tr>
   <tr>
        <td>Maze 4x4</td>
      <td >
          <video src="https://github.com/user-attachments/assets/cb324702-a027-431e-808a-4474be77994c" width="100%" controls autoplay loop></video>
      </td>
        <td>
          <video src="https://github.com/user-attachments/assets/0461f438-b88a-4fd4-bff8-b3ddb1394d3b" width="100%" controls autoplay loop></video>
      </td>
  </tr>
     <tr>
        <td>Maze 5x5</td>
      <td >
          <video src="https://github.com/user-attachments/assets/0f650f24-744f-4250-856d-90d127e381bd" width="100%" controls autoplay loop></video>
      </td>
        <td>
          <video src="https://github.com/user-attachments/assets/e7958084-d327-4a86-978b-5430195d5e7d" width="100%" controls autoplay loop></video>
      </td>
  </tr>
    <tr>
        <td>Maze 6x6</td>
      <td >
          <video src="https://github.com/user-attachments/assets/bff875d9-5c1b-491b-a3db-07d6e433bc5c" width="100%" controls autoplay loop></video>
      </td>
        <td>
          <video src="https://github.com/user-attachments/assets/02bcc87b-11cc-4bcb-958a-b6a4573f459a" width="100%" controls autoplay loop></video>
      </td>
  </tr>
</table>


### OOD Generalization

OOD Solution Lengths: 

<table style="width: 100%; text-align: center; margin-top: 20px;">
    <tr>
        <td> <b>Problem Setup</b></td>
        <td colspan=2> <b>Examples</b></td>
    </tr>
    <tr>
        <td>Maze 6x6</td>
      <td >
          <video src="https://github.com/user-attachments/assets/042d4f68-b75d-47c4-b114-0afa6f2d696b" width="100%" controls autoplay loop></video>
      </td>
        <td>
          <video src="https://github.com/user-attachments/assets/29509f81-e36d-4db5-9ba3-c658410d99b9" width="100%" controls autoplay loop></video>
      </td>
  </tr>
</table>

OOD Maze Sizes:

<table style="width: 100%; text-align: center; margin-top: 20px;">
    <tr>
        <td> <b>Problem Setup</b></td>
        <td colspan=2> <b>Examples</b></td>
    </tr>
    <tr>
        <td>Maze 7x7</td>
      <td >
          <video src="https://github.com/user-attachments/assets/2aa97273-04ac-4c76-a2cf-92e96286dc5e" width="100%" controls autoplay loop></video>
      </td>
        <td>
          <video src="https://github.com/user-attachments/assets/20c3e8d8-5343-41a8-bfac-d64c1d899d7d" width="100%" controls autoplay loop></video>
      </td>
  </tr>
   <tr>
        <td>Maze 8x8</td>
      <td >
          <video src="https://github.com/user-attachments/assets/a327336a-735b-4821-807e-cb0fe9413a4d" width="100%" controls autoplay loop></video>
      </td>
        <td>
          <video src="https://github.com/user-attachments/assets/46d350f9-a19f-4d4a-95a6-74fea5ad7e3e" width="100%" controls autoplay loop></video>
      </td>
  </tr>
</table>


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
- [Nano Banana](https://aistudio.google.com/models/gemini-2-5-flash-image): Help generate the project logo.

## 📜 Citation 

The technical report is ongoing. If you find MiniVeo3-Reasoner useful, we would appreciate it if you consider citing our work:

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
