---
title: 'maze-dataset: Maze Generation with Algorithmic Variety and Representational Flexibility'
tags:
  - Python
  - machine learning
  - distributional shift
  - maze generation
  - datasets
authors:
  - name: Michael Igorevich Ivanitskiy
    orcid: 0000-0002-4213-4993
    affiliation: 1
    corresponding: true
  - name: Aaron Sandoval
    orcid: 0009-0002-8380-6140
    affiliation: 4
  - name: Alex F. Spies
    orcid: 0000-0002-8708-1530
    affiliation: 2
  - name: Tilman Räuker
    orcid: 0009-0009-6321-4413
    affiliation: 3
  - name: Brandon Knutson
    orcid: 0009-0004-8413-0239
    affiliation: 1
  - name: Cecilia Diniz Behn
    orcid: 0000-0002-8078-5105
    affiliation: 1
  - name: Samy Wu Fung
    orcid: 0000-0002-2926-4582
    affiliation: 1
affiliations:
  - name: Colorado School of Mines, Department of Applied Mathematics and Statistics
    index: 1
  - name: Imperial College London
    index: 2
  - name: UnSearch.org
    index: 3
  - name: Independent
    index: 4
date: 9 April 2025
bibliography: refs.bib
header-includes: \input{preamble.tex}
---

# Summary

Solving mazes is a classic problem in computer science and artificial intelligence, and humans have been constructing mazes for thousands of years. Although finding the shortest path through a maze is a solved problem, this very fact makes it an excellent testbed for studying how machine learning algorithms solve problems and represent spatial information. We introduce `maze-dataset`, a user-friendly Python library for generating, processing, and visualizing datasets of mazes. This library supports a variety of maze generation algorithms which can be configured with various parameters, and the resulting mazes can be filtered to satisfy desired properties. Also provided are tools for converting mazes to and from various formats suitable for a variety of neural network architectures, such as rasterized images, tokenized text sequences, and various visualizations. As well as providing a simple interface for generating, storing, and loading these datasets, `maze-dataset` is extensively tested, type hinted, benchmarked, and documented.

\input{figures/tex/fig1_diagram.tex}


# Statement of Need

While maze generation itself is straightforward, the architectural challenge comes from building a system supporting many algorithms with configurable parameters, property filtering, and representation transformation. This library aims to greatly streamline the process of generating and working with datasets of mazes that can be described as subgraphs of an $n \times n$ lattice with boolean connections and, optionally, start and end points that are nodes in the graph. Furthermore, we place emphasis on a wide variety of possible text output formats aimed at evaluating the spatial reasoning capabilities of Large Language Models (LLMs) and other text-based transformer models.

For interpretability and behavioral research, algorithmic tasks offer benefits by allowing systematic data generation and task decomposition, as well as simplifying the process of circuit discovery [@interpretability-survey]. Although mazes are well suited for these investigations, we found that existing maze generation packages [@cobbe2019procgen; @harriesMazeExplorerCustomisable3D2019; @gh_Ehsan_2022; @gh_Nemeth_2019; @easy_to_hard] lack support for transforming between multiple representations and provide limited control over the maze generation process.

## Related Works

A multitude of public and open-source software packages exist for generating mazes [@easy_to_hard; @gh_Ehsan_2022; @gh_Nemeth_2019]. However, nearly all of these packages produce mazes represented as rasterized images or other visual formats rather than the underlying graph structure, and this makes it difficult to work with these datasets.

- Most prior works provide mazes in visual or raster formats, and we provide a variety of similar output formats:
  - [`RasterizedMazeDataset`](https://understanding-search.github.io/maze-dataset/maze_dataset/dataset/rasterized.html#RasterizedMazeDataset), utilizing [`as_pixels()`](https://understanding-search.github.io/maze-dataset/maze_dataset.html#LatticeMaze.as_pixels), which can exactly mimic the outputs provided in `easy-to-hard-data` [@easy_to_hard] and can be configured to be similar to the outputs of @gh_Nemeth_2019
  - [`as_ascii()`](https://understanding-search.github.io/maze-dataset/maze_dataset.html#LatticeMaze.as_ascii) provides a format similar to [@eval-gpt-visual; @gh-oppenheimj2018maze]
  - [`MazePlot`](https://understanding-search.github.io/maze-dataset/maze_dataset/plotting.html#MazePlot) provides a feature‑rich plotting utility with support for multiple paths, heatmaps over positions, and more. This is similar to the outputs of [@mdl-suite; @mathematica-maze; @mazegenerator-net; @gh_Ehsan_2022]


- The text format provided by [`SolvedMaze(...).as_tokens()`](https://understanding-search.github.io/maze-dataset/maze_dataset.html#MazeDataset.as_tokens) is similar to that of [@eval-LLM-graphs] but with many more options, detailed in \hyperref[sec:tokenized-output-formats]{section: \textit{\nameref{sec:tokenized-output-formats}}}.

- Preserving metadata about the generation algorithm with the dataset itself is essential for studying the effects of distributional shifts. Our package efficiently stores the dataset along with its metadata in a single human-readable file [@zanj]. As far as we are aware, no existing packages do this reliably.

- Storing mazes as images or adjacency matrices is not only difficult to work with, but also inefficient. We use a highly efficient method detailed in \hyperref[sec:implementation]{section: \textit{\nameref{sec:implementation}}}.

- Our package is easily installable with source code freely available. It is extensively tested, type hinted, benchmarked, and documented. Many other maze generation packages lack this level of rigor and scope, and some [@ayaz2008maze] appear to simply no longer be accessible.



\newpage

# Features

We direct readers to our [examples](https://understanding-search.github.io/maze-dataset/examples/maze_examples.html), [docs](https://understanding-search.github.io/maze-dataset/maze_dataset.html), and [notebooks](https://understanding-search.github.io/maze-dataset/notebooks/) for more information. Our package can be installed from [PyPi](https://pypi.org/project/maze-dataset/) via `pip install maze-dataset`, or directly from the [git repository](https://github.com/understanding-search/maze-dataset) [@maze-dataset-github].

Datasets of mazes are created from a [`MazeDatasetConfig`](https://understanding-search.github.io/maze-dataset/maze_dataset.html#MazeDatasetConfig) configuration object, which allows specifying the number of mazes, their size, the generation algorithm, and various parameters for the generation algorithm. Datasets can also be filtered after generation to satisfy certain properties. Custom filters can be specified, and some filters are included in [`MazeDatasetFilters`](https://understanding-search.github.io/maze-dataset/maze_dataset/dataset/filters.html#MazeDatasetFilters).


## Visual Output Formats {#visual-output-formats}

Internally, mazes are [`SolvedMaze`](https://understanding-search.github.io/maze-dataset/maze_dataset.html#SolvedMaze) objects, which have path information and a tensor optimized for storing sub-graphs of a lattice. These objects can be converted to and from several formats, shown in \autoref{fig:output-fmts}, to maximize their utility in different contexts.

In previous work, maze tasks have been used with Recurrent Convolutional Neural Network (RCNN) derived architectures [@deepthinking]. To facilitate the use of our package in this context, we replicate the format of [@easy_to_hard] and provide the [`RasterizedMazeDataset`](https://understanding-search.github.io/maze-dataset/maze_dataset/dataset/rasterized.html#RasterizedMazeDataset) class which returns rasterized pairs of (input, target) mazes as shown in \autoref{fig:e2h-raster}.

\input{figures/tex/fig2_formats.tex}

\input{figures/tex/fig3_raster.tex}






\newpage

## Tokenized Output Formats {#sec:tokenized-output-formats}

Autoregressive transformer models can be quite sensitive to the exact format of input data, and may even use delimiter tokens to perform reasoning steps [@pfau2024dotbydot; @spies2024causalworldmodels]. To facilitate systematic investigation of the effects of different representations of data on text model performance, we provide a variety of text output formats, with an example given in \autoref{fig:token-regions}. We utilize Finite State Transducers [@Gallant2015Transducers] for efficiently storing valid tokenizers.

\input{figures/tex/fig4_tokenfmt.tex}

## Benchmarks {#benchmarks}

We benchmarks for generation time across various configurations in \autoref{tab:benchmarks} and \autoref{fig:benchmarks}. Experiments were performed on a [standard GitHub runner](https://docs.github.com/en/actions/using-github-hosted-runners/using-github-hosted-runners/about-github-hosted-runners#standard-github-hosted-runners-for-public-repositories) without parallelism. Additionally, maze generation under certain constraints may not always be successful, and for this we provide a way to estimate the success rate of a given configuration, described in \autoref{fig:sre}.



\input{figures/tex/tab1_benchmarks.tex}

\input{figures/tex/fig6_benchmarks.tex}

\input{figures/tex/fig7_sre.tex}


\newpage

# Implementation {#sec:implementation}

Using an adjacency matrix for storing mazes would be memory inefficient by failing to exploit the highly sparse structure, while using an adjacency list could lead to a poor lookup time. This package utilizes a simple, efficient representation of mazes as subgraphs of a finite lattice, detailed in \autoref{fig:maze-impl}, which we call a [`LatticeMaze`](https://understanding-search.github.io/maze-dataset/maze_dataset.html#LatticeMaze).

\input{figures/tex/fig8_impl.tex}

Our package is implemented in Python[@python], and makes use of the extensive scientific computing ecosystem, including NumPy [@numpy] for array manipulation, plotting tools [@matplotlib; @seaborn], Jupyter notebooks [@jupyter], and PySR [@pysr] for symbolic regression.






# Usage in Research

This package was originally built for the needs of the [@maze-transformer-github] project, which aims to investigate spatial planning and world models in autoregressive transformer models trained on mazes [@ivanitskiy2023structuredworldreps; @spies2024causalworldmodels; @maze-dataset-arxiv-2023]. It was extended for work on understanding the mechanisms by which recurrent convolutional and implicit networks [@fung2022jfb] solve mazes given a rasterized view [@knutson2024logicalextrapolation], which required matching the pixel-padded and endpoint constrained output format of [@easy_to_hard]. Ongoing work using `maze-dataset` aims to investigate the effects of varying the tokenization format on the performance of pretrained LLMs on spatial reasoning.

At the time of writing, this software package has been actively used in work by other groups:

- By [@nolte2024multistep] to compare the effectiveness of transformers trained with the MLM-$\mathcal{U}$ [@MLMU-kitouni2024factorization] multistep prediction objective against standard autoregressive training for multi-step planning on our maze task.

- By [@wang2024imperative] and [@chen2024iaimperative] to study imperative learning.

- By [@zhang2025tscend] to introduce a novel framework for reasoning diffusion models.

- By [@dao2025alphamaze] to improve spatial reasoning in LLMs with GRPO.

- By [@cai2025morse] to create a multimodal reasoning benchmark, via mazes in videos.

- By [@xu2025visual] to study visual planning in LLMs.

- By [@lee2025adaptive] to evaluate adaptive inference-time scaling with diffusion models on maze navigation tasks.

- By [@zhang2025vfscale] to test verifier-free diffusion models.

\newpage

# Acknowledgements

\input{figures/tex/acknowledgements.tex}


# References
