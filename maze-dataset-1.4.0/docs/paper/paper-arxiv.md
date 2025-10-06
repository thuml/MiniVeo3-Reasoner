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
header-includes: |
  \usepackage{graphicx}
  \usepackage{tikz}
  \usetikzlibrary{calc}
  \tikzset{ % Define a TikZ style for an external hyperlink node.
    hyperlink node url/.style={
      alias=sourcenode,
      append after command={
        let \p1 = (sourcenode.north west),
            \p2 = (sourcenode.south east),
            \n1 = {\x2-\x1},
            \n2 = {\y1-\y2} in
        node[inner sep=0pt, outer sep=0pt, anchor=north west, at=(\p1)]
            {\href{#1}{\XeTeXLinkBox{\phantom{\rule{\n1}{\n2}}}}}
      }
    }
  }
  \providecommand{\XeTeXLinkBox}[1]{#1}
  \newcommand{\docslink}[2]{\href{https://understanding-search.github.io/maze-dataset/#1}{#2}}
  \newcommand{\docslinkcode}[2]{\href{https://understanding-search.github.io/maze-dataset/#1}{\texttt{#2}}}
  \newcommand{\secref}[1]{\hyperref[#1]{section: \textit{\nameref{#1}}}}
---

# Summary

Solving mazes is a classic problem in computer science and artificial intelligence, and humans have been constructing mazes for thousands of years. Although finding the shortest path through a maze is a solved problem, this very fact makes it an excellent testbed for studying how machine learning algorithms solve problems and represent spatial information. We introduce `maze-dataset`, a user-friendly Python library for generating, processing, and visualizing datasets of mazes. This library supports a variety of maze generation algorithms providing mazes with or without loops, mazes that are connected or not, and many other variations. These generation algorithms can be configured with various parameters, and the resulting mazes can be filtered to satisfy desired properties. Also provided are tools for converting mazes to and from various formats suitable for a variety of neural network architectures, such as rasterized images, tokenized text sequences, and various visualizations. As well as providing a simple interface for generating, storing, and loading these datasets, `maze-dataset` is extensively tested, type hinted, benchmarked, and documented.

\input{figures/tex/fig1_diagram.tex}

# Statement of Need

While maze generation itself is straightforward, the architectural challenge comes from building a system supporting many algorithms with configurable parameters, property filtering, and representation transformation. This library aims to greatly streamline the process of generating and working with datasets of mazes that can be described as subgraphs of an $n \times n$ lattice with boolean connections and, optionally, start and end points that are nodes in the graph. Furthermore, we place emphasis on a wide variety of possible text output formats aimed at evaluating the spatial reasoning capabilities of Large Language Models (LLMs) and other text-based transformer models.

For interpretability and behavioral research, algorithmic tasks offer benefits by allowing systematic data generation and task decomposition, as well as simplifying the process of circuit discovery [@interpretability-survery]. Although mazes are well suited for these investigations, we found that existing maze generation packages [@cobbe2019procgen; @harriesMazeExplorerCustomisable3D2019; @gh_Ehsan_2022; @gh_Nemeth_2019; @easy_to_hard] lack support for transforming between multiple representations and provide limited control over the maze generation process.

## Related Works

A multitude of public and open-source software packages exist for generating mazes [@easy_to_hard; @gh_Ehsan_2022; @gh_Nemeth_2019]. However, nearly all of these packages produce mazes represented as rasterized images or other visual formats rather than the underlying graph structure, and this makes it difficult to work with these datasets.

- Most prior works provide mazes in visual or raster formats, and we provide a variety of similar output formats:
  - [`RasterizedMazeDataset`](https://understanding-search.github.io/maze-dataset/maze_dataset/dataset/rasterized.html#RasterizedMazeDataset), utilizing [`as_pixels()`](https://understanding-search.github.io/maze-dataset/maze_dataset.html#LatticeMaze.as_pixels), which can exactly mimic the outputs provided in `easy-to-hard-data`[@easy_to_hard] and can be configured to be similar to the outputs of @gh_Nemeth_2019
  - [`as_ascii()`](https://understanding-search.github.io/maze-dataset/maze_dataset.html#LatticeMaze.as_ascii) provides a format similar to [@eval-gpt-visual; @gh-oppenheimj2018maze]
  - [`MazePlot`](https://understanding-search.github.io/maze-dataset/maze_dataset/plotting.html#MazePlot) provides a feature-rich plotting utility with support for multiple paths, heatmaps over positions, and more. This is similar to the outputs of [@mdl-suite; @mathematica-maze; @mazegenerator-net; @gh_Ehsan_2022]


- The text format provided by [`SolvedMaze(...).as_tokens()`](https://understanding-search.github.io/maze-dataset/maze_dataset.html#MazeDataset.as_tokens) is similar to that of [@eval-LLM-graphs], but provides over 5.8 million unique formats for converting mazes to a text stream, detailed in \secref{sec:tokenized-output-formats}.

- For rigorous investigations of the response of a model to various distributional shifts, preserving metadata about the generation algorithm with the dataset itself is essential. To this end, our package efficiently stores the dataset along with its metadata in a single human-readable file [@zanj]. As far as we are aware, no existing packages do this reliably.

- Storing mazes as images is not only difficult to work with, but also inefficient. We use a highly efficient method detailed in \secref{sec:implementation}.

- Our package is easily installable with source code freely available. It is extensively tested, type hinted, benchmarked, and documented. Many other maze generation packages lack this level of rigor and scope, and some [@ayaz2008maze] appear to simply no longer be accessible.


# Features

We direct readers to our [examples](https://understanding-search.github.io/maze-dataset/examples/maze_examples.html), [docs](https://understanding-search.github.io/maze-dataset/maze_dataset.html), and [notebooks](https://understanding-search.github.io/maze-dataset/notebooks/) for more information.

## Generation and Basic Usage {#generation}

Our package can be installed from [PyPi](https://pypi.org/project/maze-dataset/) via `pip install maze-dataset`, or directly from the [git repository](https://github.com/understanding-search/maze-dataset) [@maze-dataset-github].

To create a dataset, we first create a [`MazeDatasetConfig`](https://understanding-search.github.io/maze-dataset/maze_dataset.html#MazeDatasetConfig) configuration object, which specifies the seed, number, and size of mazes, as well as the generation algorithm and its corresponding parameters. This object is passed to a [`MazeDataset`](https://understanding-search.github.io/maze-dataset/maze_dataset.html#MazeDataset) class to create a dataset. Crucially, this [`MazeDataset`](https://understanding-search.github.io/maze-dataset/maze_dataset.html#MazeDataset) mimics the interface of a PyTorch [@pytorch] [`Dataset`](https://pytorch.org/docs/stable/data.html), and can thus be easily incorporated into existing data pre-processing and training pipelines, e.g., through the use of a `DataLoader` class.

```python
from maze_dataset import (
  MazeDataset, MazeDatasetConfig, LatticeMazeGenerators
)
# create a config
cfg: MazeDatasetConfig = MazeDatasetConfig(
    name="example", # names need not be unique
    grid_n=3,   # size of the maze
    n_mazes=32, # number of mazes in the dataset
    maze_ctor=LatticeMazeGenerators.gen_dfs, # many algorithms available
    # (optional) algorithm-specific parameters
    maze_ctor_kwargs={"do_forks": True, ...}, 
    # (optional) many options for restricting start/end points
    endpoint_kwargs={"deadend_start": True, ...},
)
# create a dataset
dataset: MazeDataset = MazeDataset.from_config(
  cfg, # pass the config
  ..., # other options for disk loading, parallelization, etc.
)
``` 

When initializing a dataset, options which do not affect the mazes themselves can be specified through the [`from_config()`](https://understanding-search.github.io/maze-dataset/maze_dataset.html#MazeDataset.from_config) factory method as necessary. These options allow for saving/loading existing datasets instead of re-generating, parallelization options for generation, and more. Available maze generation algorithms are static methods of the [`LatticeMazeGenerators`](https://understanding-search.github.io/maze-dataset/maze_dataset.html#LatticeMazeGenerators) namespace class and include generation algorithms based on randomized depth-first search, Wilson's algorithm [@wilson], percolation [@percolation; @percolation-clustersize], Kruskal's algorithm [@kruskal1956shortest], and others.

Furthermore, a dataset of mazes can be filtered to satisfy certain properties. Custom filters can be specified, and some filters are included in [`MazeDatasetFilters`](https://understanding-search.github.io/maze-dataset/maze_dataset/dataset/filters.html#MazeDatasetFilters). For example, we can require a minimum path length of three steps from the origin to the target:

```python
dataset_filtered: MazeDataset = dataset.filter_by.path_length(min_length=3)
```

All implemented maze generation algorithms are stochastic by nature. For reproducibility, the `seed` parameter of [`MazeDatasetConfig`](https://understanding-search.github.io/maze-dataset/maze_dataset.html#MazeDatasetConfig) may be set. In practice, using provided deduplication filters, we find that exact duplicate mazes are generated very infrequently, even when generating very large datasets.

For use cases where mazes of different sizes, generation algorithms, or other parameter variations are required, we provide the [`MazeDatasetCollection`](https://understanding-search.github.io/maze-dataset/maze_dataset/dataset/collected_dataset.html#MazeDatasetCollection) class, which allows for creating a single iterable dataset from multiple independent configurations.

## Visual Output Formats {#visual-output-formats}

Internally, mazes are [`SolvedMaze`](https://understanding-search.github.io/maze-dataset/maze_dataset.html#SolvedMaze) objects, which have path information and a tensor optimized for storing sub-graphs of a lattice. These objects can be converted to and from several formats to maximize their utility in different contexts.

\input{figures/tex/fig2_formats.tex}

In previous work, maze tasks have been used with Recurrent Convolutional Neural Network (RCNN) derived architectures [@deepthinking]. To facilitate the use of our package in this context, we replicate the format of [@easy_to_hard] and provide the [`RasterizedMazeDataset`](https://understanding-search.github.io/maze-dataset/maze_dataset/dataset/rasterized.html#RasterizedMazeDataset) class which returns rasterized pairs of (input, target) mazes as shown in \autoref{fig:e2h-raster} below.

![Input is the rasterized maze without the path marked (left), and provide as a target the maze with all but the correct path removed (right). Configuration options exist to adjust whether endpoints are included and if empty cells should be filled in.](figures/maze-raster-input-target.pdf){#fig:e2h-raster width=30%}


## Tokenized Output Formats {#sec:tokenized-output-formats}

Autoregressive transformer models can be quite sensitive to the exact format of input data, and may even use delimiter tokens to perform reasoning steps [@pfau2024dotbydot; @spies2024causalworldmodels]. To facilitate systematic investigation of the effects of different representations of data on text model performance, we provide a variety of tokenized text output formats.

We convert mazes to token sequences in two steps. First, the maze is stringified using [`as_tokens()`](https://understanding-search.github.io/maze-dataset/maze_dataset.html#MazeDataset.as_tokens). The [`MazeTokenizerModular`](https://understanding-search.github.io/maze-dataset/maze_dataset/tokenization.html#MazeTokenizerModular) class provides a powerful interface for configuring maze stringification behavior. Second, the sequence of strings is tokenized into integers using `encode()`. Tokenization uses a fixed vocabulary for simplicity. Mazes up to $50 \times 50$ are supported when using a unique token for each position, and up to $128 \times 128$ are supported when positions in the maze are represented as a pair of coordinates.

There are many algorithms by which one might tokenize a 2D maze into a 1D format usable by autoregressive text models. Training multiple models on the encodings output from each of these algorithms may produce very different internal representations, learned solution algorithms, and levels of performance. To allow exploration of how different maze tokenization algorithms affect these models, the [`MazeTokenizerModular`](https://understanding-search.github.io/maze-dataset/maze_dataset/tokenization.html#MazeTokenizerModular) class contains a rich set of options to customize how mazes are stringified. This class contains 19 discrete parameters, resulting in over 5.8 million unique tokenizers. There are 6 additional parameters available whose functionality is not verified via automated testing, but further expand the the number of tokenizers by a factor of $44/3$ to 86 million.

All output sequences consist of four token regions representing different features of the maze; an example output sequence is shown in \autoref{fig:token-regions}.

\input{figures/tex/fig4_tokenfmt.tex}

Each [`MazeTokenizerModular`](https://understanding-search.github.io/maze-dataset/maze_dataset/tokenization.html#MazeTokenizerModular) is constructed from a set of several \docslinkcode{maze_dataset/tokenization.html#_TokenizerElement}{\_TokenizerElement} objects, each of which specifies how different token regions or other elements of the stringification are produced.

\input{figures/tex/fig5_mmt.tex}

The tokenizer architecture is purposefully designed such that adding and testing a wide variety of new tokenization algorithms is fast and minimizes disturbances to functioning code. This is enabled by the modular architecture and the automatic inclusion of any new tokenizers in integration tests. To create a new variety of tokenizer, developers forking the library may simply create their own \docslinkcode{maze_dataset/tokenization.html#_TokenizerElement}{\_TokenizerElement} subclass and implement the abstract methods. If the behavior change is sufficiently small, simply adding a parameter to an existing \docslinkcode{maze_dataset/tokenization.html#_TokenizerElement}{\_TokenizerElement} subclass and updating its implementation will suffice.

The breadth of tokenizers is also easily scaled in the opposite direction. Due to the exponential scaling of parameter combinations, adding a small number of new features can significantly slow certain procedures which rely on constructing all possible tokenizers, such as integration tests. If any existing subclass contains features which aren't needed, a developer tool decorator [`@mark_as_unsupported`](https://understanding-search.github.io/maze-dataset/maze_dataset/tokenization/modular/element_base.html#mark_as_unsupported) is provided which can be applied to the unneeded \docslinkcode{maze_dataset/tokenization.html#_TokenizerElement}{\_TokenizerElement} subclasses to prune those features and compact the available space of tokenizers.

## Benchmarks of Generation Speed {#benchmarks}

We provide approximate benchmarks for relative generation time across various algorithms, parameter choices, maze sizes, and dataset sizes in \autoref{tab:benchmarks} and \autoref{fig:benchmarks}. Experiments were performed on a \href{https://docs.github.com/en/actions/using-github-hosted-runners/using-github-hosted-runners/about-github-hosted-runners#standard-github-hosted-runners-for-public-repositories}{standard GitHub runner} without parallelism.

\input{figures/tex/tab1_benchmarks.tex}


![Plot of maze generation time. Generation time scales exponentially with maze size for all algorithms. Generation time per maze does not depend on the number of mazes being generated, and there is minimal overhead to initializing the generation process for a small dataset. Wilson's algorithm is notably less efficient than others and has high variance. Note that values are averaged across all parameter sets for that algorithm. More information can be found on the [benchmarks page](https://understanding-search.github.io/maze-dataset/benchmarks/).](figures/benchmarks/gridsize-vs-gentime.pdf){#fig:benchmarks width=90%}

## Success Rate Estimation {#sec:success-rate-estimation}

In order to replicate the exact dataset distribution of [@easy_to_hard], the parameter [`MazeDatasetConfig.endpoint_kwargs:`](https://understanding-search.github.io/maze-dataset/maze_dataset/dataset/maze_dataset_config.html#MazeDatasetConfig.endpoint_kwargs) [`EndpointKwargsType`](https://understanding-search.github.io/maze-dataset/maze_dataset/dataset/maze_dataset_config.html#EndpointKwargsType) allows for additional constraints such as enforcing that the start or end point be in a "dead end" with only one accessible neighbor cell. However, combining these constraints with cyclic mazes (such as those generated with percolation), as was required for the work in [@knutson2024logicalextrapolation], can lead to an absence of valid start and end points. Placing theoretical bounds on this success rate is difficult, as it depends on the exact maze generation algorithm and parameters used. To deal with this, our package provides a way to estimate the success rate of a given configuration using a symbolic regression model trained with PySR [@pysr]. More details on this can be found in [`estimate_dataset_fractions.ipynb`](https://understanding-search.github.io/maze-dataset/notebooks/estimate_dataset_fractions.html). Using the estimation algorithm simply requires the user to call [`cfg_new: MazeDatasetConfig = cfg.success_fraction_compensate()`](https://understanding-search.github.io/maze-dataset/maze_dataset.html#MazeDatasetConfig.success_fraction_compensate), providing their initial `cfg` and then using the returned `cfg_new` in its place.

### Success Rate Estimation Algorithm

The base function learned by symbolic regression provides limited insight and may be subject to change. It is defined as [`cfg_success_predict_fn`](https://understanding-search.github.io/maze-dataset/maze_dataset/dataset/success_predict_math.html#cfg_success_predict_fn), and takes a 5 dimensional float vector created by `MazeDatasetConfig._to_ps_array()` which represents the [percolation value, grid size, endpoint deadend configuration, endpoint uniqueness, categorical generation function index].

However, the outputs of this function are not directly usable due to minor divergences at the endpoints with respect to the percolation probability $p$. Since we know that maze success is either guaranteed or impossible for $p=0$ and $p=1$, we define the [`soft_step`](https://understanding-search.github.io/maze-dataset/maze_dataset/dataset/success_predict_math.html#soft_step) function to nudge the raw output of the symbolic regression. This function is defined with the following components:

shifted sigmoid $\sigma_s$, amplitude scaling $A$, and $h$ function given by
$$
  \sigma_s(x) = (1 + e^{-10^3 \cdot (x-0.5)})^{-1}
  \qquad A(q,a,w) = w \cdot (1 - |2q-1|^a)
$$
$$
  h(q,a) = q \cdot (1 - |2q-1|^a) \cdot (1-\sigma_s(q)) + (1-(1-q) \cdot (1 - |2(1-q)-1|^a)) \cdot \sigma_s(q)
$$

We combine these to get the [`soft_step`](https://understanding-search.github.io/maze-dataset/maze_dataset/dataset/success_predict_math.html#soft_step) function, which is identity-like for $p \approx 0.5$, and pushes $x$ to extremes otherwise.
$$
  \text{soft\_step}(x, p, \alpha, w) = h(x, A(p, \alpha, w))
$$

Finally, we define
$$
  \text{cfg\_success\_predict\_fn}(\mathbf{x}) = \text{soft\_step}(\text{raw\_val}, x_0, 5, 10)
$$

where `raw_val` is the output of the symbolic regression model. The parameter $x_0$ is the percolation probability, while all other parameters from `_to_ps_array()` only affect `raw_val`.

![An example of both empirical and predicted success rates as a function of the percolation probability $p$ for various maze sizes, percolation with and without depth first search, and `endpoint_kwargs` requiring that both the start and end be in unique dead ends. Empirical measures derived from a sample of 128 mazes. More information can be found on the [benchmarks page](https://understanding-search.github.io/maze-dataset/benchmarks/).](figures/ep/ep_deadends_unique-crop.pdf){width=100%}

# Implementation {#sec:implementation}

We refer to our \href{https://github.com/understanding-search/maze-dataset}{repository} and \docslink{maze_dataset.html}{docs} for documentation and up-to-date implementation details.

This package utilizes a simple, efficient representation of mazes as subgraphs of a finite lattice, which we call a [`LatticeMaze`](https://understanding-search.github.io/maze-dataset/maze_dataset.html#LatticeMaze). Using an adjacency matrix for storing mazes would be memory inefficient by failing to exploit the highly sparse structure -- for example, for a 2-dimensional maze, only 4 off-diagonal bands would be have nonzero values. On the other hand, using an adjacency list could lead to a poor lookup time for whether any given connection exists.

Instead, we describe mazes with the following representation: for a $2$-dimensional lattice with $r$ rows and $c$ columns, we initialize a boolean array
$$
  A = \{0, 1\}^{2 \times r \times c}
$$
which we refer to in the code as a [`connection_list`](https://understanding-search.github.io/maze-dataset/maze_dataset.html#LatticeMaze.connection_list). The value at $A[0,i,j]$ determines whether a *downward* connection exists from node $[i,j]$ to $[i+1, j]$. Likewise, the value at $A[1,i,j]$ determines whether a *rightward* connection to $[i, j+1]$ exists. Thus, we avoid duplication of data about the existence of connections and facilitate fast lookup time, at the cost of requiring additional care with indexing. Note that this setup allows for a periodic lattice. Generation of mazes is detailed in [`LatticeMazeGenerators`](https://understanding-search.github.io/maze-dataset/maze_dataset.html#LatticeMazeGenerators).

To produce solutions to mazes, two points are selected uniformly at random without replacement from the connected component of the maze, and the $A^*$ algorithm [@A_star] is applied to find the shortest path between them. The endpoint selection can be controlled via [`MazeDatasetConfig.endpoint_kwargs:`](https://understanding-search.github.io/maze-dataset/maze_dataset/dataset/maze_dataset_config.html#MazeDatasetConfig.endpoint_kwargs) [`EndpointKwargsType`](https://understanding-search.github.io/maze-dataset/maze_dataset/dataset/maze_dataset_config.html#EndpointKwargsType), and complications caused by this are detailed in \secref{sec:success-rate-estimation}. A maze with a solution is denoted a [`SolvedMaze`](https://understanding-search.github.io/maze-dataset/maze_dataset.html#SolvedMaze), which inherits from [`LatticeMaze`](https://understanding-search.github.io/maze-dataset/maze_dataset.html#LatticeMaze).

Parallelization is implemented via the `multiprocessing` module in the Python standard library, and parallel generation can be controlled via keyword arguments to [`MazeDataset.from_config()`](https://understanding-search.github.io/maze-dataset/maze_dataset.html#MazeDataset.from_config).

\newpage

# Usage in Research

This package was originally built for the needs of the [@maze-transformer-github] project, which aims to investigate spatial planning and world models in autoregressive transformer models trained on mazes [@ivanitskiy2023structuredworldreps; @spies2024causalworldmodels; @maze-dataset-arxiv-2023]. It was extended for work on understanding the mechanisms by which recurrent convolutional and implicit networks [@fung2022jfb] solve mazes given a rasterized view [@knutson2024logicalextrapolation], which required matching the pixel-padded and endpoint constrained output format of [@easy_to_hard]. Ongoing work using `maze-dataset` aims to investigate the effects of varying the tokenization format on the performance of pretrained LLMs on spatial reasoning.

This package has also been utilized in work by other groups:

- By [@nolte2024multistep] to compare the effectiveness of transformers trained with the MLM-$\mathcal{U}$ [@MLMU-kitouni2024factorization] multistep prediction objective against standard autoregressive training for multi-step planning on our maze task.

- By [@wang2024imperative] and [@chen2024iaimperative] to study the effectiveness of imperative learning.

- By [@zhang2025tscend] to introduce a novel framework for reasoning diffusion models.

- By [@dao2025alphamaze] to improve spatial reasoning in LLMs with GRPO.

# Acknowledgements

\input{figures/tex/acknowledgements.tex}

\newpage

# References