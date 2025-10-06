"""turning a maze into text

- `MazeTokenizerModular` is the new recommended way to do this as of 1.0.0
- legacy `TokenizationMode` enum and `MazeTokenizer` class for supporting existing code
- a variety of helper classes and functions

There are many algorithms by which one might tokenize a 2D maze into a 1D format usable by autoregressive text models. Training multiple models on the encodings output from each of these algorithms may produce very different internal representations, learned solution algorithms, and levels of performance. To explore how different maze tokenization algorithms affect these models, the `MazeTokenizerModular` class contains a rich set of options to customize how mazes are stringified. This class contains 19 discrete parameters, resulting in 5.9 million unique tokenizers. But wait, there's more! There are 6 additional parameters available in the library which are untested but further expand the the number of tokenizers by a factor of $44/3$ to 86 million.

All output sequences consist of four token regions representing different features of the maze. These regions are distinguished by color in Figure below.

- <span style="background-color:rgb(217,210,233)">Adjacency list</span>: A text representation of the lattice graph
- <span style="background-color:rgb(217,234,211)">Origin</span>: Starting coordinate
- <span style="background-color:rgb(234,209,220)">Target</span>: Ending coordinate
- <span style="background-color:rgb(207,226,243)">Path</span>: Maze solution sequence from the start to the end

![Example text output format with token regions highlighted.](figures/outputs-tokens-colored.tex)

Each `MazeTokenizerModular` is constructed from a set of several `_TokenizerElement` objects, each of which specifies how different token regions or other elements of the stringification are produced.

![Nested internal structure of `_TokenizerElement` objects inside a typical `MazeTokenizerModular` object.](figures/TokenizerElement_structure.pdf)

Optional delimiter tokens may be added in many places in the output. Delimiter options are all configured using the parameters named `pre`, `intra`, and `post` in various `_TokenizerElement` classes. Each option controls a unique delimiter token.
Here we describe each `_TokenizerElement` and the behaviors they support. We also discuss some of the model behaviors and properties that may be investigated using these options.

### Coordinates

The `_CoordTokenizer` object controls how coordinates in the lattice are represented in across all token regions. Options include:

- **Unique tokens**: Each coordinate is represented as a single unique token `"(i,j)"`
- **Coordinate tuple tokens**: Each coordinate is represented as a sequence of 2 tokens, respectively encoding the row and column positions: `["i", ",", "j"]`

### Adjacency List

The `_AdjListTokenizer` object controls this token region. All tokenizations represent the maze connectivity as a sequence of connections or walls between pairs of adjacent coordinates in the lattice.

- `_EdgeSubset`: Specifies the subset of lattice edges to be tokenized
  - **All edges**: Every edge in the lattice
  - **Connections**: Only edges which contain a connection
  - **Walls**: Only edges which contain a wall
- `_EdgePermuter`: Specifies how to sequence the two coordinates in each lattice edge
  - **Random**
  - **Sorted**: The smaller coordinate always comes first
  - **Both permutations**: Each edge is represented twice, once with each permutation. This option attempts to represent connections in a more directionally symmetric manner. Including only one permutation of each edge may affect models' internal representations of edges, treating a path traversing the edge differently depending on if the coordinate sequence in the path matches the sequence in the adjacency list.
- `shuffle_d0`: Whether to shuffle the edges randomly or sort them in the output by their first coordinate
- `connection_token_ordinal`: Location in the sequence of the token representing whether the edge is a connection or a wall

### Path

The `_PathTokenizer` object controls this token region. Paths are all represented as a sequence of steps moving from the start to the end position.

- `_StepSize`: Specifies the size of each step
  - **Singles**: Every coordinate traversed between start and end is directly represented
  - **Forks**: Only coordinates at forking points in the maze are represented. The paths between forking points are implicit. Using this option might train models more directly to represent forking points differently from coordinates where the maze connectivity implies an obvious next step in the path.
- `_StepTokenizer`: Specifies how an individual step is represented
  - **Coordinate**: The coordinates of each step are directly tokenized using a `_CoordTokenizer`
  - **Cardinal direction**: A single token corresponding to the cardinal direction taken at the starting position of that step. E.g., `NORTH`, `SOUTH`. If using a `_StepSize` other than **Singles**, this direction may not correspond to the final direction traveled to arrive at the end position of the step.
  - **Relative direction**: A single token corresponding to the first-person perspective relative direction taken at the starting position of that step. E.g., `RIGHT`, `LEFT`.
  - **Distance**: A single token corresponding to the number of coordinate positions traversed in that step. E.g., using a `_StepSize` of **Singles**, the **Distance** token would be the same for each step, corresponding to a distance of 1 coordinate. This option is only of interest in combination with a `_StepSize` other than **Singles**.

A `_PathTokenizer` contains a sequence of one or more unique `_StepTokenizer` objects. Different step representations may be mixed and permuted, allowing for investigation of model representations of multiple aspects of a maze solution at once.

## Tokenized Outputs for Training and Evaluation {#token-training}

During deployment we provide only the prompt up to the `<PATH_START>` token.

Examples of usage of this dataset to train autoregressive transformers can be found in our `maze-transformer` library [@maze-transformer-github]. Other tokenization and vocabulary schemes are also included, such as representing each coordinate as a pair of $i,j$ index tokens.

## Extensibility

The tokenizer architecture is purposefully designed such that adding and testing a wide variety of new tokenization algorithms is fast and minimizes disturbances to functioning code. This is enabled by the modular architecture and the automatic inclusion of any new tokenizers in integration tests. To create a new tokenizer, developers forking the library may simply create their own `_TokenizerElement` subclass and implement the abstract methods. If the behavior change is sufficiently small, simply adding a parameter to an existing `_TokenizerElement` subclass and updating its implementation will suffice. For small additions, simply adding new cases to existing unit tests will suffice.

The breadth of tokenizers is also easily scaled in the opposite direction. Due to the exponential scaling of parameter combinations, adding a small number of new features can significantly slow certain procedures which rely on constructing all possible tokenizers, such as integration tests. If any existing subclass contains features which aren't needed, a developer tool decorator is provided which can be applied to the unneeded `_TokenizerElement` subclasses to prune those features and compact the available space of tokenizers.

"""

from maze_dataset.tokenization.maze_tokenizer_legacy import (
	MazeTokenizer,
	TokenizationMode,
	get_tokens_up_to_path_start,
)
from maze_dataset.tokenization.modular.element_base import _TokenizerElement
from maze_dataset.tokenization.modular.elements import (
	AdjListTokenizers,
	CoordTokenizers,
	EdgeGroupings,
	EdgePermuters,
	EdgeSubsets,
	PathTokenizers,
	PromptSequencers,
	StepSizes,
	StepTokenizers,
	TargetTokenizers,
)
from maze_dataset.tokenization.modular.maze_tokenizer_modular import (
	MazeTokenizerModular,
)

# we don't sort alphabetically on purpose, we sort by the type
__all__ = [
	# submodules
	"modular",
	"common",
	"maze_tokenizer_legacy",
	"maze_tokenizer",
	# legacy tokenizer
	"MazeTokenizer",
	"TokenizationMode",
	# MMT
	"MazeTokenizerModular",
	# element base
	"_TokenizerElement",
	# elements
	"PromptSequencers",
	"CoordTokenizers",
	"AdjListTokenizers",
	"EdgeGroupings",
	"EdgePermuters",
	"EdgeSubsets",
	"TargetTokenizers",
	"StepSizes",
	"StepTokenizers",
	"PathTokenizers",
	# helpers
	"get_tokens_up_to_path_start",
]
