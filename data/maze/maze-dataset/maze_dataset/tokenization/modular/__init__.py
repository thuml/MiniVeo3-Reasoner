"""implements `ModularMazeTokenizer` and related code

the structure of a typical `MazeTokenizerModular` is something like this:
```
+----------------------------------------------------+
|                  MazeTokenizerModular              |
|  +-----------------------------------------------+ |
|  |                 _PromptSequencer              | |
|  |         +-----------------------------+       | |
|  |         |       _CoordTokenizer       |       | |
|  |         +-----------------------------+       | |
|  |     +------------------------------------+    | |
|  |     |         _AdjListTokenizer          |    | |
|  |     | +-----------+    +-------------+   |    | |
|  |     | |_EdgeSubset|    |_EdgeGrouping|   |    | |
|  |     | +-----------+    +-------------+   |    | |
|  |     |          +-------------+           |    | |
|  |     |          |_EdgePermuter|           |    | |
|  |     |          +-------------+           |    | |
|  |     +------------------------------------+    | |
|  |         +-----------------------------+       | |
|  |         |      _TargetTokenizer       |       | |
|  |         +-----------------------------+       | |
|  |  +------------------------------------------+ | |
|  |  |              _PathTokenizer              | | |
|  |  |  +---------------+   +----------------+  | | |
|  |  |  |   _StepSize   |   | _StepTokenizer |  | | |
|  |  |  +---------------+   +----------------+  | | |
|  |  |                      | _StepTokenizer |  | | |
|  |  |                      +----------------+  | | |
|  |  |                             :            | | |
|  |  +------------------------------------------+ | |
|  +-----------------------------------------------+ |
+----------------------------------------------------+
```

Optional delimiter tokens may be added in many places in the output. Delimiter options are all configured using the parameters named `pre`, `intra`, and `post` in various `_TokenizerElement` classes. Each option controls a unique delimiter token.
Here we describe each `_TokenizerElement` and the behaviors they support. We also discuss some of the model behaviors and properties that may be investigated using these options.

### Coordinates {#coordtokenizer}

The `_CoordTokenizer` object controls how coordinates in the lattice are represented in across all token regions. Options include:

- **Unique tokens**: Each coordinate is represented as a single unique token `"(i,j)"`
- **Coordinate tuple tokens**: Each coordinate is represented as a sequence of 2 tokens, respectively encoding the row and column positions: `["i", ",", "j"]`

### Adjacency List {#adjlisttokenizer}

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

### Path {#pathtokenizer}

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

"""

__all__ = [
	# modules
	"all_instances",
	"all_tokenizers",
	"element_base",
	"elements",
	"fst_load",
	"fst",
	"hashing",
	"maze_tokenizer_modular",
	"save_hashes",
]
