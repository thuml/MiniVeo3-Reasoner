"""preserving legacy imports"""

from maze_dataset.tokenization.maze_tokenizer_legacy import (
	MazeTokenizer,
	TokenizationMode,
)
from maze_dataset.tokenization.modular.maze_tokenizer_modular import (
	MazeTokenizerModular,
)

__all__ = [
	"MazeTokenizer",
	"TokenizationMode",
	"MazeTokenizerModular",
]
