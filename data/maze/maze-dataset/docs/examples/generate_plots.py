"""Generate HTML examples of various maze configurations.

This script generates a variety of maze examples with different configurations,
saves them as SVG files, and creates an HTML page to display them with searchable
configuration details.
"""

import json
from pathlib import Path
from typing import Any, Callable

import jinja2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
from tqdm import tqdm

from maze_dataset import MazeDatasetConfig
from maze_dataset.dataset.configs import _get_configs_for_examples
from maze_dataset.dataset.maze_dataset import MazeDataset
from maze_dataset.plotting import MazePlot

plt.rcParams["svg.fonttype"] = "none"  # preserve text as text

# Define the examples directory
EXAMPLES_DIR: Path = Path("docs/examples")
"where everything is happening"

PLOTS_DIR: Path = EXAMPLES_DIR / "plots"
"where to put the plots"

HTML_PATH: Path = EXAMPLES_DIR / "maze_examples.html"
"where to write html"

TEMPLATE_FILE: str = "maze_examples.html.jinja2"
"template file for the html, relative to `EXAMPLES_DIR`"

N_EXAMPLES_GENERATE: int = 6
"number of examples to generate for each configuration"


def generate_maze_plots(config_src: dict[str, Any]) -> tuple[str, dict]:
	"""Generate mazes from the given configuration and plots and metadata for it

	returns (config fname, metadata)
	"""
	# Extract maze config parameters
	name: str = config_src["name"]
	grid_n: int = config_src["grid_n"]
	maze_ctor: Callable = config_src["maze_ctor"]
	maze_ctor_kwargs: dict = config_src["maze_ctor_kwargs"]
	endpoint_kwargs: dict = config_src.get("endpoint_kwargs", {})
	# so we don't error out
	endpoint_kwargs["except_on_no_valid_endpoint"] = False

	# Create a MazeDatasetConfig
	md_config: MazeDatasetConfig = MazeDatasetConfig(
		name=name,
		grid_n=grid_n,
		n_mazes=N_EXAMPLES_GENERATE,
		maze_ctor=maze_ctor,
		maze_ctor_kwargs=maze_ctor_kwargs,
		endpoint_kwargs=endpoint_kwargs,
	)
	md_config = md_config.success_fraction_compensate(
		safety_margin=1.5, except_if_all_success_expected=False
	)

	# get the fname and the path
	cfg_fname: str = md_config.to_fname()
	this_cfg_path: Path = PLOTS_DIR / cfg_fname
	this_cfg_path.mkdir(exist_ok=True)

	# Generate the maze directly
	dataset: MazeDataset = MazeDataset.from_config(
		cfg=md_config,
		do_generate=True,
		load_local=False,
		save_local=False,
		verbose=False,
	)

	# create and save the plots
	for i, maze in enumerate(dataset):
		Image.fromarray(maze.as_pixels()).save(this_cfg_path / f"pixels-{i}.png")
		fig: Figure = MazePlot(maze).plot(plain=True).fig
		fig.tight_layout()
		fig.savefig(this_cfg_path / f"plot-{i}.svg")
		plt.close(fig)
		if i > N_EXAMPLES_GENERATE:
			break

	# Prepare JSON configuration
	metadata_json = config_src.copy()
	metadata_json["maze_ctor"] = config_src["maze_ctor"].__name__
	metadata_json["config"] = md_config.serialize()
	metadata_json["fname"] = cfg_fname

	# save the metadata
	with open(this_cfg_path / "metadata.json", "w") as f:
		f.write(json.dumps(metadata_json, indent=2))

	return cfg_fname, metadata_json


def main() -> None:
	"""Main function to generate maze examples."""
	print(f"Generating maze examples in {EXAMPLES_DIR}")

	# setup
	PLOTS_DIR.mkdir(exist_ok=True)
	configs_sources: list[dict] = _get_configs_for_examples()
	print(f"Found {len(configs_sources)} maze configurations")

	# Generate plots
	maze_examples: list[dict] = []
	all_tags: set[str] = set()

	for _i, cfg_src in tqdm(
		enumerate(configs_sources),
		desc="Generating maze examples",
		total=len(configs_sources),
	):
		fname, metadata = generate_maze_plots(cfg_src)

		# Update the set of all tags
		all_tags.update(metadata["tags"])

		# Prepare the example data for the template
		maze_examples.append(metadata)

	# Set up Jinja2 and generate HTML
	jinja_env: jinja2.Environment = jinja2.Environment(
		loader=jinja2.FileSystemLoader(EXAMPLES_DIR),
		autoescape=jinja2.select_autoescape(["html", "xml"]),
	)

	# render the html
	template: jinja2.Template = jinja_env.get_template(TEMPLATE_FILE)
	with open(HTML_PATH, "w") as f:
		f.write(
			template.render(
				maze_examples=maze_examples,
				all_tags=sorted(all_tags),
			)
		)

	print(f"Generated HTML at {HTML_PATH}")


if __name__ == "__main__":
	main()
