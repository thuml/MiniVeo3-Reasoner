"""python project makefile template -- docs generation script
originally by Michael Ivanitskiy (mivanits@umich.edu)
https://github.com/mivanit/python-project-makefile-template
license: https://creativecommons.org/licenses/by-sa/4.0/
modifications from the original should be denoted with `~~~~~`
as this makes it easier to find edits when updating
"""

import argparse
import inspect
import json
import re
import warnings
from dataclasses import asdict, dataclass, field
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
	import tomllib  # type: ignore[import-not-found]
except ImportError:
	import tomli as tomllib  # type: ignore

import jinja2
import pdoc  # type: ignore[import-not-found]
import pdoc.doc  # type: ignore[import-not-found]
import pdoc.extract  # type: ignore[import-not-found]
import pdoc.render  # type: ignore[import-not-found]
import pdoc.render_helpers  # type: ignore[import-not-found]
from markupsafe import Markup

"""
 ######  ######## ######## ##     ## ########
##    ## ##          ##    ##     ## ##     ##
##       ##          ##    ##     ## ##     ##
 ######  ######      ##    ##     ## ########
      ## ##          ##    ##     ## ##
##    ## ##          ##    ##     ## ##
 ######  ########    ##     #######  ##
"""
# setup
# ============================================================

CONFIG_PATH: Path = Path("pyproject.toml")
TOOL_PATH: str = "tool.makefile.docs"

HTML_TO_MD_MAP: Dict[str, str] = {
	"&gt;": ">",
	"&lt;": "<",
	"&amp;": "&",
	"&quot;": '"',
	"&#39": "'",
	"&apos;": "'",
}

pdoc.render_helpers.markdown_extensions["alerts"] = True  # type: ignore[assignment]
pdoc.render_helpers.markdown_extensions["admonitions"] = True  # type: ignore[assignment]


_CONFIG_NOTEBOOKS_INDEX_TEMPLATE: str = r"""<!doctype html>
<html lang="en">
<head>
	<meta charset="UTF-8">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<title>Notebooks</title>
	<link rel="stylesheet" href="../resources/css/bootstrap-reboot.min.css">
	<link rel="stylesheet" href="../resources/css/theme.css">
	<link rel="stylesheet" href="../resources/css/content.css">
</head>
<body>    
	<h1>Notebooks</h1>
	<p>
		You can find the source code for the notebooks at
		<a href="{{ notebook_url }}">{{ notebook_url }}</a>.
	<ul>
		{% for notebook in notebooks %}
		<li><a href="{{ notebook.html }}">{{ notebook.ipynb }}</a> {{ notebook.desc }}</li>
		{% endfor %}
	</ul>
	<a href="../">Back to index</a>
</body>
</html>
"""


def deep_get(
	d: dict,
	path: str,
	default: Any = None,
	sep: str = ".",
	warn_msg_on_default: Optional[str] = None,
) -> Any:
	"Get a value from a nested dictionary"
	output: Any = reduce(
		lambda x, y: x.get(y, default) if isinstance(x, dict) else default,  # function
		path.split(sep) if isinstance(path, str) else path,  # sequence
		d,  # initial
	)

	if warn_msg_on_default and output == default:
		warnings.warn(warn_msg_on_default.format(path=path))

	return output


# CONFIGURATION -- read from CONFIG_PATH, assumed to be a pyproject.toml
# ============================================================


@dataclass
class Config:
	# under main pyproject.toml
	package_name: str = "unknown"
	package_repo_url: str = "unknown"
	package_version: str = "unknown"
	# under tool_path
	output_dir_str: str = "docs"
	markdown_headings_increment: int = 2
	warnings_ignore: list[str] = field(default_factory=list)
	notebooks_enabled: bool = False
	notebooks_descriptions: dict[str, str] = field(default_factory=dict)
	notebooks_source_path_str: str = "notebooks"
	notebooks_output_path_relative_str: str = "notebooks"
	notebooks_index_template: str = _CONFIG_NOTEBOOKS_INDEX_TEMPLATE

	@property
	def package_code_url(self) -> str:
		if "unknown" not in (self.package_name, self.package_version):
			return self.package_repo_url + "/blob/" + self.package_version
		else:
			return "unknown"

	@property
	def module_name(self) -> str:
		return self.package_name.replace("-", "_")

	@property
	def output_dir(self) -> Path:
		return Path(self.output_dir_str)

	@property
	def notebooks_source_path(self) -> Path:
		return Path(self.notebooks_source_path_str)

	@property
	def notebooks_output_path(self) -> Path:
		return self.output_dir / self.notebooks_output_path_relative_str


CONFIG: Config

_CFG_PATHS: dict[str, str] = dict(
	package_name="project.name",
	package_repo_url="project.urls.Repository",
	package_version="project.version",
	output_dir_str=f"{TOOL_PATH}.output_dir",
	markdown_headings_increment=f"{TOOL_PATH}.markdown_headings_increment",
	warnings_ignore=f"{TOOL_PATH}.warnings_ignore",
	notebooks_enabled=f"{TOOL_PATH}.notebooks.enabled",
	notebooks_descriptions=f"{TOOL_PATH}.notebooks.descriptions",
	notebooks_index_template=f"{TOOL_PATH}.notebooks.index_template",
	notebooks_source_path_str=f"{TOOL_PATH}.notebooks.source_path",
	notebooks_output_path_relative_str=f"{TOOL_PATH}.notebooks.output_path_relative",
)


def set_global_config() -> None:
	"""set global var `CONFIG` from pyproject.toml"""
	global CONFIG

	# get the default and read the data
	cfg_default: Config = Config()

	with CONFIG_PATH.open("rb") as f:
		pyproject_data = tomllib.load(f)

	# apply the mapping from toml path to attribute
	cfg_partial: dict = {
		key: deep_get(
			d=pyproject_data,
			path=path,
			default=getattr(cfg_default, key),
			warn_msg_on_default=f"could not find {path}"
			if key.startswith("package")
			else None,
		)
		for key, path in _CFG_PATHS.items()
	}

	# set the global var
	CONFIG = Config(**cfg_partial)

	# add the package meta to the pdoc globals
	pdoc.render.env.globals["package_version"] = CONFIG.package_version
	pdoc.render.env.globals["package_name"] = CONFIG.package_name
	pdoc.render.env.globals["package_repo_url"] = CONFIG.package_repo_url
	pdoc.render.env.globals["package_code_url"] = CONFIG.package_code_url


"""
##     ## ########
###   ### ##     ##
#### #### ##     ##
## ### ## ##     ##
##     ## ##     ##
##     ## ##     ##
##     ## ########
"""
# markdown
# ============================================================


def replace_heading(match: re.Match) -> str:
	current_level: int = len(match.group(1))
	new_level: int = min(
		current_level + CONFIG.markdown_headings_increment, 6
	)  # Cap at h6
	return "#" * new_level + match.group(2)


def increment_markdown_headings(markdown_text: str) -> str:
	"""Increment all Markdown headings in the given text by the specified amount.

	Args:
	    markdown_text (str): The input Markdown text.

	Returns:
	    str: The Markdown text with incremented heading levels.

	"""
	# Regular expression to match Markdown headings
	heading_pattern: re.Pattern = re.compile(r"^(#{1,6})(.+)$", re.MULTILINE)

	# Replace all headings with incremented versions
	return heading_pattern.sub(replace_heading, markdown_text)


def format_signature(sig: inspect.Signature, colon: bool) -> str:
	"""Format a function signature for Markdown. Returns a single-line Markdown string."""
	# First get a list with all params as strings.
	result = pdoc.doc._PrettySignature._params(sig)  # type: ignore
	return_annot = pdoc.doc._PrettySignature._return_annotation_str(sig)  # type: ignore

	def _format_param(param: str) -> str:
		"""Format a parameter for Markdown, including potential links."""
		# This is a simplified version. You might need to adjust this
		# to properly handle links in your specific use case.
		return f"`{param}`"

	# Format each parameter
	pretty_result = [_format_param(param) for param in result]

	# Join parameters
	params_str = ", ".join(pretty_result)

	# Add return annotation
	anno = ")"
	if return_annot:
		anno += f" -> `{return_annot}`"
	if colon:
		anno += ":"

	# Construct the full signature
	rendered = f"`(`{params_str}`{anno}`"

	return rendered


def markup_safe(sig: inspect.Signature) -> str:
	"mark some text as safe, no escaping needed"
	output: str = str(sig)
	return Markup(output)


def use_markdown_format() -> None:
	"set some functions to output markdown format"
	pdoc.render_helpers.format_signature = format_signature
	pdoc.render.env.filters["markup_safe"] = markup_safe
	pdoc.render.env.filters["increment_markdown_headings"] = increment_markdown_headings


"""
##    ## ########
###   ## ##     ##
####  ## ##     ##
## ## ## ########
##  #### ##     ##
##   ### ##     ##
##    ## ########
"""


# notebook
# ============================================================
def convert_notebooks() -> None:
	try:
		import nbconvert
		import nbformat
	except ImportError:
		raise ImportError(
			'nbformat and nbconvert are required to convert notebooks to HTML, add "nbconvert>=7.16.4" to dev/docs deps'
		)

	# create output directory
	CONFIG.notebooks_output_path.mkdir(parents=True, exist_ok=True)

	# read in the notebook metadata
	notebook_names: list[Path] = list(CONFIG.notebooks_source_path.glob("*.ipynb"))
	notebooks: list[dict[str, str]] = [
		dict(
			ipynb=notebook.name,
			html=notebook.with_suffix(".html").name,
			desc=CONFIG.notebooks_descriptions.get(notebook.stem, ""),
		)
		for notebook in notebook_names
	]

	# Render the index template
	template: jinja2.Template = jinja2.Template(CONFIG.notebooks_index_template)
	rendered_index: str = template.render(notebooks=notebooks)

	# Write the rendered index to a file
	index_path: Path = CONFIG.notebooks_output_path / "index.html"
	index_path.write_text(rendered_index)

	# convert with nbconvert
	for notebook in notebook_names:
		output_notebook: Path = (
			CONFIG.notebooks_output_path / notebook.with_suffix(".html").name
		)
		with open(notebook, "r") as f:
			nb: nbformat.NotebookNode = nbformat.read(f, as_version=4)
			html_exporter: nbconvert.HTMLExporter = nbconvert.HTMLExporter()
			body: str
			body, _ = html_exporter.from_notebook_node(nb)
			with open(output_notebook, "w") as f:
				f.write(body)


"""
##     ##    ###    #### ##    ##
###   ###   ## ##    ##  ###   ##
#### ####  ##   ##   ##  ####  ##
## ### ## ##     ##  ##  ## ## ##
##     ## #########  ##  ##  ####
##     ## ##     ##  ##  ##   ###
##     ## ##     ## #### ##    ##
"""
# main
# ============================================================


def pdoc_combined(*modules, output_file: Path) -> None:
	"""Render the documentation for a list of modules into a single HTML file.

	Args:
		*modules: Paths or names of the modules to document.
		output_file: Path to the output HTML file.

	This function will:
	1. Extract all modules and submodules.
	2. Generate documentation for each module.
	3. Combine all module documentation into a single HTML file.
	4. Write the combined documentation to the specified output file.

	Rendering options can be configured by calling `pdoc.render.configure` in advance.

	"""
	# Extract all modules and submodules
	all_modules: Dict[str, pdoc.doc.Module] = {}
	for module_name in pdoc.extract.walk_specs(modules):
		all_modules[module_name] = pdoc.doc.Module.from_name(module_name)

	# Generate HTML content for each module
	module_contents: List[str] = []
	for module in all_modules.values():
		module_html = pdoc.render.html_module(module, all_modules)
		module_contents.append(module_html)

	# Combine all module contents
	combined_content = "\n".join(module_contents)

	# Write the combined content to the output file
	with output_file.open("w", encoding="utf-8") as f:
		f.write(combined_content)


def ignore_warnings() -> None:
	# Process and apply the warning filters
	for message in CONFIG.warnings_ignore:
		warnings.filterwarnings("ignore", message=message)


if __name__ == "__main__":
	# parse args
	# --------------------------------------------------
	argparser: argparse.ArgumentParser = argparse.ArgumentParser()
	argparser.add_argument(
		"--serve",
		"-s",
		action="store_true",
		help="Whether to start an HTTP server to serve the documentation",
	)
	argparser.add_argument(
		"--warn-all",
		"-w",
		action="store_true",
		help=f"Whether to show all warnings, instead of ignoring the ones specified in pyproject.toml:{TOOL_PATH}.warnings_ignore",
	)
	argparser.add_argument(
		"--combined",
		"-c",
		action="store_true",
		help="Whether to combine the documentation for multiple modules into a single markdown file",
	)
	parsed_args = argparser.parse_args()

	# configure pdoc
	# --------------------------------------------------
	# read what we need from the pyproject.toml, add stuff to pdoc globals
	set_global_config()

	# ignore warnings if needed
	if not parsed_args.warn_all:
		ignore_warnings()

	pdoc.render.configure(
		edit_url_map={
			CONFIG.package_name: CONFIG.package_code_url,
		},
		template_directory=(
			CONFIG.output_dir / "resources/templates/html/"
			if not parsed_args.combined
			else CONFIG.output_dir / "resources/templates/markdown/"
		),
		show_source=True,
		math=True,
		mermaid=True,
		search=True,
	)

	print(json.dumps(asdict(CONFIG), indent=2))

	# do the rendering
	# --------------------------------------------------
	if not parsed_args.combined:
		pdoc.pdoc(
			CONFIG.module_name,
			output_directory=CONFIG.output_dir,
		)
	else:
		use_markdown_format()
		pdoc_combined(
			CONFIG.module_name,
			output_file=CONFIG.output_dir / "combined" / f"{CONFIG.package_name}.md",
		)

	# convert notebooks if needed
	if CONFIG.notebooks_enabled:
		convert_notebooks()

	# http server if needed
	# --------------------------------------------------
	if parsed_args.serve:
		import http.server
		import os
		import socketserver

		port: int = 8000
		os.chdir(CONFIG.output_dir)
		with socketserver.TCPServer(
			("", port), http.server.SimpleHTTPRequestHandler
		) as httpd:
			print(f"Serving at http://localhost:{port}")
			httpd.serve_forever()
