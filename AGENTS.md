# AGENTS.md

This repository uses the following conventions for maze dataset generation. Follow these defaults unless a task explicitly states otherwise.

## Maze Generation Standards

- `cs` means `const speed`.
  - When a dataset/script name includes `cs`, pass `--const-speed` to `data/maze/maze_generator.py`.
  - When it does not include `cs`, do not pass `--const-speed`.

- Default path-length range:
  - `min-path-length = 2`
  - `max-path-length = 2 * grid_n`
  - If a task needs a different range, that exception must be stated explicitly in the script name, script comments, or the task description.

- Default sample count:
  - `n-mazes = 100`
  - If a task needs a different count, that exception must be stated explicitly.

- Resolution standard:
  - `4x4 -> --fig-size-pixel 128`
  - `8x8 -> --fig-size-pixel 256`
  - If another grid size is used, do not assume silently; state the intended resolution explicitly.

## Script Naming Guidance

- Use clear names that expose exceptions.
  - Examples:
    - `generate4.sh`: standard 4x4 generation
    - `generate4cs.sh`: 4x4 with constant speed
    - `generate4hard.sh`: 4x4 with non-default path-length range

## Exception Policy

When a script or task deviates from the defaults above, document the deviation in at least one of these places:

- the script filename
- a top-of-file comment in the script
- the task description / run command

## Output Expectations

- Maze samples should be exported as paired files:
  - `*.mp4`
  - `*_00.png`
- Keep filename prefixes consistent with the grid size unless a task explicitly requires another prefix.
