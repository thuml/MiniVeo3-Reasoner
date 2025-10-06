"""`DEFAULT_GENERATORS` is a list of generator name, generator kwargs pairs used in tests and demos"""

DEFAULT_GENERATORS: list[tuple[str, dict]] = [
	("gen_dfs", dict()),
	("gen_dfs", dict(do_forks=False)),
	("gen_dfs", dict(accessible_cells=20)),
	("gen_dfs", dict(max_tree_depth=0.5)),
	("gen_wilson", dict()),
	# ("gen_percolation", dict(p=0.1)),
	(
		"gen_percolation",
		dict(p=1.0),
	),  # anything less than this and tests will stochastically fail
	("gen_dfs_percolation", dict(p=0.1)),
	("gen_dfs_percolation", dict(p=0.4)),
	# ("gen_prim", dict()),
	# ("gen_prim", dict(do_forks=False)),
	# ("gen_prim", dict(accessible_cells=0.5)),
	# ("gen_prim", dict(max_tree_depth=0.5)),
	# ("gen_prim", dict(accessible_cells=0.5, max_tree_depth=0.5)),
	("gen_kruskal", dict()),
	("gen_recursive_division", dict()),
]
