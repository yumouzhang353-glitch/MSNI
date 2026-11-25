EPS = 1e-12
PARAMETERS = {
    "continuity_parameters": {
        "maximum_effective_dimension": 1.5,
        "minimum_nb": 2,
        "fit_quantile": 0.95,
        "opposing_cosine_maximum": -0.5,
        "minimum_density_ratio": 0.5,
        "density_rank_factors": [1, 2, 4, 8],
    },
    "fragment_ratio": 0.2,
    "anchor_dimension_limit": 1.25,
    "field_parameters": {"C1": 0.5, "C3": 1.0, "C5": 1.0},
    "threshold_versions": {
        "T1": {"row": "f0", "distance": "tc_vs_tc_quantile"},
        "T2": {"row": "c1", "distance": "tc_vs_t0_quantile"},
        "T3": {"row": "f0", "distance": "tc_vs_t0_quantile"},
        "T4": {"row": "geometric_blend", "distance": "tc_vs_mean_quantile"},
        "T5": {"row": "c1", "distance": "t0_vs_t0_quantile"},
        "T6": {"row": "c1", "distance": "mean_relative_distance_le_one"},
        "T7": {"row": "geometric_blend", "distance": "tc_vs_tc_quantile"},
        "T8": {"row": "c1", "distance": "tc_vs_mean_quantile"},
        "T9": {"row": "c1", "distance": "both_native_distance_predicates"},
    },
    "neighbor_geometry": {
        "query_excluded": True,
        "ranks": ["max(3,r)", "max(3,2r)"],
        "weighted": "original reverse-neighbor reliability",
        "ridge_largest_eigenvalue_fraction": 0.1,
        "spectral_tolerance": 1e-10,
        "normalization": "trace/effective_dimension",
    },
    "fixed_defaults": True,
    "natural_neighbor_round_symbol": "r",
}
PROTOCOL = {"neighbor_geometry": PARAMETERS["neighbor_geometry"]}
