"""Theme presets controlling colors and fonts in the rendered graph.

A theme is a flat dict of role -> color/font value. The visualizer reads roles
by name (`theme["bg"]`, `theme["key_input"]`, ...) so adding a new key here
requires the visualizer to opt in. To add a new full theme, copy LIGHT and
edit values; to tweak one role, pass a dict override:

    theme = {**THEMES["light"], "key_input": "#abcdef"}
    visualize(model, theme=theme)
"""

from copy import deepcopy
from typing import Dict


LIGHT: Dict[str, str] = {
    "bg": "white",
    "module_fill": "#f8f9fa",
    "module_border": "#c4c4c4",
    "module_text": "#2d2d2d",
    "key_input": "#d4edda",
    "key_intermediate": "#e8daef",
    "key_output": "#d1ecf1",
    "key_state": "#fff3cd",          # recurrent / state keys (LSTM, GRU)
    "key_text": "#2d2d2d",
    "key_border": "#888888",
    "edge_internal": "#888888",
    "edge_key": "#6c757d",
    "edge_state": "#d4a017",         # dashed edge for state keys
    "cluster_border": "#dee2e6",
    "cluster_fill": "#ffffff",
    "cluster_inner_fill": "#f1f3f5",
    "probabilistic_fill": "#fde2e4",  # ProbabilisticTensorDictModule
    "probabilistic_border": "#d62828",
    "legend_fill": "#ffffff",
    "legend_border": "#c4c4c4",
    "font": "Helvetica",
}

DARK: Dict[str, str] = {
    "bg": "#1e1e2e",
    "module_fill": "#313244",
    "module_border": "#585b70",
    "module_text": "#cdd6f4",
    "key_input": "#a6e3a1",
    "key_intermediate": "#cba6f7",
    "key_output": "#89b4fa",
    "key_state": "#f9e2af",
    "key_text": "#1e1e2e",
    "key_border": "#9399b2",
    "edge_internal": "#6c7086",
    "edge_key": "#9399b2",
    "edge_state": "#f9e2af",
    "cluster_border": "#45475a",
    "cluster_fill": "#181825",
    "cluster_inner_fill": "#313244",
    "probabilistic_fill": "#f38ba8",
    "probabilistic_border": "#f38ba8",
    "legend_fill": "#181825",
    "legend_border": "#45475a",
    "font": "Helvetica",
}

PRINT: Dict[str, str] = {
    # High-contrast, monochrome-friendly. Roles are distinguished by border
    # weight and fill intensity rather than hue, so this prints cleanly.
    "bg": "white",
    "module_fill": "white",
    "module_border": "black",
    "module_text": "black",
    "key_input": "white",
    "key_intermediate": "#e0e0e0",
    "key_output": "#999999",
    "key_state": "#cccccc",
    "key_text": "black",
    "key_border": "black",
    "edge_internal": "black",
    "edge_key": "black",
    "edge_state": "black",
    "cluster_border": "black",
    "cluster_fill": "white",
    "cluster_inner_fill": "#f5f5f5",
    "probabilistic_fill": "white",
    "probabilistic_border": "black",
    "legend_fill": "white",
    "legend_border": "black",
    "font": "Helvetica",
}

THEMES: Dict[str, Dict[str, str]] = {
    "light": LIGHT,
    "dark": DARK,
    "print": PRINT,
}


def resolve_theme(theme) -> Dict[str, str]:
    """Return a fresh theme dict from a name or dict.

    Accepts a string preset name ("light", "dark", "print") or a partial dict
    that overrides on top of "light". Returns a deep copy so callers cannot
    mutate the preset.
    """
    if theme is None:
        return deepcopy(LIGHT)
    if isinstance(theme, str):
        if theme not in THEMES:
            raise ValueError(
                f"Unknown theme '{theme}'. Available: {sorted(THEMES)}"
            )
        return deepcopy(THEMES[theme])
    if isinstance(theme, dict):
        merged = deepcopy(LIGHT)
        merged.update(theme)
        return merged
    raise TypeError(f"theme must be str or dict, got {type(theme).__name__}")
