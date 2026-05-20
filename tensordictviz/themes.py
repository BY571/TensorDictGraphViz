"""Theme presets — colors, fonts, and structural attributes.

A theme is a flat dict. Beyond colors it now also carries *structural*
attributes (node shapes, edge routing, corner style, layout direction,
spacing) so templates can look genuinely different, not merely recolored.

``resolve_theme`` merges every preset and dict-override on top of LIGHT,
so a partial theme only needs to specify what differs:

    visualize(model, theme="blueprint")
    visualize(model, theme={"bg": "#fffaf0"})        # tweak one role
    visualize(model, theme={"module_rounded": False}) # tweak structure

Structural keys
    rankdir         layout direction ("TB" | "LR" | "BT" | "RL")
    splines         edge routing ("ortho" | "spline" | "polyline" | "curved")
    key_shape       node shape for key nodes ("ellipse" | "box" | ...)
    module_shape    node shape for module nodes
    module_rounded  True -> rounded module corners, False -> sharp
    nodesep         min space between sibling nodes (inches, as str)
    ranksep         min space between ranks (inches, as str)
"""

from copy import deepcopy
from typing import Dict


# LIGHT is the complete reference theme. Every other theme is merged on top
# of it, so presets below only need to list what they change.
LIGHT: Dict[str, str] = {
    # structure
    "rankdir": "TB",
    "splines": "ortho",
    "key_shape": "ellipse",
    "module_shape": "box",
    "module_rounded": True,
    "nodesep": "0.28",
    "ranksep": "0.5",
    # canvas
    "bg": "white",
    "font": "Helvetica",
    # modules
    "module_fill": "#f5f6f8",
    "module_border": "#aab2bd",
    "module_text": "#1f2933",
    # key roles — fill + a tinted (darker) border of the same hue
    "key_input": "#d4ecd9",
    "key_input_border": "#5c9469",
    "key_intermediate": "#e7dcf3",
    "key_intermediate_border": "#8a6cb0",
    "key_output": "#d0e6f2",
    "key_output_border": "#5a8fa8",
    "key_state": "#fbe7c6",
    "key_state_border": "#c79330",
    "key_text": "#1f2933",
    # edges
    "edge_internal": "#9aa5b1",
    "edge_key": "#6b7280",
    "edge_state": "#c79330",
    "edge_penwidth": "1.5",
    # clusters
    "cluster_border": "#e1e4e8",
    "cluster_fill": "#ffffff",
    "cluster_inner_fill": "#f1f3f5",
    # probabilistic module highlight
    "probabilistic_fill": "#fbe0e3",
    "probabilistic_border": "#c0392b",
    # legend
    "legend_fill": "#ffffff",
    "legend_border": "#d8dce1",
}

DARK: Dict[str, str] = {
    "bg": "#1e1e2e",
    "module_fill": "#313244",
    "module_border": "#585b70",
    "module_text": "#cdd6f4",
    "key_input": "#a6e3a1",
    "key_input_border": "#74c478",
    "key_intermediate": "#cba6f7",
    "key_intermediate_border": "#a684d4",
    "key_output": "#89b4fa",
    "key_output_border": "#6a93d8",
    "key_state": "#f9e2af",
    "key_state_border": "#d4be7a",
    "key_text": "#1e1e2e",
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
}

PRINT: Dict[str, str] = {
    # Monochrome-friendly: roles differ by fill intensity, not hue.
    "bg": "white",
    "module_fill": "white",
    "module_border": "black",
    "module_text": "black",
    "key_input": "white",
    "key_input_border": "black",
    "key_intermediate": "#e0e0e0",
    "key_intermediate_border": "black",
    "key_output": "#999999",
    "key_output_border": "black",
    "key_state": "#cccccc",
    "key_state_border": "black",
    "key_text": "black",
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
}

BLUEPRINT: Dict[str, str] = {
    # Engineering-drawing look: cyan line-art on navy, monospace, sharp corners.
    "splines": "ortho",
    "key_shape": "box",
    "module_shape": "box",
    "module_rounded": False,
    "bg": "#0d2137",
    "font": "Courier New",
    "module_fill": "#14365a",
    "module_border": "#6cc5e8",
    "module_text": "#cfe9f7",
    # All key fills share one dark tone; the role is read from the border color.
    "key_input": "#14365a",
    "key_input_border": "#7ad9a4",
    "key_intermediate": "#14365a",
    "key_intermediate_border": "#c4a3e8",
    "key_output": "#14365a",
    "key_output_border": "#6cc5e8",
    "key_state": "#14365a",
    "key_state_border": "#f0c674",
    "key_text": "#dcefff",
    "edge_internal": "#8fd4f2",
    "edge_key": "#5ce1ff",
    "edge_state": "#ffd97a",
    "edge_penwidth": "2.6",
    "cluster_border": "#3a5f80",
    "cluster_fill": "#0d2137",
    "cluster_inner_fill": "#14365a",
    "probabilistic_fill": "#14365a",
    "probabilistic_border": "#ff7b7b",
    "legend_fill": "#0d2137",
    "legend_border": "#3a5f80",
}

EDITORIAL: Dict[str, str] = {
    # Minimal, near-monochrome with a single indigo accent and generous air.
    "splines": "spline",
    "nodesep": "0.4",
    "ranksep": "0.7",
    "bg": "#ffffff",
    "module_fill": "#ffffff",
    "module_border": "#d4d4d8",
    "module_text": "#27272a",
    # Roles read from grey intensity; the indigo accent is reserved for the
    # final output key and the probabilistic module — the things that matter.
    "key_input": "#f4f4f5",
    "key_input_border": "#a1a1aa",
    "key_intermediate": "#e4e4e7",
    "key_intermediate_border": "#71717a",
    "key_output": "#e0e7ff",
    "key_output_border": "#4f46e5",
    "key_state": "#fef3c7",
    "key_state_border": "#b45309",
    "key_text": "#27272a",
    "edge_internal": "#d4d4d8",
    "edge_key": "#a1a1aa",
    "edge_state": "#b45309",
    "cluster_border": "#ececee",
    "cluster_fill": "#ffffff",
    "cluster_inner_fill": "#fafafa",
    "probabilistic_fill": "#ffffff",
    "probabilistic_border": "#4f46e5",
    "legend_fill": "#ffffff",
    "legend_border": "#ececee",
}

VIVID: Dict[str, str] = {
    # Bold flat saturated fills with strong matching borders — modern, tidy.
    "bg": "#ffffff",
    "module_fill": "#eef2f7",
    "module_border": "#94a3b8",
    "module_text": "#0f172a",
    "key_input": "#34d399",
    "key_input_border": "#059669",
    "key_intermediate": "#a78bfa",
    "key_intermediate_border": "#7c3aed",
    "key_output": "#38bdf8",
    "key_output_border": "#0284c7",
    "key_state": "#fbbf24",
    "key_state_border": "#d97706",
    "key_text": "#0f172a",
    "edge_internal": "#94a3b8",
    "edge_key": "#475569",
    "edge_state": "#d97706",
    "cluster_border": "#cbd5e1",
    "cluster_fill": "#ffffff",
    "cluster_inner_fill": "#f8fafc",
    "probabilistic_fill": "#fb7185",
    "probabilistic_border": "#e11d48",
    "legend_fill": "#ffffff",
    "legend_border": "#cbd5e1",
}

THEMES: Dict[str, Dict[str, str]] = {
    "light": LIGHT,
    "dark": DARK,
    "print": PRINT,
    "blueprint": BLUEPRINT,
    "editorial": EDITORIAL,
    "vivid": VIVID,
}


def resolve_theme(theme) -> Dict[str, str]:
    """Return a complete theme dict from a name or partial dict.

    Every result is LIGHT with the requested preset / overrides merged on
    top, so a theme is always complete and presets can be partial. Returns
    a deep copy — callers cannot mutate a preset.
    """
    merged = deepcopy(LIGHT)
    if theme is None:
        return merged
    if isinstance(theme, str):
        if theme not in THEMES:
            raise ValueError(
                f"Unknown theme '{theme}'. Available: {sorted(THEMES)}"
            )
        merged.update(THEMES[theme])
        return merged
    if isinstance(theme, dict):
        merged.update(theme)
        return merged
    raise TypeError(f"theme must be str or dict, got {type(theme).__name__}")
