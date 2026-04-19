"""Jupyter widget: interactive browser of a FieldML document's evaluator graph."""

from __future__ import annotations

from typing import Any


def explore(doc: Any) -> Any:
    """Return an ipywidgets layout displaying the evaluator graph.

    Three panes:
      - Left: SelectMultiple of evaluator names.
      - Center: ipycytoscape graph with edges from ReferenceEvaluators to their sources.
      - Right: HTML detail for the currently-selected node.

    Raises ImportError if [viz] extra isn't installed.
    """
    try:
        import ipywidgets as widgets
        from ipycytoscape import CytoscapeWidget
    except ImportError as exc:
        raise ImportError(
            "doc.explore() requires the [viz] extra: pip install pyfieldml[viz]"
        ) from exc

    from pyfieldml.model.evaluators import ReferenceEvaluator

    region = doc.region

    # Build ipycytoscape node/edge lists. Nodes carry a `kind` for styling.
    nodes = [
        {"data": {"id": name, "label": name, "kind": type(ev).__name__}}
        for name, ev in region.evaluators.items()
    ]
    edges: list[dict[str, Any]] = []
    for name, ev in region.evaluators.items():
        if isinstance(ev, ReferenceEvaluator) and ev.source is not None:
            edges.append({"data": {"source": name, "target": ev.source.name}})

    graph = CytoscapeWidget()
    graph.graph.add_graph_from_json({"nodes": nodes, "edges": edges})

    tree = widgets.SelectMultiple(
        options=list(region.evaluators.keys()),
        description="Evaluators:",
        rows=12,
    )
    detail = widgets.HTML("<i>Select an evaluator to see details.</i>")

    def _on_select(change: dict[str, Any]) -> None:
        selected = change.get("new") or ()
        if not selected:
            detail.value = "<i>No selection.</i>"
            return
        name = selected[0]
        ev = region.evaluators[name]
        value_type = getattr(ev.value_type, "name", "—")
        detail.value = (
            f"<b>{name}</b><br>"
            f"Kind: <code>{type(ev).__name__}</code><br>"
            f"Value type: <code>{value_type}</code>"
        )

    tree.observe(_on_select, names="value")
    return widgets.HBox([tree, graph, detail])
