import networkx as nx, json, math
from pathlib import Path
from pyvis.network import Network
from typing import Dict, List, Any, Union

def visualise_hnx_with_pyvis(
        gpickle_path: Union[str, Path],
        outfile: str = "hypergraph.html",
        *,
        collapse: bool = False,               # True ⇒ replace orange diamonds with cliques
        notebook: bool = False,               # True ⇒ embed in Jupyter / Streamlit
        physics: bool = True,                 # False ⇒ fix positions (helps huge graphs)
        large_graph_threshold: int = 500,     # auto-tweak when size exceeds this
        height: str = "850px",
        width: str  = "100%",
):
    """
    Render the saved incidence graph produced by `save_hypergraph`
    as an interactive PyVis HTML visual.

    Parameters
    ----------
    collapse : bool
        If True, hyper-edge nodes are *not* shown; instead every entity pair
        inside a sentence is connected directly (clique expansion).
    notebook : bool
        Use PyVis' notebook mode so the HTML appears inline in Jupyter /
        Streamlit.  (File is still written to `outfile`.)
    physics : bool
        Toggle the Barnes-Hut physics engine.
    large_graph_threshold : int
        When the total number of nodes exceeds this value, physics is disabled
        and hierarchical repulsion is applied for readability.
    """

    def _certainty_to_hex(cert: float) -> str:
        """
        Map certainty 0.0-1.0  →  red (#F44336) .. yellow .. green (#4CAF50).
        """
        cert = max(0.0, min(1.0, cert))             # clamp
        # simple HSV: H = 0 (red) .. 120 (green)
        h = 120 * cert
        # convert HSV→RGB
        c = 1; x = 1 - abs((h / 60) % 2 - 1); m = 0
        if   h < 60:   r, g, b = c, x, 0
        elif h < 120:  r, g, b = x, c, 0
        else:          r, g, b = 0, c, x            # should not happen
        r, g, b = [(v + m) * 255 for v in (r, g, b)]
        return f"#{int(r):02X}{int(g):02X}{int(b):02X}"

    # ── load incidence graph ────────────────────────────────────────────────
    gpickle_path = Path(gpickle_path)
    inc_g = nx.read_gpickle(str(gpickle_path))

    # ── configure PyVis ─────────────────────────────────────────────────────
    net = Network(height=height, width=width,
                  bgcolor="#ffffff", directed=False, notebook=notebook)
    if physics:
        net.barnes_hut()
    else:
        net.toggle_physics(False)

    net.show_buttons(filter_=["physics"])           # user can tweak live

    # disable physics automatically for very large graphs
    if len(inc_g) > large_graph_threshold:
        net.toggle_physics(False)
        net.hierarchical_repulsion()

    # ── add nodes ───────────────────────────────────────────────────────────
    is_edge = nx.get_node_attributes(inc_g, "is_hyperedge")

    for nid, attrs in inc_g.nodes(data=True):
        if is_edge.get(nid):                        # hyper-edge
            edge_info  = attrs.get("edge_attrs", {})
            cert       = edge_info.get("certainty", 1.0)
            colour     = _certainty_to_hex(cert)
            level      = edge_info.get("level")
            title_html = f"<b>level:</b> {level}<br><b>certainty:</b> {cert}<br>{json.dumps(edge_info, indent=2)}"

            # Let's make different hyperedge types visually distinct
            shape = "diamond"
            size = 28
            label = "" # No label for most hyperedges to reduce clutter

            if level == "contextual_link":
                shape = "star"
                size = 15
            elif level == "section":
                shape = "square"
                # For sections, showing the title can be useful
                label = (edge_info.get("text", "")[:30] + "…") if edge_info.get("text") else ""


            if not collapse or level == "section":
                net.add_node(
                    nid, label=label, shape=shape, size=size,
                    color=colour, title=title_html
                )
        else:                                      # entity or modifier node
            # FIX: Pull attributes from the 'node_attrs' dictionary where they are stored.
            node_info = attrs.get("node_attrs", {})
            
            # Use the actual text for the label, not the internal ID.
            label_text = node_info.get("text", nid)
            node_label = node_info.get("label")

            # Customize appearance based on the node's role.
            color = "#1976D2"  # Blue for entities
            shape = "dot"
            if node_label == "MODIFIER":
                color = "#4DD0E1"  # Cyan for modifiers
                shape = "ellipse"

            net.add_node(
                nid, label=label_text, shape=shape, size=16,
                color=color, title=json.dumps(node_info, indent=2)
            )

    # ── add edges ───────────────────────────────────────────────────────────
    if collapse:
        # connect every entity pair that co-occurs in the same sentence
        for hn in [n for n, flag in is_edge.items() if flag]:
            neigh = list(inc_g.neighbors(hn))
            for i in range(len(neigh)):
                for j in range(i + 1, len(neigh)):
                    net.add_edge(neigh[i], neigh[j],
                                 title=f"via {hn}",
                                 physics=physics)
    else:
        for u, v, e_attrs in inc_g.edges(data=True):
            cert = e_attrs.get("certainty", 1.0)
            colour = _certainty_to_hex(cert)
            net.add_edge(u, v, color=colour,
                         title=json.dumps(e_attrs, indent=2),
                         physics=physics)

    # ── output ──────────────────────────────────────────────────────────────
    outfile = Path(outfile)
    net.write_html(str(outfile))
    print(f"✓ Interactive visual saved → {outfile.resolve()}")

    if notebook:
        # Return the HTML for embedding
        from IPython.display import IFrame, display
        display(IFrame(src=str(outfile), width="100%", height="850px"))

visualise_hnx_with_pyvis("results/med_ner_hypergraph_prod.gpickle")