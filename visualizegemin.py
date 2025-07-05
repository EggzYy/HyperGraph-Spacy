import json
import hypernetx as hnx
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Any, Union, Optional
from pathlib import Path
from collections import defaultdict

class Node:
    """Represents a canonical node (a concept or a modifier)."""
    def __init__(self, canonical_id, text, label):
        self.canonical_id = canonical_id
        self.text = text
        self.label = label
        self.mentions = []
        self.umls_cuis = set()
        self.global_modifiers = []

    def add_mention(self, instance_id, sentence_text, start_char, end_char, local_modifiers):
        self.mentions.append({
            "instance_id": instance_id,
            "sentence_text": sentence_text,
            "start_char": start_char,
            "end_char": end_char,
            "local_modifiers": local_modifiers
        })

    def add_umls_cui(self, cui):
        if cui:
            self.umls_cuis.add(cui)

    def add_global_modifier(self, modifier_info):
        self.global_modifiers.append(modifier_info)

    def __repr__(self):
        return f"Node(id='{self.canonical_id}', text='{self.text}', label='{self.label}')"

class Hyperedge:
    """Represents a hyperedge with a specific type."""
    def __init__(self, name, edge_type, nodes=None, text_content=""):
        self.name = name
        self.type = edge_type
        self.nodes = nodes if nodes else set()
        self.text_content = text_content

    def add_node(self, canonical_id):
        self.nodes.add(canonical_id)

    def __repr__(self):
        return f"Hyperedge(name='{self.name}', type='{self.type}', nodes={len(self.nodes)})"

class AtomHypergraph:
    """A robust class to fully rebuild and visualize the hypergraph from atom data."""
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.hyperedges: Dict[str, Hyperedge] = {}
        self._instance_to_canonical: Dict[str, str] = {}
        self._span_to_canonical: Dict[tuple[int, int], str] = {}

    def get_node(self, canonical_id: str) -> Optional[Node]:
        return self.nodes.get(canonical_id)

    def build_from_json_data(self, data: Dict[str, Any]):
        if not isinstance(data, dict):
            print("Error: Input data is not a dictionary.")
            return

        # Pass 1: Create ALL nodes (Entities AND Modifiers)
        context_graph = data.get("context_graph", {})
        for atom in data.get("entity_atoms", []):
            cid = atom.get("canonical_id")
            if cid and cid not in self.nodes:
                self.nodes[cid] = Node(cid, atom["text"], atom["label"])
            if atom.get("instance_id") and atom.get("start_char") is not None:
                self._instance_to_canonical[atom["instance_id"]] = cid
                self._span_to_canonical[(atom["start_char"], atom["end_char"])] = cid

        for mod in context_graph.get("modifiers", []):
            mod_id = mod.get("id")
            if mod_id and mod_id not in self.nodes:
                self.nodes[mod_id] = Node(mod_id, mod["text"], "MODIFIER")

        # Pass 2: Populate entity nodes with mentions and UMLS data
        for atom in data.get("entity_atoms", []):
            node = self.get_node(atom.get("canonical_id"))
            if node:
                node.add_mention(atom["instance_id"], atom["sentence_text"], atom["start_char"], atom["end_char"], atom.get("contextual_modifiers", []))
                for link in atom.get("umls_linking", []):
                    node.add_umls_cui(link.get("cui"))

        # Pass 3: Create ALL hyper-edges and populate global modifiers
        # 3a. Sentence edges
        for hint in data.get("sentence_relation_hints", []):
            node_ids = {self._instance_to_canonical.get(inst_id) for inst_id in hint.get("entities_involved_ids", [])}
            valid_node_ids = {nid for nid in node_ids if nid}
            if valid_node_ids:
                self.hyperedges[hint["id"]] = Hyperedge(hint["id"], "sentence", valid_node_ids, hint["sentence_text"])

        # 3b. Section edges
        section_to_ids = defaultdict(set)
        for atom in data.get("entity_atoms", []):
            section_title = atom.get("section_title", "UNCATEGORIZED")
            cid = atom.get("canonical_id")
            if section_title != "UNCATEGORIZED" and cid:
                section_to_ids[section_title].add(cid)
        for i, (title, ids) in enumerate(section_to_ids.items()):
            edge_name = f"sec_{i+1}"
            self.hyperedges[edge_name] = Hyperedge(edge_name, "section", ids, f"Section: {title}")
        
        # 3c. Contextual Links and Global Modifiers
        modifier_details = {mod['id']: mod for mod in context_graph.get("modifiers", [])}
        for i, edge in enumerate(context_graph.get("edges", [])):
            target_span_id, mod_id = edge.get("target_id"), edge.get("modifier_id")
            target_node_id = self._find_best_containing_span_from_id(target_span_id)
            if target_node_id and mod_id:
                edge_name = f"ctx_link_{i+1}"
                self.hyperedges[edge_name] = Hyperedge(edge_name, "contextual_link", {target_node_id, mod_id})
                target_node = self.get_node(target_node_id)
                mod_info = modifier_details.get(mod_id)
                if target_node and mod_info:
                    target_node.add_global_modifier(mod_info)

    def _find_best_containing_span_from_id(self, target_span_id: str) -> Optional[str]:
        if not target_span_id: return None
        try:
            target_start = int(target_span_id.split('_')[-2])
            target_end = int(target_span_id.split('_')[-1])
        except (IndexError, ValueError):
            return None
        
        best_match_id, min_len = None, float('inf')
        for (ent_start, ent_end), canonical_id in self._span_to_canonical.items():
            if ent_start <= target_start and ent_end >= target_end:
                length = ent_end - ent_start
                if length < min_len:
                    min_len, best_match_id = length, canonical_id
        return best_match_id

    def print_summary(self):
        print("Hypergraph Summary:")
        print(f"  Total Nodes Reconstructed: {len(self.nodes)}")
        print(f"  Total Hyperedges Reconstructed: {len(self.hyperedges)}")

    def save_to_json(self, filepath="hypergraph_processed.json"):
        """Saves the structured hypergraph data to a JSON file."""
        output_data = {"nodes": [], "hyperedges": []}
        for node_obj in self.nodes.values():
            node_dict = {
                "id": node_obj.canonical_id,
                "text": node_obj.text,
                "label": node_obj.label,
                "umls_cuis": sorted(list(node_obj.umls_cuis)),
                "mentions": node_obj.mentions,
                "global_modifiers": node_obj.global_modifiers
            }
            output_data["nodes"].append(node_dict)

        for edge_obj in self.hyperedges.values():
            edge_dict = {
                "name": edge_obj.name,
                "type": edge_obj.type,
                "nodes": sorted(list(edge_obj.nodes)),
                "text_content": edge_obj.text_content,
            }
            output_data["hyperedges"].append(edge_dict)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4)
            print(f"\nHypergraph successfully saved to '{filepath}'")
        except IOError as e:
            print(f"Error saving hypergraph to '{filepath}': {e}")

    def to_hypernetx_object(self):
        """Converts the internal hypergraph to a HyperNetX.Hypergraph object."""
        edge_dict = {edge.name: list(edge.nodes) for edge in self.hyperedges.values() if edge.nodes}
        if not edge_dict:
            return hnx.Hypergraph()

        H = hnx.Hypergraph(edge_dict)
        for nid, node_obj in self.nodes.items():
            if nid in H.nodes:
                H.nodes[nid].properties.update({'label': node_obj.label, 'text': node_obj.text, 'mentions': len(node_obj.mentions)})
        for eid, edge_obj in self.hyperedges.items():
            if eid in H.edges:
                H.edges[eid].properties.update({'type': edge_obj.type})
        return H

    
    def save_hypergraph(
        self,
        filepath: Union[str, Path] = "hypergraph.gpickle",
    ) -> None:
        """
        Persist the hypergraph so it can be restored later—even with a newer
        HyperNetX version.

        • Builds / refreshes the current HyperNetX object via
          `self.to_hypernetx_object()`
        • Converts it to a plain NetworkX bipartite incidence graph
        • Copies every node/edge property into that graph
        • Marks which vertices are *true* hyper-edges with the
          boolean attribute ``is_hyperedge``
        • Writes the result with ``nx.write_gpickle``

        Parameters
        ----------
        filepath : str | pathlib.Path
            Destination file. A “.gpickle” extension is conventional but not
            required.
        """
        H = self.to_hypernetx_object()               # up-to-date HNX graph

        if not H.nodes and not H.edges:
            print("Nothing to save – HyperNetX graph is empty.")
            return

        # 1️⃣  turn it into a classic incidence graph (bipartite)
        inc_graph = H.bipartite().copy()

        # 2️⃣  copy over node properties
        for u in H.nodes:
            props = dict(H.nodes[u].properties)
            if props:
                inc_graph.nodes[u]["node_props"] = props

        # 3️⃣  mark which vertices represent *hyper-edges*
        nx.set_node_attributes(inc_graph, False, "is_hyperedge")
        for e in H.edges:                       # iterate by edge-name
            inc_graph.nodes[e]["is_hyperedge"] = True
            eprops = dict(H.edges[e].properties)
            if eprops:
                # store edge-specific metadata on the same vertex
                inc_graph.nodes[e]["edge_props"] = eprops

        # 4️⃣  make sure the directory exists and write the file
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        nx.write_gpickle(inc_graph, filepath)

        print(f"✔ Hypergraph written to '{filepath.resolve()}'")

    def visualize(self, output_filename="gemin/hypergraph_visualization.png"):
        """Visualizes the full hypergraph with color-coded edge types."""
        H = self.to_hypernetx_object()
        if not H.nodes:
            print("Hypergraph is empty. Skipping visualization.")
            return

        plt.figure(figsize=(24, 20))
        bipartite_graph = H.bipartite()
        pos = nx.spring_layout(bipartite_graph, k=0.3, iterations=50, seed=42)

        # --- Styling based on node and edge properties ---
        node_radii = {nid: 0.25 + (props.get('mentions', 0) * 0.05) for nid, props in H.nodes.properties.items()}
        node_labels = {nid: f"{props.get('text', '')}\n({props.get('mentions', 0)})" if props.get('label') != 'MODIFIER' else props.get('text', '') for nid, props in H.nodes.properties.items()}
        
        # <-- FIX: Changed from H.nodes.properties.values() to H.nodes.properties.items()
        node_colors = ['lightcoral' if props.get('label') == 'MODIFIER' else 'skyblue' for _, props in H.nodes.properties.items()]

        # Color edges by type
        edge_color_map = {'sentence': 'lightgrey', 'section': 'lightblue', 'contextual_link': 'palegreen'}
        ordered_edge_ids = list(H.edges)
        edge_facecolors = [edge_color_map.get(H.edges[eid].properties.get('type', 'sentence'), 'lightgrey') for eid in ordered_edge_ids]

        hnx.draw(
            H,
            pos=pos,
            with_node_labels=True,
            node_labels=node_labels,
            node_radius=node_radii,
            nodes_kwargs={'facecolors': node_colors, 'edgecolors': 'black', 'linewidths': 0.5},
            node_labels_kwargs={'fontsize': 9, 'fontweight': 'bold'},
            edges_kwargs={'facecolors': edge_facecolors, 'edgecolors': 'gray', 'alpha': 0.7}
        )
        
        plt.title("Full Medical Hypergraph (Color-coded Edges, Size by Mentions)", fontsize=18)
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"\nFull hypergraph visualization saved to '{output_filename}'")
        plt.close()



def load_json_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file '{filepath}': {e}")
        return None

if __name__ == "__main__":
    json_file_path = "results/hg_atoms_data_prod.json" 
    data = load_json_file(json_file_path)

    if data:
        hg = AtomHypergraph()
        hg.build_from_json_data(data)
        hg.save_hypergraph("results/medical_hypergraph_2.gpickle")
        hg.print_summary()
        hg.save_to_json("results/hypergraph_output_2.json")
        hg.visualize("results/hypergraph_visualization_2.png")
    else:
        print("Could not load data, hypergraph processing aborted.")
