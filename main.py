import json, pathlib
from medical_hypergraph_pipeline import MedicalHypergraphPipeline

if __name__ == "__main__":
    # --- Configuration ---
    # Centralize all configurable paths and parameters
    CONFIG = {
        "model_name": "en_core_sci_scibert",
        "target_rules_path": "results/target_rules_prod.json",
        "context_rules_path": "resources/context_rules.json",
        "section_rules_path": "resources/section_rules.json",
        "rush_rules_path": "resources/rush_rules.tsv",
        "hypergraph_atoms_path": "results/hg_atoms_data_prod.json",
        "hypergraph_gpickle_path": "results/med_ner_hypergraph_prod.gpickle",
        "linker_config": {
            "resolve_abbreviations": True,
            "linker_name": "umls",
            "threshold": 0.7,
        }
    }

    # --- Sample Text ---
    medical_text = """
Mechanical biliary obstruction results in raised levels of ALP, GGT and often bilirubin.
ALP will usually be markedly raised in comparison with ALT.s
Levels of ALP and GGT elevated in similar proportions signify a hepatobiliary source.
Otherwise alternative causes of single enzyme elevation should be considered.
When due to choledocholithiasis, the levels of ALP and GGT tend to fluctuate (in comparison to stricture forming disease) and may be associated with a normal bilirubin.
Enzyme titres tend to rise and fall gradually and may be preceded by a peaked rise in liver transaminases which can reach > 1000I/U.
The AST:ALT ratio (De Ritis ratio) may assist in differentiating the site of biliary obstruction.
When associated with a cholestatic picture, an AST:ALT ratio of <1.5 suggests an extrahepatic obstruction.
In such circumstances the ALT titre is frequently considerably higher than AST.
An AST:ALT ratio of > 1.5 indicates intrahepatic (mechanical or medical) cholestasis is more likely.
Drug-induced cholestasis usually presents with a preferential rise in ALP, rather than GGT, or with an ALT:ALP ratio of<2.
Causative drugs would include: antibiotics, immunosuppressants, tricyclic antidepressants and angiotensin converting enzyme inhibitors.
In Primary Biliary Cirrhosis, an autoimmune condition of the intrahepatic biliary ducts, the level of ALP is generally greater than that of GGT.
In this case, transaminases are invariably normal or only minimally elevated.
If either of these two criteria is absent, imaging and liver biopsy become necessary.
AST and ALP are used within some scoring criteria to monitor the effects of ursodeoxycholic acid in the management of PBC.
A recent study has shown that a raised AST:ALT ratio outperforms other non-histological indicators of cirrhosis in PBC, but still only achieves a low sensitivity and a specificity of 65-79%.
As with PBC, liver enzymes play a key role in the diagnosis of Primary Sclerosing Cholangitis (PSC).
Alcohol induces hepatic enzymes leading to a raised GGT with an ALP which may be normal, or disproportionately lower than the GGT.
A GGT: ALP ratio > 2.5 in association with jaundice suggests alcohol as a cause of liver disease.
The presence of a macrocytosis, due to either an associated dietary deficiency of folate or B12, or due to a direct suppression of bone marrow by alcohol is supportive of the diagnosis of alcoholic liver disease.
A raised GGT is not diagnostic of alcohol abuse, with research showing it remains high in former drinkers as well as current drinkers.
    """

    # --- 1. Initialize the Pipeline ---
    print("⇢ Initializing the medical hypergraph pipeline...")
    pipeline = MedicalHypergraphPipeline(CONFIG)
    
    # --- 2. Process Text and Build Hypergraph ---
    print("\n⇢ Processing text and building hypergraph...")
    H, hypergraph_data = pipeline.process_text(medical_text)

    # --- 3. Report and Save Results ---
    print("\n--- Extraction Summary ---")
    
    # --- PRIMARY FIX: Use the new 'entity_instance_atoms' key. ---
    print(f"  • Entity Instances Found: {len(hypergraph_data['entity_atoms'])}")
    
    print(f"  • Probabilistic Sentence Hints: {len(hypergraph_data['sentence_relation_hints'])}")
    cg_data = hypergraph_data['context_graph']
    print(f"  • Context Graph: {len(cg_data['targets'])} targets, {len(cg_data['modifiers'])} modifiers, {len(cg_data['edges'])} edges")
    print(f"  • Hypergraph: {len(H.nodes)} nodes, {len(H.edges)} hyper-edges")

    atoms_path = pathlib.Path(CONFIG["hypergraph_atoms_path"])
    with atoms_path.open("w", encoding="utf8") as f:
        json.dump(hypergraph_data, f, indent=2)
    print(f"\n✓ Hypergraph atoms saved to {atoms_path.resolve()}")

    db_path = pathlib.Path(CONFIG["hypergraph_gpickle_path"])
    pipeline.save_hypergraph(H, db_path)

    # --- 4. Demonstrate Persistence ---
    print("\n⇢ Demonstrating persistence...")
    H2 = pipeline.load_hypergraph(db_path)
    assert len(H2.edges) == len(H.edges)
    print("✓ Hypergraph round-trip load successful.")

    pipeline.save_rules()

    # --- 5. Example Queries (Now More Robust) ---
    print("\n--- Example Queries ---")
    print("Uncertain Sentences (certainty = 0.5):")
    found_uncertain = False
    
    for edge_id in H.edges:
        edge = H.edges[edge_id]
        if edge.attrs and edge.attrs.get("level") == "sentence" and edge.attrs.get("certainty") == 0.5:
            print(f"  - {edge.attrs.get('text')}")
            found_uncertain = True
    if not found_uncertain:
        print("  - None")
            
    section_edges = [
        H.edges[edge_id] for edge_id in H.edges 
        if H.edges[edge_id].attrs and H.edges[edge_id].attrs.get("level") == "section"
    ]

    if section_edges:
        # Sort to make the output deterministic
        for section_edge in sorted(section_edges, key=lambda e: e.uid):
            section_name = section_edge.attrs.get('text', 'Unknown Section')
            print(f"\n--- Entities in '{section_name}' ---")
            # Sort nodes by their text attribute for deterministic output
            nodes = [H.nodes[node_id] for node_id in section_edge.elements]
            for node in sorted(nodes, key=lambda n: n.attrs.get('text', '')):
                node_attrs = node.attrs
                print(f"  - {node_attrs.get('text')} (Label: {node_attrs.get('label')})")
    else:
        print("\n--- No semantic sections found in the text ---")