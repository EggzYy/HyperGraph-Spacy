# Example Usage

## Input
See `input.txt` for a sample medical text about liver enzyme analysis.

## Running the Pipeline
```
python main.py
```

**Sample Output:**
- Entity Instances Found: ...
- Probabilistic Sentence Hints: ...
- Context Graph: ...
- Hypergraph: ...
- Hypergraph atoms saved to results/hg_atoms_data_prod.json
- Hypergraph round-trip load successful.

## Visualizing the Hypergraph
```
python visualizegemin.py
```

**Sample Output:**
- Hypergraph successfully saved to 'results/hypergraph_output_2.json'
- Full hypergraph visualization saved to 'results/hypergraph_visualization_2.png'

## Output Files
- `results/hg_atoms_data_prod.json`: Extracted hypergraph atoms
- `results/medical_hypergraph_2.gpickle`: Saved hypergraph object
- `results/hypergraph_visualization_2.png`: Hypergraph visualization

## Example Entity Atom (JSON)
```
{
  "instance_id": "inst_0",
  "canonical_id": "ent_1",
  "text": "Mechanical biliary obstruction",
  "label": "ENTITY",
  "start_char": 1,
  "end_char": 31,
  "sentence_text": "Mechanical biliary obstruction results in raised levels of ALP, GGT and often bilirubin.",
  "section_title": "UNCATEGORIZED",
  "umls_linking": [
    {
      "cui": "C0400979",
      "score": 0.784,
      "definition": "Blockage in the biliary tract..."
    }
  ],
  "contextual_modifiers": []
}
```
