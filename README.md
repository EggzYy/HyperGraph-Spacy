# WORK IN PROGRESS
# Medical Hypergraph NLP Pipeline

This project provides a robust pipeline for extracting, linking, and visualizing medical concepts from clinical text using spaCy, medspaCy, scispaCy, and hypergraph modeling. It is designed for advanced biomedical NLP tasks, including context detection, sectionization, and entity linking to UMLS.

## Features
- **Medical Entity Extraction**: Uses custom rules and spaCy models for accurate entity recognition.
- **Context & Section Detection**: Integrates medspaCy for context and section rule application.
- **Entity Linking**: Links recognized entities to UMLS concepts using scispaCy.
- **Hypergraph Construction**: Models relationships as hypergraphs for advanced analysis.
- **Visualization**: Visualizes hypergraphs with color-coded nodes and edges.

## Project Structure
- `main.py` – Example pipeline usage and entry point.
- `medical_hypergraph_pipeline.py` – Core pipeline logic.
- `visualizegemin.py` – Hypergraph reconstruction and visualization.
- `resources/` – Rule files for context, section, and sentence splitting.
- `results/` – Output data, hypergraph files, and visualizations.
- `input.txt` – Example input text.

## Setup
1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download spaCy/scispaCy models** (if not auto-downloaded):
   ```bash
   python -m spacy download en_core_sci_scibert
   ```

## Usage
Run the main pipeline:
```bash
python main.py
```

Visualize the hypergraph:
```bash
python visualizegemin.py
```

## Example
See [EXAMPLES.md](EXAMPLES.md) for sample input and output.

## License
See [LICENSE](LICENSE).

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md).

## Credits
- medspaCy, scispaCy, spaCy, HyperNetX, NetworkX
- Department of Biomedical Informatics, University of Utah (PyRuSH rules)
