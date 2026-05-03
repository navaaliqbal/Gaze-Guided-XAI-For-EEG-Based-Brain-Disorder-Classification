# Sequence diagram rendering

File: `docs/diagrams/load_eeg_heatmap.puml`

To render locally (requires PlantUML + Graphviz):

1. Install Graphviz and Java.
2. Install PlantUML (jar) or use the VSCode PlantUML extension.

Command-line (with plantuml.jar):

```bash
java -jar plantuml.jar load_eeg_heatmap.puml
```

This produces `load_eeg_heatmap.png` in the same folder.

Alternatively, open the file in VSCode with the "PlantUML" extension and click preview → export.

Notes:
- The `.puml` file models the "Load EEG + Heatmap" flow described in the SDD. Modify participants or messages as needed.
