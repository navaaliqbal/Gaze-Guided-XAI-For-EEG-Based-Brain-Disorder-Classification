from pathlib import Path

# Root that contains folders like 0000045/ with 0000045.edf inside
TARGET_ROOT = Path(r"D:\TobiiPro.SDK.Python.Windows_2.1.0.1\64\recordings")  # change this

# Root where EDFs exist under subdirs (normal/, abnormal/)
EDF_ROOT = Path(r"D:\NMT_events\NMT_events\edf\eval")

def build_label_index(edf_root: Path) -> dict[str, str]:
    """
    Map '0000045' -> 'normal' or 'abnormal'
    Assumes EDFs are located under ...\train\normal\*.edf or ...\train\abnormal\*.edf
    """
    labels = {}
    for p in edf_root.rglob("*.edf"):
        # parent folder name should be 'normal' or 'abnormal'
        label = p.parent.name.lower()
        labels.setdefault(p.stem, label)
    return labels

def report_folder_labels(target_root: Path, edf_root: Path):
    label_index = build_label_index(edf_root)

    for folder in sorted(target_root.iterdir()):
        if not folder.is_dir():
            continue

        stem = folder.name  # e.g., 0000045
        label = label_index.get(stem)

        edf_in_folder = list(folder.glob("*.edf"))
        has_edf = "yes" if edf_in_folder else "no"

        if label:
            print(f"{stem} -> {label} (edf_in_folder={has_edf})")
        else:
            print(f"{stem} -> NOT FOUND IN EDF ROOT (edf_in_folder={has_edf})")

if __name__ == "__main__":
    report_folder_labels(TARGET_ROOT, EDF_ROOT)
