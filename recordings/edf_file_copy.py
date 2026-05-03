from pathlib import Path
import shutil

# Root that contains folders like 0000045/ with 0000045.json inside
TARGET_ROOT = Path(r"D:\TobiiPro.SDK.Python.Windows_2.1.0.1\64\recordings")  # change this

# Root where EDFs exist under subdirs (e.g., normal/, abnormal/)
EDF_ROOT = Path(r"D:\NMT_events\NMT_events\edf\train")

def index_edf_files(edf_root: Path) -> dict[str, Path]:
    """
    Build a map: '0000045' -> full path to 0000045.edf
    If duplicates exist across subdirs, first one wins.
    """
    mapping = {}
    for p in edf_root.rglob("*.edf"):
        mapping.setdefault(p.stem, p)
    return mapping

def copy_matching_edfs(target_root: Path, edf_root: Path):
    edf_map = index_edf_files(edf_root)

    for folder in target_root.iterdir():
        if not folder.is_dir():
            continue

        stem = folder.name  # e.g., "0000045"
        src_edf = edf_map.get(stem)
        if not src_edf:
            print(f"EDF not found for {stem}")
            continue

        dst_edf = folder / f"{stem}.edf"

        # Avoid overwrite; add suffix if needed
        if dst_edf.exists():
            i = 1
            while True:
                alt = folder / f"{stem}_{i}.edf"
                if not alt.exists():
                    dst_edf = alt
                    break
                i += 1

        shutil.copy2(src_edf, dst_edf)
        print(f"Copied {src_edf} -> {dst_edf}")

if __name__ == "__main__":
    copy_matching_edfs(TARGET_ROOT, EDF_ROOT)
