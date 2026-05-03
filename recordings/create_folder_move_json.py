from pathlib import Path
import shutil
# Directory containing the renamed JSON files (e.g., 0000045.json)
DIR = Path(r"D:\TobiiPro.SDK.Python.Windows_2.1.0.1\64\recordings")  # change this

def create_folders_from_json_names(directory: Path):
    for json_path in directory.glob("*.json"):
        folder = directory / json_path.stem  # "0000045.json" -> folder "0000045"
        folder.mkdir(exist_ok=True)

def move_jsons_into_named_folders(directory: Path):
    for json_path in directory.glob("*.json"):
        target_dir = directory / json_path.stem  # folder "0000045"
        target_dir.mkdir(exist_ok=True)

        target_path = target_dir / json_path.name

        # Handle collision inside target folder
        if target_path.exists():
            i = 1
            while True:
                alt = target_dir / f"{json_path.stem}_{i}.json"
                if not alt.exists():
                    target_path = alt
                    break
                i += 1

        shutil.move(str(json_path), str(target_path))
if __name__ == "__main__":
    create_folders_from_json_names(DIR)
    move_jsons_into_named_folders(DIR)
