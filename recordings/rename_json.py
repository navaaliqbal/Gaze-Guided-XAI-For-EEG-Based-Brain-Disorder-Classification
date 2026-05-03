from pathlib import Path
import json

# Directory containing the JSON files
DIR = Path(r"D:\TobiiPro.SDK.Python.Windows_2.1.0.1\64\recordings")  # change this

def extract_edf_stem(json_path: Path) -> str | None:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        edf_name = data.get("edf_file")
        if not edf_name:
            return None
        return Path(edf_name).stem  # "0000045.edf" -> "0000045"
    except Exception as e:
        print(f"Skip {json_path.name}: {e}")
        return None

def rename_jsons(directory: Path):
    for json_path in directory.glob("*.json"):
        stem = extract_edf_stem(json_path)
        if not stem:
            continue

        new_path = json_path.with_name(stem + ".json")

        # Handle name collision
        counter = 1
        while new_path.exists() and new_path != json_path:
            new_path = json_path.with_name(f"{stem}_{counter}.json")
            counter += 1

        if new_path != json_path:
            json_path.rename(new_path)
            print(f"{json_path.name} -> {new_path.name}")

if __name__ == "__main__":
    rename_jsons(DIR)
