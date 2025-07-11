import os

CPP_EXTENSIONS = [".cpp", ".hpp"]
project_root = "."

def has_real_code(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith(("//", "/*", "*", "#")):
                    return True
        return False
    except:
        return False

results = []

for root, _, files in os.walk(project_root):
    for file in files:
        if any(file.endswith(ext) for ext in CPP_EXTENSIONS):
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, project_root)
            if has_real_code(full_path):
                results.append(("✅ CODE", rel_path))
            else:
                results.append(("🟡 NO CODE", rel_path))

for status, path in sorted(results, key=lambda x: x[1].lower()):
    print(f"{status}\t{path}")
