import os

# Mapping of substrings to new names
RENAME_MAP = [
    ("paper_tf", "Paper Transformer.pkl"),
    ("mm_tf", "Multi-modal Transformer.pkl"),
    ("tf", "Transformer.pkl"),
    ("mlp", "MLP.pkl"),
]

def get_new_name(filename):
    for key, new_name in RENAME_MAP:
        if key in filename:
            return new_name
    return None

def walk_and_rename(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".pkl"):
                new_name = get_new_name(fname)
                if new_name and fname != new_name:
                    src = os.path.join(dirpath, fname)
                    dst = os.path.join(dirpath, new_name)
                    print(f"Renaming {src} -> {dst}")
                    os.rename(src, dst)

if __name__ == "__main__":
    walk_and_rename(".")

