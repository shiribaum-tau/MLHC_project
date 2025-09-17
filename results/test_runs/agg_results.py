import os
import re


OUT_DIR = "agg_results"

os.makedirs(OUT_DIR, exist_ok=True)

for root, dirs, files in os.walk("."):
    if "full_results.pkl" in files:
        m = re.search(r"([^/]+_test)/out$", root)
        if m:
            name = m.group(1)
        else:
            import ipdb;ipdb.set_trace()
        new_path = os.path.join(OUT_DIR, name + ".pkl")
        full_path = os.path.join(root, "full_results.pkl")
        op = f"cp {full_path} {new_path}"
        os.system(op)
    # import ipdb;ipdb.set_trace()