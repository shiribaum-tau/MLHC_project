import os
import joblib
import pandas as pd

disease_names = {
    't2d': 'Type 2 Diabetes',
    'skin_cancer': 'Skin Cancer',
    'panc': 'Pancreatic Cancer',
    'bc_all': 'Breast Cancer (All)',
    'bc_women': 'Breast Cancer (Females)',
}

def walk_and_read_pickles(root_dir):
    all_data_36 = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.endswith('.pkl'):
                continue
            new_data = {}
            fpath = os.path.join(dirpath, fname)
            try:
                dat = joblib.load(fpath)
                print(f"Read: {fpath}")
            except Exception as e:
                print(f"Failed to read {fpath}: {e}")
            
            # import ipdb;ipdb.set_trace()
            dat_36 = dat[36]
            new_data['disease'], new_data['model'] = dirpath.replace(".", "").replace("\\", ""), fname.replace(".pkl", "")
            new_data['disease'] = disease_names[new_data['disease']]
            for key in ['auc', 'aupr', 'best_f1', 'rr_at_1000_per_million']:
                new_data[key] = dat_36[key]
            all_data_36.append(new_data)
    return all_data_36

dat = walk_and_read_pickles(".")
df = pd.DataFrame(dat)
df.to_csv("aggregated_results.csv", index=False)