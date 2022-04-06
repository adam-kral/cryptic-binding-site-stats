import json
import pickle
from pathlib import Path
from typing import Iterable

import pandas as pd
LOAD_FILES_CAP = None  # for testing load only some files (loading slow)

def load_df(json_files: Iterable[Path]):
    # super_df = pd.DataFrame()

    json_files = list(json_files)
    if LOAD_FILES_CAP:
        json_files = list(json_files)[:LOAD_FILES_CAP]  # testing hack

    total_files = len(json_files)

    dfs = []
    for i, json_file in enumerate(json_files):
        with json_file.open() as f:
            json_output = json.load(f)

        df = pd.json_normalize(json_output, 'binding_sites', ['apo_pdb_code', 'holo_pdb_code', 'apo_chain_id', 'holo_chain_id'])
        dfs.append(df)

        if i % 10 == 0:
            print(f'\rloading... {i} / {total_files} done', end='')

    return pd.concat(dfs, ignore_index=True)


def main():
    output_root = '/Users/adam/pschool/bakalarka/cryptic_output/output 2'
    files = Path(output_root).rglob('*.json')

    df = load_df(files)
    # save
    with open('df.pickle', 'wb') as handle:
        pickle.dump(df, handle)

main()