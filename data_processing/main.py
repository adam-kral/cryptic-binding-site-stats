import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def load_df(json_files: Iterable[Path]):
    super_df = pd.DataFrame()

    json_files = list(json_files)
    total_files = len(json_files)

    for i, json_file in enumerate(json_files):
        with json_file.open() as f:
            json_output = json.load(f)

        df = pd.json_normalize(json_output, 'binding_sites', ['apo_pdb_code', 'holo_pdb_code', 'apo_chain_id', 'holo_chain_id'])
        super_df = pd.concat([super_df, df])

        if i % 10 == 0:
                print(f'\rloading... {i} / {total_files} done', end='')

    return super_df


def main():
    output_root = '/Users/adam/pschool/bakalarka/cryptic_output/output'
    files = Path(output_root).rglob('*.json')

    df = load_df(files)
    df

main()