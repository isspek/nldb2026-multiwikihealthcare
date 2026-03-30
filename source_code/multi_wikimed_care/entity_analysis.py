from pathlib import Path
import pandas as pd

'''
python -m code.multi_wikimed_care.entity_analysis
'''

data_folder = Path('data/multi-wikimedcare')

merged_data = []
for dfile in data_folder.rglob('chunk_*_all_related_entities_out_v1.csv'):
    ent_df = pd.read_csv(dfile, sep=';')
    merged_data.append(ent_df)

merged_data = pd.concat(merged_data)
merged_data = merged_data[merged_data['is_healthcare_related']==True]

print(f'Filtered out entities are {len(merged_data)}')
print(merged_data)