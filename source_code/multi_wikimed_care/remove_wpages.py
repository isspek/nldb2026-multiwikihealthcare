from pathlib import Path
from code.multi_wikimed_care.langs import use_case_langs
import pandas as pd
data = pd.read_csv('data/multi-wikimedcare/all_related_entities_wikipages_wcategories.csv')
print(f'Number of the Wiki pages: {len(data)}')

entities = data['entity'].tolist()
entities = list(map(lambda x: x.replace('/', '-'), entities))
main_folder = Path('data/multi-wikimedcare/html')

for fpath in main_folder.rglob('*.txt'):
    entity_name = fpath.name.replace('.txt', '')
    if entity_name not in entities:
        print(f'{entity_name} does not exist!! Removing the file {fpath}')
        fpath.unlink()