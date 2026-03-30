import pandas as pd
import wikipediaapi
from tqdm import tqdm
from pathlib import Path
from code.multi_wikimed_care.langs import use_case_langs
'''
python -m code.multi_wikimed_care.find_wpages_other_lang
'''

data_folder = Path('data/multi-wikimedcare')

merged_data = []
for dfile in data_folder.rglob('chunk_*_all_related_entities_out_v1.csv'):
    ent_df = pd.read_csv(dfile, sep=';')
    merged_data.append(ent_df)

merged_data = pd.concat(merged_data)
merged_data = merged_data[merged_data['is_healthcare_related']==True]
merged_data.duplicated(subset=['wikipedia_page'])

print('Number of articles')
print(len(merged_data['wikipedia_page'].unique()))


wiki = wikipediaapi.Wikipedia(user_agent='LLM Health Research (tokidrean@gmail.com)', language='en')

processed_data = []
for sample in tqdm(merged_data.to_dict(orient='records'), total=len(merged_data)):
    wikipedia_page = sample['wikipedia_page']
    wikipedia_page = wikipedia_page.replace('https://en.wikipedia.org/wiki/', '')
    page = wiki.page(wikipedia_page)
    for lang, linked_page in page.langlinks.items():
        if lang in use_case_langs:
            sample[f'wikipage_{lang}'] = linked_page.title

    processed_data.append(sample)

processed_data = pd.DataFrame(processed_data)
processed_data.to_csv('data/multi-wikimedcare/all_related_entities_wikipages.csv')