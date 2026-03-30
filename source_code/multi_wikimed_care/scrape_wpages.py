from code.multi_wikimed_care.langs import use_case_langs
from pathlib import Path
import wikipediaapi
import pandas as pd

'''
python -m code.multi_wikimed_care.scrape_wpages
'''

# data_file = 'data/multi-wikimedcare/all_related_entities_wikipages.csv'
data_file = 'data/multi-wikimedcare/all_related_entities_wikipages_wcategories.csv'
main_folder = Path('data/multi-wikimedcare/')
data = pd.read_csv(data_file, sep=',')
data = data.dropna(subset=['wikipage_de', 'wikipage_it', 'wikipage_tr', 'wikipage_it'], how='all')

print(f'Number of file, after dropping empty pages in other langs {len(data)}')

# html_data = True
html_data = False
for lang in use_case_langs:
    if lang == 'en':
        column_name = 'wikipedia_page'
    else:
        column_name = f'wikipage_{lang}'

    samples = data.to_dict(orient='records')
    print(f'Scraping {lang} Wikipedia pages')

    if html_data:
        wiki_wiki = wikipediaapi.Wikipedia(user_agent='LLM Health Research (tokidrean@gmail.com)', language=lang,
                                           extract_format=wikipediaapi.ExtractFormat.HTML)
        subfolder = 'html'
    else:
        wiki_wiki = wikipediaapi.Wikipedia(user_agent='LLM Health Research (tokidrean@gmail.com)', language=lang,
                                           extract_format=wikipediaapi.ExtractFormat.WIKI)
        subfolder = 'clean'

    for sample in samples:
        entity = sample['entity'].replace('/', '-')
        fpath = main_folder / subfolder / lang / f'{entity}.txt'
        if fpath.is_file() and fpath.stat().st_size > 0:
           print(f'Skipping {entity}')
           continue
        wiki_link = sample[column_name] if lang != 'en' else sample[column_name].replace('https://en.wikipedia.org/wiki/','')
        p_wiki = wiki_wiki.page(wiki_link)
        fpath.write_text(p_wiki.text, encoding='utf-8')