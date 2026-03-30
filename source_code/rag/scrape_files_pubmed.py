from pathlib import Path
import pandas as pd
from tqdm import tqdm
from paperscraper.pubmed import get_and_dump_pubmed_papers

lang= 'tr'
lang_term = 'turkish'
query_file = f'data/multi-wikimedcare/final_ds/{lang}/qa_llama_filtered_qtranslated.csv'
query_data = pd.read_csv(query_file)
entities = query_data['entity'].unique()


for entity in tqdm(entities):
    if '/' in entity:
        _entity = entity.replace('/',' ')
    else:
        _entity = entity
        
    output_filepath = Path(f'source_code/rag/dump/{lang_term}/{_entity}.jsonl')
    if output_filepath.exists() and output_filepath.stat().st_size > 0:
        print(f"Skipping {entity} — file already exists and is not empty.")
        continue
    query =[[_entity], [lang_term]]
    get_and_dump_pubmed_papers(query, output_filepath=str(output_filepath), max_results=200)