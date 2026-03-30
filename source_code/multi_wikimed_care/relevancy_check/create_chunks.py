from argparse import ArgumentParser
import pandas as pd

def create_unique_chunks(df, column, chunk_size=1000, out_folder="", prefix="chunk"):
    n = len(df)
    num_chunks = (n + chunk_size - 1) // chunk_size  # ceil division
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = df.iloc[start:end]   # preserve row order
        chunk.to_csv(f"{out_folder}/{prefix}_{i+1}.csv", index=False)
    
    return num_chunks


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--trg_lang')
    parser.add_argument('--evidence_file')
    args = parser.parse_args()

    trg_lang = args.trg_lang
    data_file = args.evidence_file
    human_annotated_file = f"data/multi-wikimedcare/relevancy_check_{trg_lang}.json"
    
    human_annotated_data = [] 
    for _,row in pd.read_json(human_annotated_file).iterrows():
        human_annotated_data.append(row['data']['atomic_fact'])
    
    output_folder = f"data/multi-wikimedcare/atomic_chunks/{trg_lang}"
    input_data = pd.read_csv(data_file)
    input_data = input_data[~input_data['atomic_fact'].isin(human_annotated_data)]
    input_data= input_data[input_data['intersection']=='yes']
    input_data= input_data[input_data['fact_lang']==trg_lang]
    
    print(f'Number of {trg_lang} automic fact {len(input_data)}')
    
    num_chunks = create_unique_chunks(input_data, "atomic_fact", out_folder=output_folder, chunk_size=1000)
    