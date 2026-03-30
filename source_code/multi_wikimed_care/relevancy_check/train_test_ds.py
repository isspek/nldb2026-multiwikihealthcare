from sklearn.model_selection import train_test_split
import json
from pathlib import Path
import pandas as pd


def normalize_label(llm_label):
    label_json = json.loads(llm_label)
    return label_json['answer']

if __name__ == '__main__':
    trg_lang = 'zh'
    main_dir = Path(f'data/multi-wikimedcare/atomic_chunks/{trg_lang}')
    output_dir = Path(f'data/multi-wikimedcare/relevancy_ds/{trg_lang}')
    # english_ds_files = ['chunk_1_openai.csv', 'chunk_2_openai.csv', 'chunk_3_openai.csv']
    english_ds_files = ['chunk_1_deepseek.csv', 'chunk_2_deepseek.csv', 'chunk_3_deepseek.csv']
    
    train_data = []
    for ds_file in english_ds_files:
        data = pd.read_csv(main_dir/ds_file)
        print(data['atomic_fact'])
        data['label'] = data['llm_output'].apply(lambda x: normalize_label(x))
        train_data.append(data)

    train_data = pd.concat(train_data)
    train_data.drop_duplicates(subset=["atomic_fact"], inplace=True)
    
    print(train_data.groupby(['label'])['atomic_fact'].count())
    train_data = train_data[['atomic_fact', 'label']]

    train_df, dev_df = train_test_split(
        train_data,
        test_size=0.2,
        stratify=train_data["label"],   # ensures class balance
        random_state=42         # reproducibility
    )
    
    print('Train data')
    print(train_df.groupby(['label'])['atomic_fact'].count())
    train_df.to_csv(output_dir/'train.csv')
    
    print('Dev data')
    print(dev_df.groupby(['label'])['atomic_fact'].count())
    dev_df.to_csv(output_dir/'dev.csv')
