import pandas as pd
import json
import time
from tqdm import tqdm
from pydantic import BaseModel
from googletrans import Translator
from diskcache import Cache
from argparse import ArgumentParser

class Question(BaseModel):
    question: str
    

CACHE_TRANS = Cache("cache_translations")
TRANSLATOR = Translator()

@CACHE_TRANS.memoize()
def apply_translation(text: str, target_language: str)-> str:
    translation = TRANSLATOR.translate(text, dest = target_language)
    return translation.text

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_file')
    parser.add_argument('--output_file')

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    input_data = pd.read_csv(input_file)
    
    results = []

    if 'qtranslated' in input_file:
        correction_df = pd.read_csv(args.correction_file)
        corrected_rows = []
        for idx, row in tqdm(input_data.iterrows(), total=len(input_data)):
            # If llm_answer has an error
            if row["translation"] == "TRANSLATION_ERROR":
                llm_output = correction_df.iloc[idx]["llm_output"]
                row["translation"] = apply_translation(llm_output, "en")
            corrected_rows.append(row)
        corrected_df = pd.DataFrame(corrected_rows)
        corrected_df.to_csv(output_file, index=False)

    for idx, row in tqdm(input_data.iterrows(), total=len(input_data)):
        question = row['llm_output']
        try:
            row['translation'] = apply_translation(question, target_language='en')
            time.sleep(1)
        except Exception as e:
            row['translation'] = 'TRANSLATION_ERROR'
        results.append(row)
    results = pd.DataFrame(results)
    results.to_csv(output_file, index=False)
    