import os
import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel
from enum import Enum
from argparse import ArgumentParser
from openai import OpenAI
from source_code.assistant.openai import OpenAIModel
from source_code.assistant.relevancy_check import system_prompts, user_prompts

class RelevantEnum(str, Enum):
    relevant = 'relevant'
    not_relevant = 'not relevant'

class RelevancyClassifier(BaseModel):
    answer: RelevantEnum

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--trg_lang')
    parser.add_argument('--data_file')
    parser.add_argument('--output_file')
    parser.add_argument('--model', default='gpt-4o-mini')
    args = parser.parse_args()

    trg_lang = args.trg_lang
    system_prompt = system_prompts[trg_lang]
    data_file = args.data_file
    output_file = args.output_file
    model = args.model

    if 'json' in data_file:
        input_data = pd.read_json(data_file)
    else:
        input_data = pd.read_csv(data_file, sep=',')

    if model == 'gpt-4o-mini':
        openai_client = OpenAIModel(model_id=model)
    else:
        client = OpenAI(
            base_url='https://router.huggingface.co/v1',
            api_key=os.environ["HF_TOKEN"]
        )
        openai_client = OpenAIModel(model_id=model, client=client)

    results = []
    for idx, row in tqdm(input_data.iterrows(), total=len(input_data)):
        if 'json' in data_file:
            input_text = row['data']['atomic_fact']
        else:
            input_text = row['atomic_fact']
        user_prompt = user_prompts[trg_lang].format(input_text=input_text)
        output = openai_client.request_llm(prompt=user_prompt, system_prompt=system_prompt, response_format= RelevancyClassifier)
        row['llm_output'] = output
        results.append(row)
    results = pd.DataFrame(results)
    results.to_csv(output_file)
    
