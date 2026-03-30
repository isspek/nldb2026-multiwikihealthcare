import json
import pandas as pd
import os
from openai import OpenAI
from tqdm import tqdm
from argparse import ArgumentParser
from source_code.assistant.openai import OpenAIModel

LLMS = {
    'llama': 'meta-llama/Llama-3.3-70B-Instruct:nebius',
    # 'deepseek': 'deepseek-ai/DeepSeek-R1-0528-fast'
    'deepseek': 'deepseek-ai/DeepSeek-R1-0528',
    'qwen': 'Qwen/Qwen3-Next-80B-A3B-Instruct:together'
}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--trg_lang')
    parser.add_argument('--model_name')
    parser.add_argument('--data_file')
    parser.add_argument('--data_field')
    parser.add_argument('--output_file')
    args = parser.parse_args()

    trg_lang = args.trg_lang
    model_name = args.model_name
    model = LLMS[model_name]
    data_file = args.data_file
    output_file = args.output_file
    data_field = args.data_field

    input_data = pd.read_csv(data_file)
    if 'deepseek' in model:
        base_url = 'https://api.studio.nebius.com/v1/'
        api_key= os.environ["NEBIUS"]
    else:
        base_url = 'https://router.huggingface.co/v1'
        api_key = os.environ["HF_TOKEN"]

    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    openai_client = OpenAIModel(model_id=model, client=client)

    results = []
    input_data_columns = input_data.columns
    for idx, row in tqdm(input_data.iterrows(), total=len(input_data)):
        if model_name in data_file:
            llm_answer = row['llm_answer']
            if llm_answer != 'MODEL_ERROR':
                results.append(row)
                continue
            else:
                print('Model error')
        u_prompt = row[data_field]
        try:
            if u_prompt == 'TRANSLATION_ERROR' and data_field=='translation':
                raise Exception('Error from translation!!!!')

            if 'instruction_prompt' in input_data_columns:
                print('instruction prompt is detected')
                system_prompt = row['instruction_prompt']
            else:
                system_prompt = None

            output = openai_client.request_llm(prompt=u_prompt, system_prompt=system_prompt,
                                               response_format=None,
                                           )
            row['llm_answer'] = output
        except Exception as e:
            print(e)
            row['llm_answer'] = 'MODEL_ERROR'
        results.append(row)

    results = pd.DataFrame(results)
    results.to_csv(output_file)
