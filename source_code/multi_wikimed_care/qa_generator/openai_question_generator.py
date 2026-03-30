import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel
from source_code.assistant.openai import OpenAIModel
from source_code.assistant.question_generation import system_prompts, user_prompts
from argparse import ArgumentParser


class Question(BaseModel):
    question: str

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--trg_lang')
    parser.add_argument('--data_file')
    parser.add_argument('--output_file')
    
    args = parser.parse_args()
    trg_lang = args.trg_lang
    
    system_prompt = system_prompts[trg_lang]
    data_file = args.data_file
    output_file = args.output_file
    
    model = 'gpt-4o-mini'
    openai_client = OpenAIModel(model_id=model)
    
    input_data = pd.read_csv(data_file)
    
    results = []
    for idx, row in tqdm(input_data.iterrows(), total=len(input_data)):
        _system_prompt = system_prompt.format(entity=row['entity'])
        fact = row['atomic_fact']
        src_paragraph = row['src_evidence']
        user_prompt = user_prompts[trg_lang].format(fact=fact, paragraph=src_paragraph)
        output = openai_client.request_llm(prompt=user_prompt, system_prompt=_system_prompt, response_format= Question)
        row['llm_output'] = output
        results.append(row)
    results = pd.DataFrame(results)
    results.to_csv(output_file)
    