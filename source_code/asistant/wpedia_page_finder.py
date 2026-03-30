from argparse import ArgumentParser
import pandas as pd
import json
from pydantic import BaseModel
from typing import Union
from code.assistant.user_prompts import ENTITY_ANALYZER
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

parser = ArgumentParser()
parser.add_argument('--input_file')
parser.add_argument('--output_file')

args = parser.parse_args()

DATA_FILE=args.input_file

# health_related entity and wikipedia page if it has
OUTPUT_FILE=args.output_file
MAX_TOKENS=5000
SEED=0
TEMPERATURE=0
llm_name='meta-llama/Llama-3.3-70B-Instruct'

class EntityAnalyzer(BaseModel):
    is_healthcare_related: bool
    wikipedia_page: Union[str, None]

system_prompt = ENTITY_ANALYZER

llm = LLM(model=llm_name, trust_remote_code=True, max_model_len=MAX_TOKENS, tensor_parallel_size=4,
          guided_decoding_backend='outlines')

json_schema = EntityAnalyzer.model_json_schema()
guided_decoding_params = GuidedDecodingParams(json=json_schema)
sampling_params = SamplingParams(max_tokens=MAX_TOKENS, guided_decoding=guided_decoding_params, seed=SEED, temperature=TEMPERATURE)

data = pd.read_csv(DATA_FILE, sep=';')
entities = data['query'].unique()

print(f'Number of unique entities {len(entities)}')

prompts = []
for entity in entities:
    user_prompt = f"### Input Entity ###\n{entity}\n### Output ###\n"

    conversation = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    prompts.append(conversation)
    outputs = llm.chat(prompts, sampling_params)

results = []
for (output, entity) in zip(outputs, entities):
    output_json = json.loads(output.outputs[0].text)
    results.append({'entity': entity,
                    'is_healthcare_related': output_json['is_healthcare_related'],
                    'wikipedia_page': output_json['wikipedia_page']
                    })

results = pd.DataFrame(results)
results.to_csv(OUTPUT_FILE, index=False, sep=';')