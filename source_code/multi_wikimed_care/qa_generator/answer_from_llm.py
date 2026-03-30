import json
import os
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from vllm import LLM, SamplingParams

os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

LLMS = {
    'commandr': 'CohereLabs/c4ai-command-r-plus',
    'aya': 'CohereLabs/aya-expanse-32b'
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
    output_file = args.output_file
    max_tokens = 4096

    seed = 0
    temperature = 0
    llm = LLM(model=model, trust_remote_code=True, tensor_parallel_size=4, max_model_len=max_tokens)
    sampling_params = SamplingParams(seed=seed, temperature=temperature, max_tokens=4096)
    input_data = pd.read_csv(data_file)

    input_data_columns = input_data.columns
    prompts = []
    for _, row in input_data.iterrows():
        u_prompt = row[data_field]
        if 'instruction_prompt' in input_data_columns:
            system_prompt = row['instruction_prompt']
            prompts.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": u_prompt}
            ])
        else:
            prompts.append([{"role": "user", "content": u_prompt}])

    results = []
    for i, (prompt, row) in enumerate(
            tqdm(zip(prompts, input_data.to_dict(orient='records')),
                 total=len(prompts),
                 desc="Generating one by one")
    ):
        try:
            output = llm.chat([prompt], sampling_params)[0]
            row['llm_answer'] = output.outputs[0].text
        except Exception as e:
            row['llm_answer'] = "MODEL_ERROR"

        results.append(row)

    # --- Save results ---
    results = pd.DataFrame(results)
    results.to_csv(output_file, index=False)