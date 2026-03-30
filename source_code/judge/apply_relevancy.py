import os
import numpy as np
import pandas as pd
from ragas import EvaluationDataset, evaluate
from ragas.metrics import ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
'''
python -m source_code.judge.apply_relevancy
'''

load_dotenv()

evaluator_llm = LangchainLLMWrapper(
    ChatOpenAI(model="gpt-4o-mini", temperature=0)
)

exclude_ents = ['Computer virus']
metrics = [ResponseRelevancy()]

src_lang = 'it'
model='llama'
non_en_data_path = f'data/multi-wikimedcare/final_data/{src_lang}/qa_{model}_filtered_qtranslated_relevancy.csv'
# non_en_data_path = f'data/multi-wikimedcare/final_data/{src_lang}/qa_{model}_filtered_qtranslated.csv'
en_data_path = f'data/multi-wikimedcare/final_data/{src_lang}/qa_{model}_filtered_qtranslated.csv'
output_file = f'data/multi-wikimedcare/final_data/{src_lang}/qa_{model}_filtered_qtranslated_relevancy.csv'
non_en_data = pd.read_csv(non_en_data_path)
# non_en_data = non_en_data[~non_en_data['entity'].isin(exclude_ents)]
en_data = pd.read_csv(en_data_path)

if 'relevancy' in non_en_data_path:
    # Identify NaN or invalid scores
    missing_mask = non_en_data['answer_relevancy'].isna()
    to_fix = non_en_data[missing_mask]
    print(f"Found {len(to_fix)} rows with NaN relevancy to recompute.")

    if len(to_fix) > 0:
        fixed_scores = []
        for i, row in enumerate(to_fix.to_dict(orient='records')):
            try:
                evaluation_data = [{'user_input': row['user_input'], 'response': row['response']}]
                evaluation_dataset = EvaluationDataset.from_list(evaluation_data)
                res = evaluate(
                    dataset=evaluation_dataset,
                    metrics=[ResponseRelevancy()],
                    llm=evaluator_llm
                )
                score = float(res.scores[0]['answer_relevancy'])
                fixed_scores.append(score)
            except Exception as e:
                print(f"⚠️ Eval failed on row {i}: {e}")
                fixed_scores.append(np.nan)
        non_en_data.loc[missing_mask, 'answer_relevancy'] = fixed_scores
        non_en_data.to_csv(non_en_data_path, index=False)
        print(f"✅ Recomputed and saved {len(to_fix)} relevancy scores back to file.")
else:
    evaluation_data = []
    for row in non_en_data.to_dict(orient='records'):
        if 'qtranslated' in non_en_data:
            evaluation_data.append({
                'user_input': row['translation'],
                'response': row['llm_answer']
            })
        else:
            try:
                translation = en_data[en_data['llm_output'] == row['llm_output']]['translation'].values[0]
                evaluation_data.append({
                'user_input': translation,
                'response': row['llm_answer']
                }
            )
            except:
                continue

    print(f'Number of eval {len(evaluation_data)}')
    evaluation_data = evaluation_data
    embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text:latest"))
    evaluation_dataset = EvaluationDataset.from_list(evaluation_data)
    results = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=evaluator_llm
    )

    combined = [{**data, **res} for data, res in zip(evaluation_data, results.scores)]
    combined_result = pd.DataFrame(combined)
    combined_result.to_csv(output_file, index=False)