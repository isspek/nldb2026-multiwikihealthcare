from argparse import ArgumentParser
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
import torch
from alignscore import AlignScore
import nltk
nltk.download('punkt_tab')


def compute_best_alignscore(result_data, batch_size, scorer):
    src_predictions, src_references = [], []
    trg_predictions, trg_references, row_indices = [], [], []

    for i, row in enumerate(result_data.to_dict(orient="records")):
        prediction = row["llm_answer"]
        src_predictions.append(prediction)
        src_references.append(row["src_evidence"])

        trg_evidences = list(ast.literal_eval(row["trg_evidences"]))
        trg_predictions.extend([prediction] * len(trg_evidences))
        trg_references.extend(trg_evidences)
        row_indices.extend([i] * len(trg_evidences))

    src_scores = []
    for start in tqdm(range(0, len(src_predictions), batch_size), desc="Source AlignScore"):
        end = start + batch_size
        batch_preds = src_predictions[start:end]
        batch_refs = src_references[start:end]
        batch_scores = scorer.score(contexts=batch_refs, claims=batch_preds)
        src_scores.extend(batch_scores)

    trg_scores = []
    for start in tqdm(range(0, len(trg_predictions), batch_size), desc="Target AlignScore"):
        end = start + batch_size
        batch_preds = trg_predictions[start:end]
        batch_refs = trg_references[start:end]
        batch_scores = scorer.score(contexts=batch_refs, claims=batch_preds)
        trg_scores.extend(batch_scores)

    best_scores = {}
    for idx, score in zip(row_indices, trg_scores):
        best_scores[idx] = max(best_scores.get(idx, -1e9), score)

    return src_scores, list(best_scores.values())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--result_data")
    parser.add_argument("--translation_data", required=False)
    parser.add_argument("--lang")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_file")
    parser.add_argument("--ckpt_path", required=True)

    args = parser.parse_args()



    result_data = pd.read_csv(args.result_data)
    translation_data = args.translation_data
    updated_result_data = []
    if translation_data:
        translation_data = pd.read_csv(translation_data)
        print(f'Number of samples in translation data: {len(translation_data)}')

        for i, row in result_data.iterrows():
            question = row['llm_output']
            try:
                src_evidence = translation_data[translation_data['llm_output'] == question]['src_evidence']
                row['src_evidence'] = src_evidence.values[0]
                updated_result_data.append(row)
            except Exception as e:
                print(e)
                continue
        print(f'Number of source evidence {len(result_data)}')

        updated_result_data = pd.DataFrame(updated_result_data)

        print(f'NUmber of instances {len(updated_result_data)}')
    else:
        updated_result_data = result_data.copy()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f'Number of data {len(updated_result_data)}')
    print(device)
    print("Computing AlignScore ...")
    scorer = AlignScore(
        model='roberta-large',
        batch_size=args.batch_size,
        device=device,
        ckpt_path=args.ckpt_path,
        evaluation_mode='nli_sp'
    )


    src_scores, best_trg_scores = compute_best_alignscore(
        updated_result_data,
        batch_size=args.batch_size,
        scorer=scorer
    )

    print(f"Mean Source AlignScore: {round(np.mean(src_scores) * 100, 4)}")
    print(f"Mean Best Target AlignScore: {round(np.mean(best_trg_scores) * 100, 4)}")

    updated_result_data['src_alignscore'] = src_scores
    updated_result_data['en_alignscore'] = best_trg_scores

    updated_result_data.to_csv(args.output_file, index=False)

    del scorer
    torch.cuda.empty_cache()
