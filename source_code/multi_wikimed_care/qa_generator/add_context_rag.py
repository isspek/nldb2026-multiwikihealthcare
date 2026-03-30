import pandas as pd
import json
from argparse import ArgumentParser
from pathlib import Path
import bm25s
'''
Prompt for the instruction:
https://huggingface.co/blog/ngxson/make-your-own-rag
'''

# instruction_prompt = f'''You are a helpful chatbot.
# Use only the following pieces of context to answer the question. Don't make up any new information:
# {'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
# '''

def index_data(entity):
    data = pd.read_json(entity, lines=True)
    try:
        corpus_json = [
            a for a in data['abstract']
            if pd.notna(a) and str(a).strip() != ''
        ]
        corpus_text = [
            abstract
            for abstract in data["abstract"]
            if isinstance(abstract, str) and abstract.strip()  # filters out None, NaN, and empty strings
        ]
    except Exception:
        return None

    if len(corpus_text) < 50:
        return None

    corpus_tokens = bm25s.tokenize(corpus_text, stopwords="en")
    retriever = bm25s.BM25(corpus=corpus_json)
    retriever.index(corpus_tokens)
    return retriever

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src_translation')
    parser.add_argument('--q_translation')
    parser.add_argument('--output_file')
    parser.add_argument('--option', type=int)
    parser.add_argument('--rag_data')

    args = parser.parse_args()

    src_translation = args.src_translation
    q_translation = args.q_translation
    output_file = args.output_file
    option = args.option
    rag_data_folder = Path(args.rag_data)

    src_data = pd.read_csv(src_translation)
    qtranslated_data = pd.read_csv(q_translation)

    print(f'Samples of {qtranslated_data}')


    new_data = []
    for entity, df in qtranslated_data.groupby(['entity']):
        entity = entity[0] if isinstance(entity, tuple) else entity
        if '/' in entity:
            entity = entity.replace('/', ' ')

        retriever = index_data(rag_data_folder / f'{entity}.jsonl')

        if not retriever:
            print(f'\'{entity}\',')
            continue

        for row in df.to_dict(orient='records'):
            question = row['translation']
            query_tokens = bm25s.tokenize(question)
            max_k = min(10, len(retriever.corpus))
            results, scores = retriever.retrieve(query_tokens, k=max_k)

            context_prompt = ''
            for i in range(results.shape[1]):
                doc, score = results[0, i], scores[0, i]
                context_prompt += f"Rank {i + 1} (score: {score:.2f})\n{doc}\n"

            instruction_prompt = (
                "You are a helpful chatbot. Use only the following pieces of context "
                "to answer the question. Don't make up any new information:\n"
                f"{context_prompt}"
            )

            row['instruction_prompt'] = instruction_prompt
            row.pop('llm_answer', None)  # safely remove if it exists
            new_data.append(row)

    new_data = pd.DataFrame(new_data)
    print(len(new_data))
    new_data.to_csv(output_file, index=False)