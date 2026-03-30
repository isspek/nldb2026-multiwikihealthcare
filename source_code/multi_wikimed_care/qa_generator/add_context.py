import pandas as pd
from argparse import ArgumentParser

'''
Prompt for the instruction:
https://huggingface.co/blog/ngxson/make-your-own-rag
'''

# instruction_prompt = f'''You are a helpful chatbot.
# Use only the following pieces of context to answer the question. Don't make up any new information:
# {'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
# '''

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src_translation')
    parser.add_argument('--q_translation')
    parser.add_argument('--output_file')
    parser.add_argument('--option', type=int)

    args = parser.parse_args()

    src_translation = args.src_translation
    q_translation = args.q_translation
    output_file = args.output_file
    option = args.option

    src_data = pd.read_csv(src_translation)
    qtranslated_data = pd.read_csv(q_translation)

    print(f'Samples of {qtranslated_data}')

    new_data = []
    for row in qtranslated_data.to_dict(orient='records'):
        translation_question = row['translation']
        src_info = src_data[src_data['llm_output'] == row['llm_output']]['src_evidence'].values[0]

        if option == 1:
            instruction_prompt = f'You are a helpful chatbot. Use only the following pieces of context to answer the question. Don\'t make up any new information:\n{src_info}'
            row['instruction_prompt'] = instruction_prompt
            del row['llm_answer']
            new_data.append(row)


    new_data = pd.DataFrame(new_data)
    print(new_data.head())
    new_data.to_csv(output_file)