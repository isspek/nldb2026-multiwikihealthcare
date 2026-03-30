import json
import pandas as pd
from argparse import ArgumentParser


def normalize_func(result):
    result = json.loads(result)
    return result['question']

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_file')
    parser.add_argument('--filter_file')
    parser.add_argument('--output_file')

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    filter_file = args.filter_file

    input_data = pd.read_csv(input_file)
    input_data['llm_output'] = input_data['llm_output'].apply(lambda x: normalize_func(x))

    filter_data = pd.read_csv(filter_file)
    print(filter_data['overall_label'])
    filter_data = filter_data.rename(columns={"overall_label": "Overall_Pass", "question": "llm_output", "overall": "Overall_Pass", "overall_pass": "Overall_Pass"})

    print(filter_data.columns)

    filter_data = filter_data[(filter_data['Overall_Pass']==True)| (filter_data['Overall_Pass']=='PASS')| (filter_data['Overall_Pass']=='Yes') | (filter_data['Overall_Pass']=='Good')]
    print(filter_data['llm_output'].tolist()[0])
    input_data = input_data[input_data['llm_output'].isin(filter_data['llm_output'].tolist())]
    print(len(input_data))
    input_data.to_csv(output_file,index=False)