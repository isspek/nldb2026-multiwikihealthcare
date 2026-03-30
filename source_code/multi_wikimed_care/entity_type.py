import pandas as pd
from flair.data import Sentence
from flair.nn import Classifier
from typing import List
tagger = Classifier.load('hunflair2')

# this check types

def ner_entity_extraction(text: str) -> List[str]:
    sentence = Sentence(text)
    tagger.predict(sentence)
    return sentence.to_dict()

if __name__ == '__main__':
    data = pd.read_csv('data/multi-wikimedcare/all_related_entities.csv', sep=';')
    entities = data['query'].unique()

    print(f'Number of unique entities {len(entities)}')

    for entity in entities:
        print('==entity==')
        print(ner_entity_extraction(entity))