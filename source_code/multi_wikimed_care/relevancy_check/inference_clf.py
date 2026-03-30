import pandas as pd
from pathlib import Path
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, 
                          pipeline)

from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_file')
    parser.add_argument('--model_path')
    parser.add_argument('--trg_lang')
    parser.add_argument('--output_file')
    
    args = parser.parse_args()
    output_file = args.output_file
    trg_lang = args.trg_lang
    model_path=args.model_path
    
    main_dir = Path(f'data/multi-wikimedcare/relevancy_ds/{trg_lang}')
    # test_df = pd.read_csv(main_dir/'test.csv')
    test_df = pd.read_csv(main_dir / args.input_file )
    texts = test_df['atomic_fact'].tolist()
    
    output_file = main_dir / output_file
    batch_size = 16
    config = AutoConfig.from_pretrained(
        model_path,
        num_labels=2,
        id2label={0: "not relevant", 1: "relevant"},
        label2id={"relevant": 1, "not relevant": 0},
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
        ignore_mismatched_sizes=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, max_length=256, truncation=True, padding=True)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device="cuda")
    
    results = pipe(texts, batch_size=batch_size)
    
    results_df = []
    for text, result in zip(texts, results):
        results_df.append(
            {'atomic_fact': text,
             'label': result['label'],
             'score': result['score']}
        )
        
    results_df = pd.DataFrame(results_df)
    results_df.to_csv(output_file, index=False)