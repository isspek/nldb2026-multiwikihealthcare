from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

lang = 'zh'
llm='qwen'
main_data = Path(f'data/multi-wikimedcare/final_data/{lang}/')
wiki_analysis_data = pd.read_csv(f'data/multi-wikimedcare/wiki_full_{lang}.csv')
wiki_analysis_data = wiki_analysis_data.drop_duplicates(subset='wikipage_en')
wiki_analysis_data.drop(columns=['entity'], inplace=True)
wiki_analysis_data = wiki_analysis_data.rename(columns={'wikipage_en': 'entity'})

non_en_answers_alignscore = pd.read_csv(main_data / f'qa_{llm}_filtered_anstranslated_alignscore.csv')
non_en_answers_alignscore = pd.merge(non_en_answers_alignscore, wiki_analysis_data, on="entity", how="inner")

en_answers_alignscore = pd.read_csv(main_data / f'qa_{llm}_filtered_qtranslated_alignscore.csv')

merged_data_alignscore = pd.merge(non_en_answers_alignscore, en_answers_alignscore, on="llm_output", how="inner", suffixes=("_trg", "_en"))
non_en_answers_relevancy = pd.read_csv(main_data / f'qa_{llm}_filtered_anstranslated_relevancy.csv')
en_answers_relevancy= pd.read_csv(main_data / f'qa_{llm}_filtered_qtranslated_relevancy.csv')

merged_data_relevancy = pd.merge(non_en_answers_relevancy, en_answers_relevancy, on="user_input", how="inner", suffixes=("_trg", "_en"))

merged_data_relevancy = merged_data_relevancy.rename(columns={'user_input': 'translation'})

if llm=='llama':
    merged_data_relevancy = merged_data_relevancy.rename(columns={'translation_trg': 'translation'})
    merged_data_alignscore = merged_data_alignscore.rename(columns={'translation_trg': 'translation'})

merged_data = pd.merge(merged_data_alignscore, merged_data_relevancy, on="translation", how="inner")
merged_data['llm_answer_en_len'] = merged_data['llm_answer_en'].apply(lambda x: len(x))
merged_data['llm_answer_trg_len'] = merged_data['llm_answer_trg'].apply(lambda x: len(x))

print(f'Number of data: {len(merged_data)}')
print(merged_data.columns)

wiki_quality_features = [
    'section_trg_count', 'section_en_count',
    'num_paragraph_trg', 'num_facts_trg',
    'num_paragraph_en', 'num_facts_en',
    'num_facts_trg', 'num_facts_en',
    'trg_ref_count', 'en_ref_count'
]


# Select LLM answer quality metrics
llm_quality_metrics = [
    'src_alignscore_trg', 'en_alignscore_trg',  # factuality (target)
    'src_alignscore_en', 'en_alignscore_en',    # factuality (English)
    'answer_relevancy_trg', 'answer_relevancy_en',
    'llm_answer_trg_len', 'llm_answer_en_len'
]

merged_data = merged_data[wiki_quality_features + llm_quality_metrics].apply(pd.to_numeric, errors='coerce')
merged_data = merged_data.dropna()
print('trg len')
print(np.mean(merged_data['llm_answer_trg_len'].to_numpy()))
print('en len')
print(np.mean(merged_data['llm_answer_en_len'].to_numpy()))

corr_matrix = merged_data.corr(method='spearman')
corr_focus = corr_matrix.loc[wiki_quality_features, llm_quality_metrics]

print("Spearman Correlation between Wiki Quality and LLM Answer Quality:\n")
print(corr_focus.round(3))

plt.figure(figsize=(12, 6))
sns.heatmap(corr_focus, annot=True, cmap="coolwarm", center=0)
plt.title(f"Correlation: Wiki Page Quality vs LLM Answer Quality ({llm})")
plt.savefig(main_data/f"wiki_{llm}_{lang}_correlation_heatmap.pdf", format='pdf', bbox_inches='tight')