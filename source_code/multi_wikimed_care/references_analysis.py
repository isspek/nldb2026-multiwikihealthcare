import ast
import re
import tldextract
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from code.multi_wikimed_care.langs import use_case_langs

def count_references(references):
    references = ast.literal_eval(references)
    return len(references)

def extract_domains(references):
    pattern_web = r"https:\/\/web\.archive\.org\/web\/\d+\/(https?:\/\/.+)"
    # pattern_docs = r"https:\/\/archive\.org\/details\/([^\/\s]+(?:\/page\/n\d+)?)"
    pattern_is =  r"https:\/\/archive\.is\/\d+\/(https?:\/\/.+)"
    references = ast.literal_eval(references)
    domains = []
    for reference in references:
        match = re.match(pattern_web, reference)
        if match:
            reference=match.group(1)
        else:
            match = re.match(pattern_is, reference)
            if match:
                reference = match.group(1)
            # else:
            #     match = re.match(pattern_docs, reference)
            #     if match:
            #         reference = match.group(1)
        extracted = tldextract.extract(reference)
        domain = f"{extracted.domain}.{extracted.suffix}"
        if 'archive' in domain:
            pass
        if 'google' in domain:
            pass
        domains.append(domain)
    return domains

if __name__ == '__main__':
    data = pd.read_csv('data/multi-wikimedcare/all_related_entities_wikipages_references.csv')
    data = data.dropna(subset=['wikipage_de', 'wikipage_tr', 'wikipage_zh',
           'wikipage_it', 'wikipage_en'], how='any')

    # # Basic Stats
    # for lang in use_case_langs:
    #     data[f'{lang}_count'] = data[f'{lang}_references'].apply(lambda x: count_references(x))
    # count_cols = [f'{lang}_count' for lang in use_case_langs]
    # count_df = data[count_cols]
    # count_long = count_df.melt(var_name='Language', value_name='ReferenceCount')
    # count_long['Language'] = count_long['Language'].str.replace('_count', '')
    #
    # langs = count_long['Language'].unique()
    #
    # pastel_palette = sns.color_palette('pastel', n_colors=len(langs))
    # custom_palette = dict(zip(langs, pastel_palette))
    # plt.figure(figsize=(12, 6))
    # sns.swarmplot(data=count_long, x='Language', y='ReferenceCount', size=3, palette=custom_palette)
    #
    # plt.xlabel('Language')
    # plt.ylabel('Number of References')
    # plt.yscale('log')
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig("reference_counts_per_language.pdf")

    # Detailed Analysis
    for lang in use_case_langs:
        all_domains= []
        for refs in data[f'{lang}_references'].to_dict().values():
            if len(refs)>0:
                all_domains.extend(extract_domains(refs))

        top_domains = Counter(all_domains).most_common(10)

        print(f'Top Domain Langs {lang}')
        for domain, count in top_domains:
            print(f"{domain}: {count}")


