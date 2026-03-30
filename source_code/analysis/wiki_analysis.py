import pandas as pd
import requests
import re
import tldextract
import whois
import json
from tqdm import tqdm
from scipy.stats import ttest_rel
from pathlib import Path
from langdetect import detect
from bs4 import BeautifulSoup

'''
python -m source_code.analysis.wiki_analysis
'''

def get_external_links_from_wikipedia(page_title, lang="en"):
    """
    Returns a list of external link URLs from the given Wikipedia page.
    page_title: e.g. "Coffee"
    lang: language code, e.g. "en", "tr"
    """
    API_URL = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "page": page_title,
        "prop": "externallinks",
        "format": "json"
    }
    headers = {
        "User-Agent": "MyWikipediaScript/1.0 (https://example.com/contact)"
    }

    resp = requests.get(API_URL, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    # The 'parse' key should contain 'externallinks' (if there are any)
    links = data.get("parse", {}).get("externallinks", [])
    return links

def extract_references(infogap_file, trg_lang, outfile):
    infogap_data = pd.read_csv(infogap_file)
    outfile = Path(outfile)
    entities = infogap_data['wikipage_en'].unique()
    trg_references = []
    en_references = []
    for entity in tqdm(entities, total=len(entities)):
        entity_other_lang = infogap_data[infogap_data['wikipage_en'] == entity][f'wikipage_{trg_lang}'].values[0]
        ext_links = get_external_links_from_wikipedia(entity_other_lang, lang=trg_lang)
        for ext_link in ext_links:
            trg_references.append({'entity': entity, 'link': ext_link})

        ext_links = get_external_links_from_wikipedia(entity, lang='en')
        for ext_link in ext_links:
            en_references.append({'entity': entity, 'link': ext_link})

    trg_references = pd.DataFrame(trg_references)
    en_references = pd.DataFrame(en_references)
    trg_references.to_csv(outfile / f'{trg_lang}_references.csv', index=False)
    en_references.to_csv(outfile / f'{trg_lang}_en_references.csv', index=False)


def is_english_domain(domain):
    # Remove TLD for analysis
    name = domain.split('.')[0]
    return bool(re.match(r'^[a-zA-Z0-9-]+$', name))

def detect_language_from_domain(domain):
    name = domain.split('.')[0]
    try:
        lang = detect(name)
        return lang
    except:
        return "unknown"

def wiki_factuality_analysis(infogap_file, trg_lang, output_file):
    if not Path(output_file).exists():
        infogap_data = pd.read_csv(infogap_file)
        entities = infogap_data['wikipage_en']

        facts_trg_dir = Path(f'data/multi-wikimedcare/facts/{trg_lang}_facts')
        facts_en_dir = Path(f'data/multi-wikimedcare/facts/en_facts')

        trg_ent_data = []
        en_ent_data = []
        for entity in tqdm(entities, total=len(entities)):
            if '/' in entity:
                _entity = entity.replace('/', ' ')
            else:
                _entity = entity
            ent_file = facts_trg_dir / f'{_entity}_{trg_lang}_gpt-4o-mini_facts.json'

            with open(ent_file) as f:
                data = json.load(f)
                num_paragraph = len(data.keys())

                num_facts = 0
                for val in data.values():
                    num_facts+= len(val)

                trg_ent_data.append({
                    'entity': entity,
                    'num_paragraph': num_paragraph,
                    'num_facts': num_facts
                })

            ent_file = facts_en_dir / f'{_entity}_en_gpt-4o-mini_facts.json'

            with open(ent_file) as f:
                data = json.load(f)
                num_paragraph = len(data.keys())

                num_facts = 0
                for val in data.values():
                    num_facts+= len(val)

                en_ent_data.append({
                    'entity': entity,
                    'num_paragraph': num_paragraph,
                    'num_facts': num_facts
                })

        trg_ent_data = pd.DataFrame(trg_ent_data)
        en_ent_data = pd.DataFrame(en_ent_data)

        merged = pd.merge(trg_ent_data, en_ent_data, on="entity", suffixes=("_trg", "_en"))

        merged.to_csv(output_file, index=False)

        # # Paired t-test on number of paragraphs
        # t_stat, p_t = ttest_rel(merged["num_paragraph_trg"], merged["num_paragraph_en"])
        # print("Paired t-test p-value (num_paragraph):", p_t)
        #
        # if p_t < 0.05:
        #     print('Statistically significant difference')
        #
        # summary = merged[["num_paragraph_trg", "num_paragraph_en"]].describe().T
        # summary["mean_ratio_en/trg"] = (
        #     summary.loc["num_paragraph_en", "mean"] / summary.loc["num_paragraph_trg", "mean"]
        # )
        #
        # print(summary)
        #
        # merged["paragraph_diff"] = merged["num_paragraph_trg"] - merged["num_paragraph_en"]
        # top10_longer = (
        #     merged[merged["paragraph_diff"] > 0]
        #     .sort_values("paragraph_diff", ascending=False)
        #     .head(10)
        # )
        # print("Top 10 entities with longer paragraphs in target language:")
        # print(top10_longer[["entity", "num_paragraph_trg", "num_paragraph_en", "paragraph_diff"]])
        #
        #
        # print('Factual Analysis')
        # t_stat, p_t = ttest_rel(merged["num_facts_trg"], merged["num_facts_en"])
        # print("Paired t-test p-value (num_facts):", p_t)
        #
        # if p_t < 0.05:
        #     print('Statistically significant difference')
        #
        # summary = merged[["num_facts_trg", "num_facts_en"]].describe().T
        # summary["mean_ratio_en/trg"] = (
        #     summary.loc["num_facts_en", "mean"] / summary.loc["num_facts_trg", "mean"]
        # )
        #
        # print(summary)
        #
        # merged["facts_diff"] = merged["num_facts_trg"] - merged["num_facts_en"]
        # top10_longer = (
        #     merged[merged["facts_diff"] > 0]
        #     .sort_values("facts_diff", ascending=False)
        #     .head(10)
        # )
        # print("Top 10 entities with longer facts in target language:")
        # print(top10_longer[["entity", "num_facts_trg", "num_facts_en", "facts_diff"]])


def wiki_page_analysis(infogap_file, trg_lang, output_file):
    if not Path(output_file).exists():
        infogap_data = pd.read_csv(infogap_file)
        entities = infogap_data['wikipage_en'].unique()

        entity_dir = Path(f'data/multi-wikimedcare/html/')

        trg_ent_data = []
        en_ent_data = []
        for entity in tqdm(entities, total=len(entities)):
            _entity = infogap_data[infogap_data['wikipage_en']==entity]['entity'].values[0]
            ent_file = entity_dir / trg_lang / f'{_entity}.txt'

            with open(ent_file) as ent_file:
                ent_content = ent_file.read()
                soup = BeautifulSoup(ent_content, 'html.parser')
                headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                for heading in headings:
                    trg_ent_data.append({"section": heading.text,
                                           "entity": entity
                                           })

            ent_file = entity_dir / 'en' / f'{_entity}.txt'

            with open(ent_file) as ent_file:
                ent_content = ent_file.read()
                soup = BeautifulSoup(ent_content, 'html.parser')
                headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                for heading in headings:
                    en_ent_data.append({"section": heading.text,
                                           "entity": entity
                                           })

        trg_ent_data = pd.DataFrame(trg_ent_data)
        en_ent_data = pd.DataFrame(en_ent_data)

        ref_counts = trg_ent_data.groupby("entity")["section"].count().reset_index(name="section_ref_count")
        en_counts = en_ent_data.groupby("entity")["section"].count().reset_index(name="section_en_count")
        merged = pd.merge(ref_counts, en_counts, on="entity", how="inner")

        merged.to_csv(output_file, index=False)

        # t_stat, p_t = ttest_rel(merged["en_count"], merged["ref_count"])
        # print("Paired t-test p-value:", p_t)
        #
        # if p_t <0.05:
        #     print('Statistically significant difference')
        #
        # summary = merged[["ref_count", "en_count"]].describe().T
        # summary["mean_ratio"] = summary["mean"]["en_count"] / summary["mean"]["ref_count"]
        # print(summary)


def domain_analysis(reference_file, en_reference_file, domain_info_file, trg_lang, outfile):
    reference_data = pd.read_csv(reference_file)
    print(f'Number of reference urls {len(reference_data)}')
    en_reference_data = pd.read_csv(en_reference_file)
    # print(en_reference_data.head())
    print(f'Number of reference urls in English {len(en_reference_data)}')
    ref_counts = reference_data.groupby("entity")["link"].count().reset_index(name="ref_count")
    en_counts = en_reference_data.groupby("entity")["link"].count().reset_index(name="en_count")

    merged = pd.merge(ref_counts, en_counts, on="entity", how="inner")
    merged.to_csv(outfile)
    # t_stat, p_t = ttest_rel(merged["en_count"], merged["ref_count"])
    # print("Paired t-test p-value:", p_t)
    #
    # if p_t <0.05:
    #     print('Statistically significant difference')
    #
    # more_refs_than_english = merged[merged["ref_count"] > merged["en_count"]]
    # print(f"Number of entities with more total references than English: {len(more_refs_than_english)}")
    # print(more_refs_than_english)
    #
    # summary = merged[["ref_count", "en_count"]].describe().T
    # summary["mean_ratio"] = summary["mean"]["en_count"] / summary["mean"]["ref_count"]
    # print(summary)
    #
    # if not Path(domain_info_file).exists():
    #     print('Domain info does not exist!!!')
    #     results = []
    #     for _, row in tqdm(reference_data.iterrows(), total=len(reference_data)):
    #         link = row['link']
    #         entity = row['entity']
    #
    #         extracted = tldextract.extract(link)
    #         domain = f"{extracted.domain}.{extracted.suffix}"
    #         english_flag = is_english_domain(domain)
    #         language = detect_language_from_domain(domain) if english_flag else "non-latin"
    #
    #         results.append({
    #             "entity": entity,
    #             "link": link,
    #             "domain": domain,
    #             "english_domain": english_flag,
    #             "detected_language": language,
    #             'trg_lang': trg_lang
    #         })
    #
    #     for _, row in tqdm(en_reference_data.iterrows(), total=len(en_reference_data)):
    #         link = row['link']
    #         entity = row['entity']
    #
    #         extracted = tldextract.extract(link)
    #         domain = f"{extracted.domain}.{extracted.suffix}"
    #         english_flag = is_english_domain(domain)
    #         language = detect_language_from_domain(domain) if english_flag else "non-latin"
    #
    #         results.append({
    #             "entity": entity,
    #             "link": link,
    #             "domain": domain,
    #             "english_domain": english_flag,
    #             "detected_language": language,
    #             'trg_lang': 'en'
    #         })
    #
    #     output = pd.DataFrame(results)
    #     output.to_csv(domain_info_file, index=False)
    #
    #
    # domain_info_data = pd.read_csv(domain_info_file)
    # trg_domain = domain_info_data[domain_info_data['trg_lang'] == trg_lang]
    #
    # top_entities = (
    #     trg_domain['entity']
    #     .value_counts()
    #     .reset_index()
    #     .rename(columns={'index': 'entity', 'entity': 'link_count'})
    #     .head(20)
    # )
    #
    # print(top_entities)
    #
    # # Count how many times each domain appears
    # top_domains = (
    #     trg_domain['domain']
    #     .value_counts()
    #     .reset_index()
    #     .rename(columns={'index': 'domain', 'domain': 'count'})
    # )
    #
    # # Show the top 20 domains
    # print(top_domains.head(20))
    #
    # non_en_link = domain_info_data[domain_info_data['link'].str.contains(fr'\.{trg_lang}', na=False)]
    #
    # non_en_counts = (
    #     non_en_link['entity']
    #     .value_counts()
    #     .reset_index()
    #     .rename(columns={'index': 'entity', 'entity': 'link_count'})
    # )

def construct_wikipage_analysis(infogap_file, trg_lang):
    infogap_data = pd.read_csv(infogap_file)
    entity_dir = Path(f'data/multi-wikimedcare/html/')

    facts_trg_dir = Path(f'data/multi-wikimedcare/facts/{trg_lang}_facts')
    facts_en_dir = Path(f'data/multi-wikimedcare/facts/en_facts')

    processed_data = []

    for row in tqdm(infogap_data.to_dict(orient='records'), total=len(infogap_data)):
        entity = row['entity']
        if '/' in entity:
            _entity = entity.replace('/', ' ')
        else:
            _entity = entity
        ent_file = entity_dir / trg_lang / f'{_entity}.txt'
        section_trg_count = 0

        with open(ent_file) as ent_file:
            ent_content = ent_file.read()
            soup = BeautifulSoup(ent_content, 'html.parser')
            headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            for heading in headings:
                section_trg_count+=1

        ent_file = entity_dir / 'en' / f'{_entity}.txt'
        section_en_count = 0
        with open(ent_file) as ent_file:
            ent_content = ent_file.read()
            soup = BeautifulSoup(ent_content, 'html.parser')
            headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            for heading in headings:
                section_en_count+=1

        wikipage_en = row['wikipage_en']

        if '/' in wikipage_en:
            wikipage_en = wikipage_en.replace('/', ' ')
        else:
            wikipage_en = wikipage_en

        ent_file = facts_trg_dir / f'{wikipage_en}_{trg_lang}_gpt-4o-mini_facts.json'

        num_facts_trg = 0
        with open(ent_file) as f:
            data = json.load(f)
            num_paragraph_trg = len(data.keys())
            for val in data.values():
                num_facts_trg += len(val)

        ent_file = facts_en_dir / f'{wikipage_en}_en_gpt-4o-mini_facts.json'

        num_facts_en = 0
        with open(ent_file) as f:
            data = json.load(f)
            num_paragraph_en = len(data.keys())

            for val in data.values():
                num_facts_en += len(val)

        entity_other_lang = infogap_data[infogap_data['entity']==entity][f'wikipage_{trg_lang}'].values[0]
        entity_en = infogap_data[infogap_data['entity']==entity][f'wikipage_en'].values[0]
        trg_ref_count = len(get_external_links_from_wikipedia(entity_other_lang, lang=trg_lang))
        en_ref_count  = len(get_external_links_from_wikipedia(entity_en, lang='en'))

        row['section_trg_count'] = section_trg_count
        row['section_en_count'] = section_en_count
        row['num_paragraph_trg'] = num_paragraph_trg
        row['num_paragraph_en'] = num_paragraph_en
        row['num_facts_trg'] = num_facts_trg
        row['num_facts_en'] = num_facts_en
        row['trg_ref_count'] = trg_ref_count
        row['en_ref_count'] = en_ref_count
        processed_data.append(row)

    return pd.DataFrame(processed_data)


if __name__ == '__main__':
    infogap_file = 'data/multi-wikimedcare/infogap_full.csv'
    # extract_references(infogap_file=infogap_file, trg_lang='tr', outfile='data/multi-wikimedcare/')
    # extract_references(infogap_file=infogap_file, trg_lang='de', outfile='data/multi-wikimedcare/')
    # extract_references(infogap_file=infogap_file, trg_lang='it', outfile='data/multi-wikimedcare/')
    # extract_references(infogap_file=infogap_file, trg_lang='zh', outfile='data/multi-wikimedcare/')

    analyzed_data = construct_wikipage_analysis(infogap_file, trg_lang= 'tr')
    analyzed_data.to_csv('data/multi-wikimedcare/wiki_full_tr.csv')
    print('Tr data is saved')

    analyzed_data = construct_wikipage_analysis(infogap_file, trg_lang= 'de')
    analyzed_data.to_csv('data/multi-wikimedcare/wiki_full_de.csv')
    print('de data is saved')

    analyzed_data = construct_wikipage_analysis(infogap_file, trg_lang= 'zh')
    analyzed_data.to_csv('data/multi-wikimedcare/wiki_full_zh.csv')
    print('zh data is saved')

    analyzed_data = construct_wikipage_analysis(infogap_file, trg_lang= 'it')
    analyzed_data.to_csv('data/multi-wikimedcare/wiki_full_it.csv')
    print('it data is saved')

    # print('Analysis on Turkish Wikipage')
    # domain_analysis(reference_file='data/multi-wikimedcare/tr_references.csv',
    #                 en_reference_file='data/multi-wikimedcare/tr_en_references.csv',
    #                 domain_info_file='data/multi-wikimedcare/tr_en_domain_info_references.csv',
    #                 outfile='data/multi-wikimedcare/tr_en_links.csv',
    #                 trg_lang = 'tr')
    # wiki_page_analysis(infogap_file=infogap_file, trg_lang='tr', output_file='data/multi-wikimedcare/tr_en_section_analysis.csv')
    # wiki_factuality_analysis(infogap_file, trg_lang='tr', output_file='data/multi-wikimedcare/tr_en_factuality_analysis.csv')
    #
    #
    # print('General data')
    # domain_info_data = pd.read_csv('data/multi-wikimedcare/tr_en_links.csv')
    # wikipage_data = pd.read_csv('data/multi-wikimedcare/tr_en_section_analysis.csv')
    # factuality_data = pd.read_csv('data/multi-wikimedcare/tr_en_factuality_analysis.csv')
    #
    # # entity, link, domain, english_domain, trtected_language, trg_lang
    # print(domain_info_data.head())
    # print(wikipage_data.head())
    # print(factuality_data.head())
    #
    # merged_data = (
    #     wikipage_data
    #     .merge(factuality_data, on="entity", how="inner")
    #     .merge(domain_info_data, on="entity", how="inner")
    # )
    # merged_data.to_csv('data/multi-wikimedcare/wiki_full_tr.csv', index=False)

    # # print('Analysis on German Wikipage')
    # domain_analysis(reference_file='data/multi-wikimedcare/de_references.csv',
    #                 en_reference_file='data/multi-wikimedcare/de_en_references.csv',
    #                 domain_info_file='data/multi-wikimedcare/de_en_domain_info_references.csv',
    #                 outfile='data/multi-wikimedcare/de_en_links.csv',
    #                 trg_lang = 'de')
    # wiki_page_analysis(infogap_file=infogap_file, trg_lang='de', output_file='data/multi-wikimedcare/de_en_section_analysis.csv')
    # wiki_factuality_analysis(infogap_file, trg_lang='de', output_file='data/multi-wikimedcare/de_en_factuality_analysis.csv')
    #
    #
    # print('General data')
    # domain_info_data = pd.read_csv('data/multi-wikimedcare/de_en_links.csv')
    # wikipage_data = pd.read_csv('data/multi-wikimedcare/de_en_section_analysis.csv')
    # factuality_data = pd.read_csv('data/multi-wikimedcare/de_en_factuality_analysis.csv')
    #
    # # entity, link, domain, english_domain, detected_language, trg_lang
    # print(domain_info_data.head())
    # print(wikipage_data.head())
    # print(factuality_data.head())
    #
    # merged_data = (
    #     wikipage_data
    #     .merge(factuality_data, on="entity", how="inner")
    #     .merge(domain_info_data, on="entity", how="inner")
    # )
    # merged_data.to_csv('data/multi-wikimedcare/wiki_full_de.csv', index=False)

    #
    # print('Analysis on Chinese Wikipage')
    # domain_analysis(reference_file='data/multi-wikimedcare/zh_references.csv',
    #                 en_reference_file='data/multi-wikimedcare/zh_en_references.csv',
    #                 domain_info_file='data/multi-wikimedcare/zh_en_domain_info_references.csv',
    #                 outfile='data/multi-wikimedcare/zh_en_links.csv',
    #                 trg_lang = 'zh')
    # wiki_page_analysis(infogap_file=infogap_file, trg_lang='zh', output_file='data/multi-wikimedcare/zh_en_section_analysis.csv')
    # wiki_factuality_analysis(infogap_file, trg_lang='zh', output_file='data/multi-wikimedcare/zh_en_factuality_analysis.csv')
    #
    #
    # print('General data')
    # domain_info_data = pd.read_csv('data/multi-wikimedcare/zh_en_links.csv')
    # wikipage_data = pd.read_csv('data/multi-wikimedcare/zh_en_section_analysis.csv')
    # factuality_data = pd.read_csv('data/multi-wikimedcare/zh_en_factuality_analysis.csv')
    #
    # # entity, link, domain, english_domain, zhtected_language, zhg_lang
    # print(domain_info_data.head())
    # print(wikipage_data.head())
    # print(factuality_data.head())
    #
    # merged_data = (
    #     wikipage_data
    #     .merge(factuality_data, on="entity", how="inner")
    #     .merge(domain_info_data, on="entity", how="inner")
    # )
    # merged_data.to_csv('data/multi-wikimedcare/wiki_full_zh.csv', index=False)
    #
    # print('Analysis on Italian Wikipage')
    # domain_analysis(reference_file='data/multi-wikimedcare/it_references.csv',
    #                 en_reference_file='data/multi-wikimedcare/it_en_references.csv',
    #                 domain_info_file='data/multi-wikimedcare/it_en_domain_info_references.csv',
    #                 outfile='data/multi-wikimedcare/it_en_links.csv',
    #                 trg_lang = 'it')
    # wiki_page_analysis(infogap_file=infogap_file, trg_lang='it', output_file='data/multi-wikimedcare/it_en_section_analysis.csv')
    # wiki_factuality_analysis(infogap_file, trg_lang='it', output_file='data/multi-wikimedcare/it_en_factuality_analysis.csv')
    #
    #
    # print('General data')
    # domain_info_data = pd.read_csv('data/multi-wikimedcare/it_en_links.csv')
    # wikipage_data = pd.read_csv('data/multi-wikimedcare/it_en_section_analysis.csv')
    # factuality_data = pd.read_csv('data/multi-wikimedcare/it_en_factuality_analysis.csv')
    #
    # # entity, link, domain, english_domain, ittected_language, itg_lang
    # print(domain_info_data.head())
    # print(wikipage_data.head())
    # print(factuality_data.head())
    #
    # merged_data = (
    #     wikipage_data
    #     .merge(factuality_data, on="entity", how="inner")
    #     .merge(domain_info_data, on="entity", how="inner")
    # )
    # merged_data.to_csv('data/multi-wikimedcare/wiki_full_it.csv', index=False)
    #
    # print(merged_data.columns)