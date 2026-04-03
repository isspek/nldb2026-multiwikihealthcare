# MultiWikiHealthCare

This repository contains the source code for the dataset generation and the experiments of the paper entitled "Zoom In Disparities in Healthcare LLM Q&A".


## Dataset Construction
To find the wikipage ids in other languages, run the following command:

```shell
python -m code.multi_wikimed_care.find_wpages_other_lang
```
This will produce `all_related_entities_wikipages.csv`.

We apply another filtering step based on Wikipage categories to remove unrelevant pages.

```shell
python -m code.multi_wikimed_care.fetch_wpage_category
```
This will produce `all_related_entities_wikipages_wcategories.csv`.

Remove the unrelated files

```shell
python -m code.multi_wikimed_care.remove_wpages
```

### Fetching References
```shell
python -m code.multi_wikimed_care.fetch_references
```
This will produce `data/multi-wikimedcare/all_related_entities_wikipages_references.csv`.

Analysis of the references

```shell
python -m code.multi_wikimed_care.references_analysis
```

### Sections Analysis

```shell
python -m code.multi_wikimed_care.sections_analysis
```

### Fact Extraction
The fact construction is largely based on [InfoGap](https://github.com/smfsamir/infogap). The additional functions related to filter can be found under `source_code/multi_wikimed_care`.

## Dataset

The [dataset](https://drive.google.com/drive/folders/1PcF66qPrUaTEWOosPT474zwqKsf9sRSb?usp=sharing) might contain inaccurate healthcare-related information. It is shared strictly for research purposes only.

The files containing `option` are the results of the case study on the paper:
- `option 1`: contextual information from Wiki pages.
- `option 2`: contextual information from RAG.

If you have questions, please contact `ibarschATdoctor.upv.es` or open a GitHub issue.


## Citation
Coming soon.