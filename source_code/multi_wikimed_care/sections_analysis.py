import dash
from dash import dcc, html, Input, Output
# import seaborn as sns
# from collections import Counter
# import matplotlib.pyplot as plt
import pandas as pd
# import numpy as np
import plotly.express as px
# from bs4 import BeautifulSoup
from pathlib import Path
import pickle
# from transformers import AutoTokenizer, AutoModel
# from source_code.multi_wikimed_care.langs import use_case_langs
from sklearn.manifold import TSNE
import pickle
import plotly.io as pio
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

pio.renderers.default = 'browser'

# def embed_text(text_list, tokenizer, model):
#     inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')
#     embeddings = model(**inputs).last_hidden_state.mean(dim=1)
#     return embeddings.detach().numpy()
#
# processed_data = []
# for lang in use_case_langs:
#     files_dir = Path(f'data/multi-wikimedcare/html/{lang}')
#     for fname in files_dir.rglob('*.txt'):
#         with open(fname, 'r') as file:
#             content = file.read()
#             soup = BeautifulSoup(content, 'html.parser')
#             # headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
#             headings = soup.find_all(['h2'])
#             for heading in headings:
#                 level = int(heading.name[1])
#                 processed_data.append({"section": heading.text,
#                                        "lang": lang,
#                                        "file": fname.name
#                                        })
#
# processed_data = pd.DataFrame(processed_data)
# sections = processed_data['section'].tolist()
#
# print(f'Number of total sections are {len(sections)}')
#
# counter = Counter(sections)
# filtered = Counter({item: count for item, count in counter.items() if count > 1})
#
# # Create filtered dataframe
# filtered_data = processed_data[processed_data['section'].isin(filtered.keys())]
# texts = list(filtered_data['section'].unique())

# === EMBEDDING PART ===
# if you don't embeddings.pkl, uncomment the below code
output_dir = 'data/multi-wikimedcare'  # <-- Define your desired output directory
# embedding = 'Qwen/Qwen3-Embedding-8B'   # <-- e.g., 'sentence-transformers/all-MiniLM-L6-v2'
#
# Path(output_dir).mkdir(parents=True, exist_ok=True)
#
# tokenizer = AutoTokenizer.from_pretrained(embedding, trust_remote_code=True)
# model = AutoModel.from_pretrained(embedding, trust_remote_code=True)
#
# embeddings = embed_text(texts, tokenizer=tokenizer, model=model)
#
# with open("data/multi-wikimedcare/sections_embeddings.pkl", "wb") as f:
#     pickle.dump({"texts": texts, "embeddings": embeddings}, f)

# Load later
with open("data/multi-wikimedcare/sections_embeddings_me5.pkl", "rb") as f:
    data = pickle.load(f)
    texts = data["texts"]
    embeddings = data["embeddings"]

print(f'Number of sections {len(texts)}')
# print(texts)
n_clusters = 40  # Adjust this based on your dataset
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
normalized_embeddings = normalize(embeddings, norm='l2')
labels = kmeans.fit_predict(normalized_embeddings)

tsne = TSNE(n_components=2, random_state=42)
reduced = tsne.fit_transform(embeddings)

df = pd.DataFrame({
    'x': reduced[:, 0],
    'y': reduced[:, 1],
    'text': texts,
    'cluster': labels.astype(str)
})

# ---------- Initialize Dash App ----------
app = dash.Dash(__name__)

# Scatter Plot
fig = px.scatter(df, x='x', y='y', color='cluster', hover_name='text',
                 title="Section Clustering", height=600,
                 custom_data=['cluster'])
fig.update_traces(marker=dict(size=10, opacity=0.8))

# ---------- Layout ----------
app.layout = html.Div([
    html.H2("Interactive Cluster Viewer"),
    dcc.Graph(id='scatter', figure=fig, style={'width': '70%', 'display': 'inline-block'}),
    html.Div(id='cluster-output', style={'width': '28%', 'display': 'inline-block', 'verticalAlign': 'top',
                                         'padding': '10px', 'overflowY': 'scroll', 'height': '600px',
                                         'border': '1px solid #ccc'})
])

# ---------- Callback ----------
@app.callback(
    Output('cluster-output', 'children'),
    Input('scatter', 'clickData')
)
def display_cluster_texts(clickData):
    if clickData is None:
        return "Click on a point to see all texts in the same cluster."

    # Get cluster label from custom_data
    clicked_cluster = clickData['points'][0]['customdata'][0]

    # Get all texts in that cluster
    texts_in_cluster = df[df['cluster'] == str(clicked_cluster)]['text'].tolist()

    return [
        html.H4(f"Cluster {clicked_cluster} ({len(texts_in_cluster)} items)"),
        html.Ul([html.Li(text) for text in texts_in_cluster[:100]])  # Limit list
    ]

if __name__ == '__main__':
    app.run(debug=True)

# sections_per_file = processed_data.groupby(['lang', 'file']).size().reset_index(name='num_sections')
# mean_sections = sections_per_file.groupby('lang').agg(
#     files=('num_sections', 'count'),
#     mean_sections_per_file=('num_sections', 'mean'),
#     std_dev_sections=('num_sections', 'std')
# ).reset_index()
#
# plt.figure(figsize=(8, 5))
# sns.set(style="whitegrid")
#
# colors = sns.color_palette('pastel')
#
# plt.bar(
#     mean_sections['lang'],
#     mean_sections['mean_sections_per_file'],
#     yerr=mean_sections['std_dev_sections'],
#     capsize=5,
#     color=colors,
#     edgecolor='gray'
# )
#
# plt.xlabel('Language')
# plt.ylabel('Average Number of the Sections per Wiki Page')
# # plt.title('Average Number of Sections per File by Language (with Std Dev)')
# plt.tight_layout()
# plt.savefig("mean_sections_by_language.pdf", format="pdf", bbox_inches="tight")

