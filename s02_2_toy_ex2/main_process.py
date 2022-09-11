import os

import numpy as np
import pandas as pd
import pykeen
from matplotlib import pyplot as plt

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

from sklearn.manifold import TSNE
import umap
import pickle


def entity_embedding_getter_TransE(model) -> pykeen.nn.Embedding:
    entity_representations_embedding = model.entity_representations[0]
    assert isinstance(entity_representations_embedding, pykeen.nn.Embedding)
    return entity_representations_embedding


def relation_embedding_getter_TransE(model) -> pykeen.nn.Embedding:
    relation_representations_embedding = model.relation_representations[0]
    assert isinstance(relation_representations_embedding, pykeen.nn.Embedding)
    return relation_representations_embedding

def relation_map_getter(model):
    return model.relation_representations[1]

cwd = os.getcwd()
filePath = os.path.join(cwd,  "data_store.pickle")
with open(filePath, "rb") as handle:
    data_dict = pickle.load(handle)

tf = data_dict.get("tf")
model = data_dict.get("model")
training = data_dict.get("training")

emb_ent = entity_embedding_getter_TransE(model)()
emb_ent = emb_ent.cpu().detach().numpy()

emb_rel = relation_embedding_getter_TransE(model)()
emb_rel = emb_rel.cpu().detach().numpy()

mapping = relation_map_getter(model)()
mapping = mapping.cpu().detach().numpy()


ent_to_id = training.entity_to_id
id_to_ent = {v: k for k, v in ent_to_id.items()}
dataArr = []
for k, v in ent_to_id.items():
    dataArr.append({'idx': v, 'entity': k})
df_ent = pd.DataFrame.from_records(dataArr)

rel_to_id = training.relation_to_id
id_to_rel = {v: k for k, v in rel_to_id.items()}
dataArr = []
for k, v in rel_to_id.items():
    dataArr.append({'idx': v, 'relation': k})
df_rel = pd.DataFrame.from_records(dataArr)


rel_id = 3
rel_map = mapping[rel_id, : ,:]

arr = []
for i in range(emb_ent.shape[0]):
    ent = emb_ent[i,:]    
    ent_tran = rel_map.dot(ent)
    arr.append(ent_tran.reshape(-1,1))
ent_trans = np.concatenate(arr, axis=0)
    




# filt = df_ent['entity'].str.contains('job')
# jobIdx = df_ent[filt]['idx'].values


# embedding = embedding[jobIdx, :]
# labels = df_ent[filt]['entity'].values
# tsne = TSNE(n_components=2, perplexity=3, random_state=0)
# ebr = tsne.fit_transform(embedding)


# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].scatter(ebr[:, 0], ebr[:, 1])
# for label, x, y in zip(labels, ebr[:, 0], ebr[:, 1]):
#     axs[0].annotate(label, xy=(x, y), xytext=(
#         0, 0), textcoords="offset points")


# trans = umap.UMAP(n_neighbors=3, random_state=42).fit(embedding)
# ebr = trans.embedding_
# axs[1].scatter(ebr[:, 0], ebr[:, 1])
# for label, x, y in zip(labels, ebr[:, 0], ebr[:, 1]):
#     axs[1].annotate(label, xy=(x, y), xytext=(
#         0, 0), textcoords="offset points")
