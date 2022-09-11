import os

import numpy as np
import pandas as pd
import pykeen
from matplotlib import pyplot as plt

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

from sklearn.manifold import TSNE
import umap


def entity_embedding_getter_TransE(model) -> pykeen.nn.Embedding:
    entity_representations_embedding = model.entity_representations[0]
    assert isinstance(entity_representations_embedding, pykeen.nn.Embedding)
    return entity_representations_embedding


def relation_embedding_getter_TransE(model) -> pykeen.nn.Embedding:
    relation_representations_embedding = model.relation_representations[0]
    assert isinstance(relation_representations_embedding, pykeen.nn.Embedding)
    return relation_representations_embedding


os.makedirs("results", exist_ok=True)

triples = """
job1 t1 appoint 
job1 t2 assign
job1 t3 pickup
job1 t4 dropoff
job2 t4 appoint 
job2 t5 assign
job2 t6 pickup
job2 t7 dropoff
job3 t8 appoint 
job3 t9 assign
job3 t2 pickup
job3 t10 dropoff
job4 t1 appoint 
job4 t2 assign
job4 t3 pickup
job4 t10 dropoff
job12 t1 appoint 
job12 t2 assign
job12 t3 pickup
job12 t4 dropoff
job13 t1 appoint 
job13 t2 assign
job13 t3 pickup
job13 t4 dropoff
job14 t1 appoint 
job14 t2 assign
job14 t3 pickup
job14 t4 dropoff
job15 t1 appoint 
job15 t2 assign
job15 t3 pickup
job15 t4 dropoff
job1 has covid
job2 has covid
job3 has CB
job4 has aids
job4 has covid
job22 t4 appoint 
job22 t5 assign
job22 t6 pickup
job22 t7 dropoff
job22 has covid
job32 t8 appoint 
job32 t9 assign
job32 t2 pickup
job32 t10 dropoff
assign before appoint
appoint before pickup
pickup before dropoff
CB in special
aids in special
covid in special
job1 in jobs
job2 in jobs
job3 in jobs
job4 in jobs
job12 in jobs
job13 in jobs
job14 in jobs
job15 in jobs
job22 in jobs
job32 in jobs
appoint in tasks
assign in tasks
pickup in tasks
dropoff in tasks
jobs has tasks
jobs has special
""".strip()



triples = np.array([triple.split() for triple in triples.split("\n")])
tf = TriplesFactory.from_labeled_triples(triples=triples)

results = pipeline(
    training=tf,
    testing=tf,
    model="TransR",
    loss="softplus",
    model_kwargs=dict(embedding_dim=20),
    optimizer_kwargs=dict(lr=1.0e-1),
    training_kwargs=dict(num_epochs=100, use_tqdm_batch=False),
    evaluation_kwargs=dict(use_tqdm=False),
    random_seed=1,
    device="cpu",
)

results.plot(
    er_kwargs=dict(
        plot_relations=False,
        plot_entities=True,
        entity_embedding_getter=entity_embedding_getter_TransE,
        relation_embedding_getter=relation_embedding_getter_TransE,
    )
)


embedding = entity_embedding_getter_TransE(results.model)()
embedding = embedding.cpu().detach().numpy()


ent_to_id = results.training.entity_to_id
id_to_ent = {v: k for k, v in ent_to_id.items()}
dataArr = []
for k, v in ent_to_id.items():
    dataArr.append({'idx': v, 'entity': k})
df_ent = pd.DataFrame.from_records(dataArr)


filt = df_ent['entity'].str.contains('job')
jobIdx = df_ent[filt]['idx'].values


embedding = embedding[jobIdx, :]
labels = df_ent[filt]['entity'].values
tsne = TSNE(n_components=2, perplexity=3, random_state=0)
ebr = tsne.fit_transform(embedding)


fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].scatter(ebr[:, 0], ebr[:, 1])
for label, x, y in zip(labels, ebr[:, 0], ebr[:, 1]):
    axs[0].annotate(label, xy=(x, y), xytext=(
        0, 0), textcoords="offset points")


trans = umap.UMAP(n_neighbors=3, random_state=42).fit(embedding)
ebr = trans.embedding_
axs[1].scatter(ebr[:, 0], ebr[:, 1])
for label, x, y in zip(labels, ebr[:, 0], ebr[:, 1]):
    axs[1].annotate(label, xy=(x, y), xytext=(
        0, 0), textcoords="offset points")
