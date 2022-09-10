import os

import numpy as np
import pykeen
from matplotlib import pyplot as plt

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory


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
job1 has covid
job2 has covid
job3 has CB
job4 has aids
job4 has covid
assign before appoint
appoint before pickup
pickup before dropoff
""".strip()

triples = np.array([triple.split() for triple in triples.split("\n")])
tf = TriplesFactory.from_labeled_triples(triples=triples)

results = pipeline(
    training=tf,
    testing=tf,
    model="TransE",
    loss="softplus",
    model_kwargs=dict(embedding_dim=4),
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
        entity_embedding_getter = entity_embedding_getter_TransE,
        relation_embedding_getter = relation_embedding_getter_TransE, 
    )
)


