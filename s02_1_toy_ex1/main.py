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
Brussels	locatedIn	Belgium
Belgium	partOf	EU
EU	hasCapital	Brussels
""".strip()

triples = np.array([triple.split() for triple in triples.split("\n")])
tf = TriplesFactory.from_labeled_triples(triples=triples)

results = pipeline(
    training=tf,
    testing=tf,
    model="TransF",
    loss="softplus",
    model_kwargs=dict(embedding_dim=2),
    optimizer_kwargs=dict(lr=1.0e-1),
    training_kwargs=dict(num_epochs=60, use_tqdm_batch=False),
    evaluation_kwargs=dict(use_tqdm=False),
    random_seed=1,
    device="cpu",
)

results.plot(
    er_kwargs=dict(
        plot_relations=True,
        plot_entities=True,
        entity_embedding_getter=entity_embedding_getter_TransE,
        relation_embedding_getter=relation_embedding_getter_TransE,
    )
)
