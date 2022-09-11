import os
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import pickle

#os.makedirs("results", exist_ok=True)

triples = """
job1 t1 appoint 
job1 t2 assign
job1 t3 pickup
job1 t4 dropoff
""".strip()


triples = np.array([triple.split() for triple in triples.split("\n")])
tf = TriplesFactory.from_labeled_triples(triples=triples)

results = pipeline(
    training=tf,
    testing=tf,
    model="TransR",
    loss="softplus",
    model_kwargs=dict(embedding_dim=2, relation_dim=2),
    optimizer_kwargs=dict(lr=1.0e-1),
    training_kwargs=dict(num_epochs=100, use_tqdm_batch=False),
    evaluation_kwargs=dict(use_tqdm=False),
    random_seed=1,
    device="cpu",
)

results.plot()

results.save_to_directory('nations_transR')

data_store = dict(tf=tf, model=results.model, training=results.training)
filePath = os.path.join(os.getcwd(), "data_store.pickle")
with open(filePath, "wb") as handle:
    pickle.dump(data_store, handle, protocol=pickle.HIGHEST_PROTOCOL)



