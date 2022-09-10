from pykeen.pipeline import pipeline
from pykeen.datasets import Nations
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

result = pipeline(
    dataset='Nations',
    model='TransE',
)

ds = Nations()
model = result.model
embedding = model.entity_representations[0]()
embedding = embedding.cpu().detach().numpy()

ent_to_id = result.training.entity_to_id
id_to_ent = {v: k for k, v in ent_to_id.items()}


tsne = TSNE(n_components=2, perplexity=10, random_state=0)
Y = tsne.fit_transform(embedding)
plt.figure(figsize=(5, 5), dpi=300)

plt.scatter(Y[:, 0], Y[:, 1])

y = np.arange(embedding.shape[0])
for idx, x, y in zip(y, Y[:, 0], Y[:, 1]):
    
    label = id_to_ent[idx]
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
    
    