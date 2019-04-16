import torch

# citation: https://stackoverflow.com/questions/49710537/pytorch-gensim-how-to-load-pre-trained-word-embeddings

def from_pretrained(embeddings, freeze=True):
    assert embeddings.dim() == 2, \
         'Embeddings parameter is expected to be 2-dimensional'
    rows, cols = embeddings.shape
    embedding = torch.nn.Embedding(num_embeddings=rows, embedding_dim=cols)
    embedding.weight = torch.nn.Parameter(embeddings)
    embedding.weight.requires_grad = not freeze
    return embedding