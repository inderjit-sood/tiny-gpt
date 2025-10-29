import torch
import torch.nn as nn

#number of distinct tokens
vocab_size = 10

#token representation - 4d vector
embedding_dimensions = 4

#number of tokens processed at once by model
block_size = 6



token_embedding_table = nn.Embedding(vocab_size, embedding_dimensions)


position_embedding_table = nn.Embedding(block_size, embedding_dimensions)

#Sequences of tokens - 2 sequences each of block_size
token_ids = torch.randint(0, vocab_size, (2, block_size))
print("Token IDs:", token_ids)

#token embedding
token_embeddings = token_embedding_table(token_ids)

print("Token embedding:", token_embeddings)

#positional embedding 
positions = torch.arange(block_size)
position_embeddings = position_embedding_table(positions)

print("Positional embedding:", position_embeddings)


#combine token + position embedding
x = token_embeddings + position_embeddings
print("Combined embedding shape:", x.shape)
