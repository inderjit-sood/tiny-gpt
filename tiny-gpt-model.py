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


#Add attention to the combined token and position embedding

#Random weighted matrix for K V Q (The meaning of K V Q comes during training, right now it is random)
key = nn.Linear(embedding_dimensions, embedding_dimensions, bias=False)

value = nn.Linear(embedding_dimensions, embedding_dimensions, bias=False)

query = nn.Linear(embedding_dimensions, embedding_dimensions, bias=False)

#Transformations onto the combined token + position embeddings

K = key(x)
V = value(x)
Q = query(x)

print("K = ",K)
print("Q = ",Q)
print("V = ",V)


#Calculate attention scores (semantic or contextual relation) - How much attention each token pays to other tokens in the sequence 
scores = Q @ K.transpose(-2, -1)

print("Raw attention score = ", scores)

#Scale down the scores as the scores might be extra large. Scale down by root of embedding dimensions

scores = scores / (embedding_dimensions ** 0.5)

print("Scaled down score = ", scores)


#Normalize the attention scores as probabilities so that for each token, the attention needed for other tokens adds to one and gives better view of attention

weights = torch.softmax(scores, dim = -1)

print("Normalized softmaxed score = ", weights)
