import numpy as np
import tokenizer


#Assinging the path to the glove vec embeddings (100-Dimensional)
'''
The link to the glove vec embeddings is given in the README file.
'''

path_to_glove_file =  './glove.6B.100d.txt'

#Initialising the embedding matrix with glove vec embeddings

num_tokens = len(tokenizer.word_index_items) + 2
embedding_dim = 100
hits = 0
misses = 0
embeddings_index = {}

with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))


# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in tokenizer.word_index_items:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

