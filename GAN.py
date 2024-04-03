from torch import nn, tensor, zeros, argmax
from itertools import chain

word_index = {}
batch_size = ...
embed_dim = ...
hidden_dim = ...
enc_drop = ...
dec_drop = ...

tokens = ...
indexes = [tensor([word_index.get(word, -1) + 1 for word in seq]) for seq in tokens]
padded = nn.utils.rnn.pad_sequence(indexes, batch_first=True) # num_seq * max_seq

batch_indexes = None # batch * max_seq

encoder = nn.Sequential(nn.Embedding(len(word_index) + 1, embed_dim), nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=enc_drop)) # add undefined, batch * max_seq * embed_dim > batch * hidden_dim
_ , (batch_hn, batch_cn) = encoder(batch_indexes)

dec_embed = nn.Linear(len(word_index) + 1, embed_dim) # add EOS
dec_lstm = nn.LSTM(embed_dim, hidden_dim, dropout=dec_drop)
dec_softmax = nn.Sequential(nn.Linear(hidden_dim, len(word_index) + 1), nn.Softmax())
gen_batch = []
init_dist = zeros(len(word_index) + 1)
init_dist[0] = 1
for a in range(batch_size):
    prev_dist = init_dist
    hn, cn = batch_hn[a], batch_cn[a]
    gen = []
    while True:
        _ , (hn, cn) = dec_lstm(dec_embed(prev_dist), (hn, cn))
        prev_dist = dec_softmax(hn)
        index = argmax(prev_dist)
        if index == 0: break
        gen.append(index - 1)
    gen_batch.append(gen)

parameters = chain(*map(lambda layer: layer.parameters(), (encoder, dec_embed, dec_lstm, dec_softmax)))

discriminator = ...