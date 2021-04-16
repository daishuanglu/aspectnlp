# third-party libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# project codes
from aspectnlp.w2v import *

def setEmbeddings(args,text_field,as_field):
    if args.embed_file == 'w2v':
        print("Loading W2V pre-trained embedding...")
        word_vecs = load_w2v_embedding(text_field.vocab.itos, args.unif, 300)
    elif args.embed_file == 'glove':
        print("Loading GloVe pre-trained embedding...")
        word_vecs = load_glove_embedding(text_field.vocab.itos, args.unif, 300)
    else:
        raise(ValueError('Error embedding file'))
    print('# word initialized {}'.format(len(word_vecs)))

    print("Loading pre-trained aspect embedding...")
    if args.aspect_file == '':
        args.aspect_embedding = load_aspect_embedding_from_w2v(as_field.vocab.itos, text_field.vocab.stoi, word_vecs)
        args.aspect_embed_dim = args.embed_dim
    else:
        args.aspect_embedding, args.aspect_embed_dim = load_aspect_embedding_from_file(as_field.vocab.itos, args.aspect_file)

    args.embedding = torch.from_numpy(np.asarray(word_vecs, dtype=np.float32))
    args.aspect_embedding = torch.from_numpy(np.asarray(args.aspect_embedding, dtype=np.float32))

    return args,word_vecs

class CNN_Gate_Aspect_Text(nn.Module):
    def __init__(self, args):
        super(CNN_Gate_Aspect_Text, self).__init__()
        self.args = args

        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        A = args.aspect_num

        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)
        self.embed.weight = nn.Parameter(args.embedding, requires_grad=True)

        self.aspect_embed = nn.Embedding(A, args.aspect_embed_dim)
        self.aspect_embed.weight = nn.Parameter(args.aspect_embedding, requires_grad=True)

        self.convs1 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])

        self.fc1 = nn.Linear(len(Ks) * Co, C)
        self.fc_aspect = nn.Linear(args.aspect_embed_dim, Co)

    def forward(self, feature, aspect):
        feature = self.embed(feature)  # (N, L, D)
        aspect_v = self.aspect_embed(aspect)  # (N, L', D)
        aspect_v = aspect_v.sum(1) / aspect_v.size(1)

        x = [F.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        y = [F.relu(conv(feature.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
        x = [i * j for i, j in zip(x, y)]

        # pooling method
        x0 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        x0 = [i.view(i.size(0), -1) for i in x0]

        x0 = torch.cat(x0, 1)
        logit = self.fc1(x0)  # (N,C)
        return logit, x, y


class CNN_Gate_Aspect_Text_atsa(nn.Module):
    def __init__(self, args):
        super(CNN_Gate_Aspect_Text_atsa, self).__init__()
        self.args = args

        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        A = args.aspect_num

        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)
        self.embed.weight = nn.Parameter(args.embedding, requires_grad=True)

        self.aspect_embed = nn.Embedding(A, args.aspect_embed_dim)
        self.aspect_embed.weight = nn.Parameter(args.aspect_embedding, requires_grad=True)

        self.convs1 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs3 = nn.ModuleList([nn.Conv1d(D, Co, K, padding=K - 2) for K in [3]])
        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(len(Ks) * Co, C)
        self.fc_aspect = nn.Linear(100, Co)

    def forward(self, feature, aspect):
        feature = self.embed(feature)  # (N, L, D)
        aspect_v = self.aspect_embed(aspect)  # (N, L', D)
        aa = [F.relu(conv(aspect_v.transpose(1, 2))) for conv in self.convs3]  # [(N,Co,L), ...]*len(Ks)
        aa = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aa]
        aspect_v = torch.cat(aa, 1)
        x = [F.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        y = [F.relu(conv(feature.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
        x = [i * j for i, j in zip(x, y)]

        # pooling method
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N,len(Ks)*Co)
        logit = self.fc1(x)  # (N,C)
        return logit, x, y