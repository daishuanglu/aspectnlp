# Standard libraries
import re
import json, io
import os
import time
import warnings

# Third-party libraries
import torch
import numpy as np
from nltk import word_tokenize
from keras.utils import to_categorical
"""fastText: https://github.com/facebookresearch/fastText"""
from fasttext import load_model
from nltk import pos_tag
import nltk
nltk.download('stopwords')
from rake_nltk import Rake
import urllib.request
warnings.filterwarnings("ignore")



def keep_alpha(s):
    return re.sub(r'[^A-Za-z /]+', '', s)

class Model(torch.nn.Module):
    def __init__(self, gen_emb, domain_emb, num_classes=3, dropout=0.5, crf=False, tag=False):
        super(Model, self).__init__()
        self.tag_dim = 45 if tag else 0

        self.gen_embedding = torch.nn.Embedding(gen_emb.shape[0], gen_emb.shape[1])
        self.gen_embedding.weight=torch.nn.Parameter(torch.from_numpy(gen_emb), requires_grad=False)
        self.domain_embedding = torch.nn.Embedding(domain_emb.shape[0], domain_emb.shape[1])
        self.domain_embedding.weight=torch.nn.Parameter(torch.from_numpy(domain_emb), requires_grad=False)

        self.conv1=torch.nn.Conv1d(gen_emb.shape[1]+domain_emb.shape[1], 128, 5, padding=2 )
        self.conv2=torch.nn.Conv1d(gen_emb.shape[1]+domain_emb.shape[1], 128, 3, padding=1 )
        self.dropout=torch.nn.Dropout(dropout)

        self.conv3=torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv4=torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv5=torch.nn.Conv1d(256, 256, 5, padding=2)
        self.linear_ae1=torch.nn.Linear(256+self.tag_dim+domain_emb.shape[1], 50)
        self.linear_ae2=torch.nn.Linear(50, num_classes)
        self.crf_flag=crf
        if self.crf_flag:
            from allennlp.modules import ConditionalRandomField
            self.crf=ConditionalRandomField(num_classes)

    def forward(self, x, x_len, x_mask, x_tag, y=None, testing=False):
        x_emb=torch.cat((self.gen_embedding(x), self.domain_embedding(x) ), dim=2)  # shape = [batch_size (128), sentence length (83), embedding output size (300+100)]
        x_emb=self.dropout(x_emb).transpose(1, 2)  # shape = [batch_size (128), embedding output size (300+100+tag_num) , sentence length (83)]
        x_conv=torch.nn.functional.relu(torch.cat((self.conv1(x_emb), self.conv2(x_emb)), dim=1) )  # shape = [batch_size, 128+128, 83]
        x_conv=self.dropout(x_conv)
        x_conv=torch.nn.functional.relu(self.conv3(x_conv) )
        x_conv=self.dropout(x_conv)
        x_conv=torch.nn.functional.relu(self.conv4(x_conv) )
        x_conv=self.dropout(x_conv)
        x_conv=torch.nn.functional.relu(self.conv5(x_conv) )
        x_conv=x_conv.transpose(1, 2) # shape = [batch_size, 83, 256]
        x_logit=torch.nn.functional.relu(self.linear_ae1(torch.cat((x_conv, x_tag, self.domain_embedding(x)), dim=2) ) ) # shape = [batch_size, 83, 20]
        x_logit=self.linear_ae2(x_logit)
        if testing:
            if self.crf_flag:
                score=self.crf.viterbi_tags(x_logit, x_mask)
            else:
                x_logit=x_logit.transpose(2, 0)
                score=torch.nn.functional.log_softmax(x_logit,dim=1).transpose(2, 0)
        else:
            if self.crf_flag:
                score=-self.crf(x_logit, y, x_mask)
            else:
                x_logit=torch.nn.utils.rnn.pack_padded_sequence(x_logit, x_len, batch_first=True)
                score=torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x_logit.data,dim=1), y.data)
        return score

# This is for color the output text
class bcolors:
    ONGREEN = '\x1b[6;30;42m'
    ONYELLOW = '\x1b[6;30;46m'
    ONRED = '\x1b[6;30;41m'
    ONPURPLE = '\x1b[6;30;45m'
    END = '\x1b[0m'


def find_phrase(word,keyphrases):
    return [kp for kp in keyphrases if word.lower() in kp.lower()]


class aspectDetector():
    def __init__(self,ftmodel_path=None,sent_len=200):
        self.max_sent_len = sent_len
        import aspectnlp
        rmls_nlp_dir = os.path.dirname(aspectnlp.__file__)
        self.domain = "custom"
        self.domain_emb = self.domain+"_emb.vec"
        self.emb_dir = os.path.join(rmls_nlp_dir, "absa_embedding")
        self.prep_dir = os.path.join(rmls_nlp_dir, "absa_prep")
        self.model_fn = os.path.join(rmls_nlp_dir, "absa_model", self.domain)
        self.gen_emb="gen.vec"
        self.gen_dim=300
        self.domain_dim=100

        # port model parameters
        self.crf=False
        self.tag=True
        self.dropout=0.25

        self.prev_word={}
        self.embedding_gen_dict = {}
        self.embedding_domain_dict = {}
        self.embedding_gen=None
        self.embedding_domain=None
        self.ftmodel=None
        self.model=None
        self._load_embeddings_and_model(ftmodel_path)

    def _load_embeddings_and_model(self,ftmodel_path=None):
        with io.open(os.path.join(self.prep_dir, 'word_idx.json')) as f:
            self.prev_word = json.load(f)
        if ftmodel_path is None:
            ftmodel_path=os.path.join(self.emb_dir, self.domain_emb + ".bin")
            try:
                self.ftmodel=load_model(ftmodel_path)
            except:
                print("Embedding not found! please add your fasttext embedding OR \n  Download our custom model from https://drive.google.com/u/0/uc?export=download&confirm=HYUN&id=1mQPKHoa4SQr-skCO5XpzWpOxGB5z02-U")
        else:
            self.ftmodel = load_model(ftmodel_path)
        self.embedding_gen = np.load(os.path.join(self.prep_dir, "gen.vec.npy"))
        self.embedding_domain = np.load(os.path.join(self.prep_dir, self.domain+'_emb.vec.npy'))

        self.model = Model(self.embedding_gen, self.embedding_domain, 3, dropout=self.dropout, crf=self.crf, tag=self.tag)
        self.model.load_state_dict(torch.load(self.model_fn, map_location=lambda storage, loc: storage))
        self.model.eval()
        return

    def _prep_emb(self, sentences):
        text = []
        for line in sentences:
            token = word_tokenize(line)
            text = text + token
        vocab = sorted(set(text))
        word_idx = {}
        wx = 0
        new_word = []
        for word in vocab:
            if word not in self.prev_word:
                wx = wx + 1
                new_word.append(word)
                word_idx[word] = wx + len(self.prev_word)
        self.prev_word.update(word_idx)
        if new_word == []:
            return

        self.embedding_gen = np.vstack((self.embedding_gen, np.zeros((len(new_word) + 3, self.gen_dim))))
        self.embedding_domain = np.vstack((self.embedding_domain, np.zeros((len(new_word) + 3, self.domain_dim))))
        for w in new_word:
            if self.embedding_domain[word_idx[w]].sum() == 0.:
                self.embedding_domain[word_idx[w]] = self.ftmodel.get_word_vector(w)
        return

    def detect(self, text, batch_size=128, extract_phrase=True, disp=False, progress=False):
        self._prep_emb(text)
        raw_X, X, X_tag = self.prep_text(text)

        self.model.gen_embedding = torch.nn.Embedding(self.embedding_gen.shape[0], self.embedding_gen.shape[1])
        self.model.gen_embedding.weight = torch.nn.Parameter(
            torch.from_numpy(self.embedding_gen).type(torch.FloatTensor), requires_grad=False)
        self.model.domain_embedding = torch.nn.Embedding(self.embedding_domain.shape[0], self.embedding_domain.shape[1])
        self.model.domain_embedding.weight = torch.nn.Parameter(
            torch.from_numpy(self.embedding_domain).type(torch.FloatTensor), requires_grad=False)
        if not self.tag:
            self.model.tag_dim = 0

        pred_y = self.test(X, X_tag, batch_size, progress)
        sent_asp = self.output_text(text, pred_y, extract_phrase, disp)
        return sent_asp

    def prep_text(self, text):
        # map part-of-speech tag to int
        pos_tag_list = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS',
                        'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG',
                        'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', ',', '.', ':', '$', '#', "``", "''", '(', ')']
        tag_to_num = {tag: i + 1 for i, tag in enumerate(sorted(pos_tag_list))}
        sentence_size = [-1, 130]
        count = 0
        for l in text:
            token = word_tokenize(l)[:self.max_sent_len]
            if len(token) > 0:
                count = count + 1
                if len(token) > sentence_size[1]:
                    sentence_size[1] = len(token)
        sentence_size[0] = count
        X = np.zeros((sentence_size[0], sentence_size[1]), np.int16)
        X_tag = np.zeros((sentence_size[0], sentence_size[1]), np.int16)

        count = -1
        raw_X = []
        for l in text:
            token = word_tokenize(l)[:self.max_sent_len]
            pos_tag_stf = [tag_to_num[tag] for (_, tag) in pos_tag(token)]
            if len(token) > 0:
                count = count + 1
                raw_X.append(token)
                # write word index and tag in train_X and train_X_tag
                for wx, word in enumerate(token):
                    X[count, wx] = self.prev_word[word]
                    X_tag[count, wx] = pos_tag_stf[wx]
        return raw_X, X, X_tag

    def test(self, test_X, test_X_tag, batch_size=128, progress=False):
        start_time = time.time()
        pred_y = np.zeros((test_X.shape[0], test_X.shape[1]), np.int16)
        for offset in range(0, test_X.shape[0], batch_size):
            batch_test_X_len = np.sum(test_X[offset:offset + batch_size] != 0, axis=1)
            batch_idx = batch_test_X_len.argsort()[::-1]
            batch_test_X_len = batch_test_X_len[batch_idx]
            batch_test_X_mask = (test_X[offset:offset + batch_size] != 0)[batch_idx].astype(np.uint8)
            batch_test_X = test_X[offset:offset + batch_size][batch_idx]
            batch_test_X_mask = torch.autograd.Variable(torch.from_numpy(batch_test_X_mask).long())
            batch_test_X = torch.autograd.Variable(torch.from_numpy(batch_test_X).long())
            if self.tag:
                batch_test_X_tag = test_X_tag[offset:offset + batch_size][batch_idx]
                batch_test_X_tag_onehot = to_categorical(batch_test_X_tag, num_classes=45 + 1)[:, :, 1:]
                batch_test_X_tag_onehot = torch.autograd.Variable(
                    torch.from_numpy(batch_test_X_tag_onehot).type(torch.FloatTensor))
            else:
                batch_test_X_tag_onehot = None
            batch_pred_y = self.model(batch_test_X, batch_test_X_len, batch_test_X_mask, batch_test_X_tag_onehot,
                                      testing=True)
            r_idx = batch_idx.argsort()
            if self.crf:
                batch_pred_y = [batch_pred_y[idx] for idx in r_idx]
                for ix in range(len(batch_pred_y)):
                    for jx in range(len(batch_pred_y[ix])):
                        pred_y[offset + ix, jx] = batch_pred_y[ix][jx]
            else:
                batch_pred_y = batch_pred_y.data.cpu().numpy().argmax(axis=2)[r_idx]
                pred_y[offset:offset + batch_size, :batch_pred_y.shape[1]] = batch_pred_y
            if progress:
                print('\rBatch[{}/{}], {} secs'.format(offset, test_X.shape[0], int(time.time() - start_time)))
        assert len(pred_y) == len(test_X)
        return pred_y

    def output_text(self, text, pred_y, extract_phrase=True, display=False):
        count = -1
        pos_tag_list = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS',
                        'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG',
                        'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
        useless_tags = ['CC', 'DT', 'PRP', 'WRB', 'MD', 'PRP$', 'WP', 'WP$', 'VBP', 'EX']
        useless_words = ['is', 'are', 'am', 'be', 'being', 'been', 's', 're', 'll']
        result_sent_asp = []
        kwd_detector = Rake()
        for l in text:
            token = word_tokenize(l)[:self.max_sent_len]
            sent_asp = {'sentence': ' '.join(token), 'aspect': []}
            if extract_phrase:
                kwd_detector.extract_keywords_from_text(l)
            if len(token) > 0:
                tag = [t for _, t in pos_tag(token)]
                count = count + 1
                for wx, word in enumerate(token):
                    kps = []
                    word = keep_alpha(word).rstrip()
                    if len(word) == 0 or (word in useless_words):
                        continue
                    if tag[wx] not in pos_tag_list:
                        # print(word, end=" ")
                        continue
                    if tag[wx] in ['DT', 'JJ', 'RB', 'TO', 'IN'] and wx + 1 < len(token):
                        pred_y[count, wx] = 0
                        pred_y[count, wx + 1] = 1
                        if wx + 2 < len(token):
                            if tag[wx + 1] in ['JJ', 'RB']:
                                pred_y[count, wx + 2] = 1
                    if pred_y[count, wx] == 1 and (tag[wx] not in useless_tags):
                        if extract_phrase:
                            kps = find_phrase(word, kwd_detector.get_ranked_phrases())
                        if len(kps) > 0:
                            sent_asp['aspect'] += kps
                        else:
                            sent_asp['aspect'].append(word)
                        if display: print(bcolors.ONGREEN + word + bcolors.END, end=" ")
                    elif pred_y[count, wx] == 2 and (tag[wx] not in useless_tags):
                        if extract_phrase:
                            kps = find_phrase(word, kwd_detector.get_ranked_phrases())
                        if len(kps) > 0:
                            sent_asp['aspect'] += kps
                        else:
                            sent_asp['aspect'].append(word)
                        if display: print(bcolors.ONGREEN + word + bcolors.END, end=" ")
                    else:
                        if display: print(word, end=" ")
                if display: print('\n')
                sent_asp['aspect'] = list(set(sent_asp['aspect']))
            
            result_sent_asp.append(sent_asp)
        return result_sent_asp