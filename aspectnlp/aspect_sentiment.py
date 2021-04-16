# Standard libraries
from itertools import cycle
import os

# Third-party Libraries
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import sent_tokenize
from textblob import TextBlob
import dill
import torchtext.data as data
import torch
from fasttext import load_model

# Project code
import mydatasets as mydatasets
# DNN configurations
from aspectnlp.config.config_custom import configuration as tbsa_config
from aspectnlp.config.config_customAS import configuration as absa_config
# word-to-vector functions
from aspectnlp.w2v import *
from aspectnlp.models import CNN_Gate_Aspect_Text
from aspectnlp.utils import load_pretrained_embedding


class AspectSentimentScorer():

    def __init__(self,ftmodel_path='', absa='aspect'):
        import aspectnlp
        self.rmls_nlp_dir = os.path.dirname(aspectnlp.__file__)
        self.ontology = ['negative', 'neutral', 'positive']
        self.sia = SentimentIntensityAnalyzer()
        self.args_absa=None
        ftmodel=None
        if not ftmodel_path:
            ftmodel = load_pretrained_embedding()
        else:
            ftmodel=load_model(ftmodel_path)
        assert(ftmodel is not None)

        if absa=='aspect':
            print("Loading pre-trained aspect embedding...")
            self.args_absa=absa_config
            self.args_absa, _=self._absa_embedding(ftmodel)
            self.absa_predict, self.absa_model=self._absa_model()
        elif absa=='topic':
            self.args_absa=tbsa_config
            self.args_absa, _=self._absa_embedding(ftmodel)
            self.absa_predict, self.absa_model=self._absa_model()

    def _absa_embedding(self,ftmodel):
        text_field_path, as_field_path, sm_field_path = self.args_absa.transfer_domain

        text_field_path=os.path.join( self.rmls_nlp_dir, os.path.normpath( text_field_path))
        as_field_path= os.path.join(self.rmls_nlp_dir,os.path.normpath(as_field_path))
        sm_field_path= os.path.join(self.rmls_nlp_dir,os.path.normpath(sm_field_path))
        print('\nLoading model vocabulary ...')

        with open(text_field_path, "rb") as f:
            self.text_field = dill.load(f)
        with open(as_field_path, "rb") as f:
            self.as_field = dill.load(f)
        with open(sm_field_path, "rb") as f:
            self.sm_field = dill.load(f)
        self.args_absa.embed_num = len(self.text_field.vocab)
        self.args_absa.class_num = len(self.sm_field.vocab) - 1
        self.args_absa.aspect_num = len(self.as_field.vocab)
        if not isinstance(self.args_absa.kernel_sizes, list):
            self.args_absa.kernel_sizes = [int(k) for k in self.args_absa.kernel_sizes.split(',')]

        if self.args_absa.embed_file == 'w2v':
            print("Loading W2V pre-trained embedding...")
            word_vecs = np.load(os.path.join(self.rmls_nlp_dir,"embeddings","w2v_words_compressed.npy"))
            print('# word initialized {}'.format(len(word_vecs)))
        else:
            print("Loading GloVe pre-trained embedding...")
            word_vecs = np.load(os.path.join(self.rmls_nlp_dir,"embeddings","glove_words_compressed.npy"))
            print('# word initialized {}'.format(len(word_vecs)))

        self.args_absa.embedding = torch.from_numpy(np.asarray(word_vecs, dtype=np.float32))
        self.args_absa.embed_dim = self.args_absa.embedding.size(1)

        print('# word initialized {}'.format(len(ftmodel.words)))
        self.args_absa.aspect_embedding = load_aspect_embedding_from_fasttext(self.as_field.vocab.itos, ftmodel)
        self.args_absa.aspect_embed_dim = ftmodel.get_dimension()
        self.args_absa.aspect_embedding = torch.from_numpy(np.asarray(self.args_absa.aspect_embedding, dtype=np.float32))

        return self.args_absa, word_vecs

    def _reorder_sm_field(self, labels, logits):
        sm_lid_swaps = [0, 0, 0]
        sm_sid_swaps=[]
        for senti in self.ontology:
            sm_lid_swaps[self.sm_field.vocab.stoi[senti] - 1] = self.ontology.index(senti)
            sm_sid_swaps.append(self.sm_field.vocab.stoi[senti]-1)
        labels = [sm_lid_swaps[l] for l in labels]
        return labels, logits[:, sm_sid_swaps]

    def _absa_model(self):
        model= CNN_Gate_Aspect_Text(self.args_absa)
        model.load_state_dict( torch.load( \
            os.path.join( self.rmls_nlp_dir, os.path.normpath( self.args_absa.pretrained) ) ))

        def predict(data_iter, model, args):
            model.eval()
            pred = []
            logit = []
            logProbs=[]
            preds=[]
            for batch in data_iter:
                feature, aspect = batch.text, batch.aspect
                with torch.no_grad():
                    feature.t_()
                    if not args.aspect_phrase:
                        aspect.unsqueeze_(0)
                    aspect.t_()
                    # batch first, index align
                if args.cuda:
                    feature, aspect = feature.cuda(), aspect.cuda()

                logit, pooling_input, relu_weights = model(feature, aspect)
                logProb, pred = torch.max(logit, 1)

            if len(pred)>0:
                preds = pred.detach().numpy()
                logProbs = logit.detach().numpy()

            return preds, logProbs

        return predict, model

    def normalizeScores(self, score, min_score=-1, max_score=1):

        return (score - min_score) / (max_score - min_score)

    def lin2compound(self,score,scale=1):
        exp_s=np.exp([s/scale for s in score])
        exp_sum_s=exp_s.sum()
        neg,neu,pos=[s/exp_sum_s for s in exp_s]
        return self.normalizeScores( (1-neu)*(pos-neg))

    def scoreVader(self, input_sentences):
        scores=[0,0,0]
        polarity=0.0
        for sentence in input_sentences:
            ps = self.sia.polarity_scores(sentence)
            polarity+=ps['compound']
            scores= [s+v for s, v in zip(scores, list(ps.values())[:-1])]
        scores=[s/len(input_sentences) for s in scores]
        return self.normalizeScores( polarity / len(input_sentences) ), scores

    def scoreTextBlob(self, input_text):
        tb = TextBlob(input_text)
        sent = tb.sentiment
        polarity = sent[0]
        subjectivity = sent[1]
        polarity = self.normalizeScores(polarity)
        scores = [1-polarity, 0.0,  polarity]
        return polarity,scores

    def scoreSentiment(self, text, classifier='vader'):
        polarity=0.5
        scores=[0.5,0.5,0.5]
        if classifier.lower() == 'vader':
            sentences = sent_tokenize(text)
            polarity,scores = self.scoreVader(sentences)
        elif classifier.lower() == 'pattern':
            polarity,scores = self.scoreTextBlob(text)
        return polarity,scores

    def scoreOverall(self,text, scorer=['pattern', 'vader']):
        scores = [0, 0, 0]
        for classifier in scorer:
            p, senti_score = self.scoreSentiment(text=text, classifier=classifier)
            scores = [s + v for s, v in zip(scores, senti_score)]
        scores = [s / len(scorer) for s in scores]
        return self.ontology[np.argmax(scores)],scores

    def scoreABSA(self,content):
        predict_data = mydatasets.SemEval(self.text_field, self.as_field, self.sm_field, content)

        predict_iter = data.Iterator(predict_data, batch_size=len(predict_data), train=False, sort=False,sort_within_batch=False)
        labels, scores = self.absa_predict(predict_iter, self.absa_model, self.args_absa)
        labels, scores = self._reorder_sm_field(labels,scores)
        return [self.ontology[l] for l in labels],scores

    def generateInputContent(self,sample):
        maxKerSize=max(self.args_absa.kernel_sizes)
        sent = sample['sentence'].split()
        words = cycle(sent)
        if len(sent) < 6:
            sent = ' '.join(sent + [next(words) for i in range(maxKerSize - len(sent))])
        else:
            sent= ' '.join(sent)
        if isinstance(sample['aspect'],list):
            for asp in sample['aspect']:
                input = {"sentence":sent,'aspect':asp,'sentiment':""}
                yield input
        else:
            input = {"sentence":sent,'aspect':sample['aspect'],'sentiment':""}
            yield input