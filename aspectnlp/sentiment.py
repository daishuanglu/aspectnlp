from gensim.models import KeyedVectors
#import keras
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer
import numpy as np
#from pycorenlp import StanfordCoreNLP
from textblob import TextBlob


class SentimentScorer:
    
    def __init__(self, deep=True):
        
        self.classifiers = ['dnn', 'pattern', 'vader']
        self.word_tokenizer = RegexpTokenizer("[\w']+")
        self.__initialize_classifiers__(deep)


    def __initialize_classifiers__(self, deep):
        
        self.sia = SentimentIntensityAnalyzer()
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        print('Initializing functionality')
        if deep:
            self.__read_w2v_dictmatrix__()
            print('Word2Vec embeddings loaded')
            self.__read_pretrained_dnn__()
            print('Pretrained DNN loaded')
        #self.nlp = StanfordCoreNLP('http://localhost:9000')
    

    def __error_checking__(self, text, classifier, return_sentences):

        # Error checking for input str
        if not isinstance(text, str):
            raise TypeError('Input "text" must be of type str')

        # Error checking for return_sentences
        if not isinstance(return_sentences, bool):
            raise TypeError('Argument "return_sentences" must be of type bool')
        
        # Error checking for input classifier
        if isinstance(classifier, str):
            classifier = classifier.lower()
            if classifier not in self.classifiers:
                raise ValueError('Input classifier must be one of the following: "dnn", "pattern", "vader"')
        elif isinstance(classifier, list):
            if len(classifier) == 0:
                raise ValueError('Input classifier cannot be an empty list')
            for i in range(len(classifier)):
                if isinstance(classifier[i], str):
                    classifier[i] = classifier[i].lower()
                    if classifier[i] not in self.classifiers:
                        raise ValueError('Input classifiers must be one of the following: "pattern", "vader"')
                else:
                    raise TypeError('All entries in input list classifier must be of type str')
            if len(set(classifier)) != len(classifier):
                classifier = list(set(classifier))
            if len(classifier) == 1:
                classifier = classifier[0]
        else:
            raise TypeError('Input classifier must be of type str or list')
        
        return text, classifier


    def __normalize_scores__(self, score, min_score=-1, max_score=1):
        
        return (score - min_score) / (max_score - min_score)


    def __read_pretrained_dnn__(self, path_model='data\\ST_w2v300_03_best.h5'):
        
        self.model_dnn = keras.models.load_model(path_model)
        self.max_len = 56


    # Function for reading Word2Vec embedding file and converting it to dict and matrix
    def __read_w2v_dictmatrix__(self, embedding_path='data\\GoogleNews-vectors-negative300.bin'):
        
        model_w2v = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
        vocab = list(model_w2v.vocab.keys())
        
        #embedding_matrix = []
        embedding_idxs = {}
        
        for i in range(len(vocab)):
            embedding_idxs[vocab[i]] = i
            #embedding_matrix.append(model_w2v[vocab[i]])
        
        # Random embedding for unknown words
        #embedding_matrix.append(np.random.uniform(-0.05, 0.05, embedding_matrix[0].shape).astype(np.float64))
    
        # Convert list to np.array
        #embedding_matrix = np.stack(embedding_matrix)
        
        #return embedding_idxs, embedding_matrix
        self.embedding_idxs = embedding_idxs
        self.embedding_idxs_lower = {k.lower():v for k,v in embedding_idxs.items()}


    def __prepare_data_dnn__(self, input_str):
        
        str_list = self.word_tokenizer.tokenize(input_str)
        if len(str_list) > self.max_len:
            str_list = str_list[0:self.max_len]
        input_idxs = np.zeros((1, self.max_len))
        for j in range(len(str_list)):
            word = str_list[j]
            if word in self.embedding_idxs.keys():
                input_idxs[0][j] = self.embedding_idxs[word]
            else:
                word = word.lower()
                if word in self.embedding_idxs_lower.keys():
                    input_idxs[0][j] = self.embedding_idxs_lower[word]
        
        input_idxs = input_idxs.astype(int)

        return input_idxs


    def __calculate_dnn__(self, sent_list):
        
        sent_results = []
        overall = 0.0
        for sentence in sent_list:
            input_idxs = self.__prepare_data_dnn__(sentence)
            pred = self.model_dnn.predict(input_idxs)
            sent_score = np.average(pred)
            overall += sent_score
            sent_results.append((sentence, sent_score))
        
        overall /= len(sent_list)

        return overall, sent_results



    def __calculate_textblob__(self, input_text, score_type='polarity'):
        
        sent_results = []
        tb = TextBlob(input_text, tokenizer=self.sent_tokenizer)

        # Overall scores
        overall_tuple = tb.sentiment

        # Polarity
        if score_type == 'polarity':
            overall = overall_tuple[0]
            overall = self.__normalize_scores__(score=overall, min_score=-1, max_score=1)
            for sent in tb.sentences:
                sent_score = sent.sentiment[0]
                sent_score = self.__normalize_scores__(score=sent_score, min_score=-1, max_score=1)
                sent_results.append((str(sent), sent_score))
        elif score_type == 'subjectivity':
            overall = overall_tuple[1]
            for sent in tb.sentences:
                sent_score = sent.sentiment[1]
                sent_results.append((str(sent), sent_score))    
        
        return overall, sent_results


    def __calculate_vader__(self, sent_list, score_type='sentiment'):
        
        sent_results = []

        if score_type == 'sentiment':
            overall = 0.0
            for sentence in sent_list:
                ps = self.sia.polarity_scores(sentence)
                sent_score = ps['compound']
                overall += sent_score
                sent_score_norm = self.__normalize_scores__(score=sent_score, min_score=-1, max_score=1)
                sent_results.append((sentence, sent_score_norm))
            overall /= len(sent_list)
            overall = self.__normalize_scores__(score=overall, min_score=-1, max_score=1)
            return overall, sent_results
        elif score_type == 'magnitude':
            overall = {}
            overall['score'] = 0.0
            overall['negative'] = 0.0
            overall['neutral'] = 0.0
            overall['positive'] = 0.0
            for sentence in sent_list:
                sent_dict = {}
                ps = self.sia.polarity_scores(sentence)
                sent_score = ps['compound']
                neg_score = ps['neg']
                neut_score = ps['neu']
                pos_score = ps['pos']
                sent_score_norm = self.__normalize_scores__(score=sent_score, min_score=-1, max_score=1)
                overall['score'] += sent_score
                overall['negative'] += neg_score
                overall['neutral'] += neut_score
                overall['positive'] += pos_score
                sent_dict['overall'] = sent_score_norm
                sent_dict['negative'] = neg_score
                sent_dict['neutral'] = neut_score
                sent_dict['positive'] = pos_score
                sent_results.append((sentence, sent_dict))
            overall['score'] /= len(sent_list)
            overall['negative'] /= len(sent_list)
            overall['neutral'] /= len(sent_list)
            overall['positive'] /= len(sent_list)
            overall['score'] = self.__normalize_scores__(score=overall['score'], min_score=-1, max_score=1)
        
        return overall, sent_results


    def __calculate_score_algorithm__(self, text, algorithm='vader'):
        
        sent_list = self.sent_tokenizer.tokenize(text)
        sent_list = [x for x in sent_list if len(self.word_tokenizer.tokenize(x)) > 0]
        
        if algorithm == 'vader':
            sent_list = self.sent_tokenizer.tokenize(text)
            overall, sent_results = self.__calculate_vader__(sent_list, score_type='sentiment')
        elif algorithm == 'pattern':
            text = ' '.join(x for x in sent_list)
            overall, sent_results = self.__calculate_textblob__(text)
        elif algorithm == 'dnn':
            overall, sent_results = self.__calculate_dnn__(sent_list)
        
        return overall, sent_results

    def score_magnitude(self, text, return_sentences=False):

        # Error checking for input str
        if not isinstance(text, str):
            raise TypeError('Input "text" must be of type str')
        
        sent_list = self.sent_tokenizer.tokenize(text)
        overall, sent_results = self.__calculate_vader__(sent_list, score_type='magnitude')
        if not isinstance(return_sentences, bool):
            raise TypeError('Attribute "return_sentences" must be of type bool')
        if return_sentences:
            return overall, sent_results
        else:
            return overall

    def score_subjectivity(self, text, return_sentences=False):

        # Error checking for input str
        if not isinstance(text, str):
            raise TypeError('Input "text" must be of type str')

        overall, sent_results = self.__calculate_textblob__(text, score_type='subjectivity')
        if not isinstance(return_sentences, bool):
            raise TypeError('Attribute "return_sentences" must be of type bool')
        if return_sentences:
            return overall, sent_results
        else:
            return overall


    def score_sentiment(self, text, classifier=['pattern', 'vader'], return_sentences=False):
        
        # Error checking and edge cases
        text, classifier = self.__error_checking__(text=text, classifier=classifier, return_sentences=return_sentences)
        if len(text) == 0:
            return 0.5
        
        # CHECK FOR LEN==0 BUT STILL NOT VALID

        # Calculate sentiment score - single algorithm or simple average for multiple algorithms
        if isinstance(classifier, str):
            overall, sent_results = self.__calculate_score_algorithm__(text=text, algorithm=classifier)
        else:
            count = 0
            for algorithm in classifier:
                if count == 0:
                    overall, sent_results = self.__calculate_score_algorithm__(text=text, algorithm=algorithm)
                    sent_results = [list(x) for x in sent_results]
                else:
                    algo_overall, algo_sent_results = self.__calculate_score_algorithm__(text=text, algorithm=algorithm)
                    overall += algo_overall
                    for i in range(len(sent_results)):
                        sent_results[i][1] += algo_sent_results[i][1]
                count += 1
            overall /= count
            sent_results = [tuple([x, y/count]) for [x, y] in sent_results]
        
        if return_sentences:
            return overall, sent_results
        else:
            return overall