# -*- coding: utf-8 -*-
"""
@author: adshan

"""
from gensim import corpora
from gensim.matutils import corpus2dense
from gensim.models import LsiModel, TfidfModel
from gensim.summarization import summarize as gensim_summarize
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import RegexpTokenizer, sent_tokenize
import numpy as np
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from . import preprocessing


class Stemmer(object):
    """Class to create a stemmer for stemming single words.
    Class Stemmer is initialized in class TextSummarizer and does not need otherwise be used
    """
    
    def __init__(self):
        
        self._stemmer = EnglishStemmer().stem
    
    def __call__(self, word):
        
        return self._stemmer(word)


class TextSummarizer:
    """Class to create a summary from a single text. After class TextSummarizer is initialized, 
    summaries can be created using the function extractSummary(text).
    """
    
    def __init__(self, stopwords=preprocessing.get_stopwords()):
        """Initializes a text summarizer class
        
        Keyword Arguments
        -----------------
            stopwords {None, list} -- A list of stopwords to be used in text summarization.
                If stopwords=None (recommended), the nltk english stopwords list will be used (default:None)
        """
        
        self.stopwords=stopwords
        self.stemmer = Stemmer()
        self.tokenizer = RegexpTokenizer("[\w']+")
        self.initializeSummarizers()
    
    
    def initializeSummarizers(self):
        
        self.lexRankSummarizer = LexRankSummarizer(self.stemmer)
        self.lexRankSummarizer.stop_words = frozenset(self.stopwords)
        self.sumBasicSummarizer = SumBasicSummarizer(self.stemmer)
        self.sumBasicSummarizer.stop_words = frozenset(self.stopwords)
        self.klSummarizer = KLSummarizer(self.stemmer)
        self.klSummarizer.stop_words = frozenset(self.stopwords)

    
    def calculateInputs(self, total_sentences, ratio=None, num_sentences=None):
        
        if num_sentences:
            if isinstance(num_sentences, int) == False:
                raise TypeError('num_sentences must be of type int or None')
            if num_sentences <= 0:
                raise ValueError('num_sentences must be a positive integer')
    
    
    def calculateNumSentences(self, num_sentences, ratio, total_sentences):
        
        if num_sentences:
            if isinstance(num_sentences, int) == False:
                raise TypeError('num_sentences must be of type int or None')
            if num_sentences < 0:
                raise ValueError('num_sentences must be a positive integer')
            ratio = min(num_sentences/total_sentences, 1.0)
            return num_sentences, ratio
        elif num_sentences == 0:
            raise ValueError('num_sentences must be a positive integer')
        
        if ratio:
            if isinstance(ratio, float) == False:
                raise TypeError('parameter ratio must be of type float')
            if ratio > 0.0 and ratio <= 1.0:
                num_sentences = max(1, int(total_sentences*ratio+0.5))
                return num_sentences, ratio
            elif ratio == 0.0:
                raise ValueError('ratio must be larger than 0.0 and smaller than 1.0')
            else:
                raise ValueError('ratio must be larger than 0.0 and smaller than 1.0')
        
        raise ValueError('num_sentences and ratio cannot both be None')


    def errorChecking(self, text):
        
        if isinstance(text, str) == False:
            raise TypeError('Input text should be of type str')    
    
    
    def tfidfModel(self, text):
        
        sent_list = sent_tokenize(text)
        texts = [self.tokenizer.tokenize(x.lower()) for x in sent_list]
        texts = [[self.stemmer(x) for x in y if x not in self.stopwords] for y in texts]
        
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        tfidf = TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        
        return corpus_tfidf, dictionary, sent_list
 
    
    def summarizeKLSum(self, text, num_sentences=1):
        
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summary = self.klSummarizer(parser.document, num_sentences)
        summary = [str(x) for x in summary]
        
        return summary
       
    
    def summarizeLexRank(self, text, num_sentences=1):
        
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summary = self.lexRankSummarizer(parser.document, num_sentences)
        summary = [str(x) for x in summary]
        
        return summary


    def summarizeLSA(self, text, num_sentences=1, num_dims=20):
        
        corpus_tfidf, dictionary, sent_list = self.tfidfModel(text)
        lsi = LsiModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=num_dims)
        
        u = lsi.projection.u  # Left singular vectors
        s = lsi.projection.s  # Singular values
        v = corpus2dense(lsi[corpus_tfidf], len(s)).T / s  # Right singular vectors
        
        length = np.sqrt(np.dot(np.square(v), np.square(u.T)).sum(axis=1))
        sent_ranks = length.argsort()[-num_sentences:][::-1]
        summary = [sent_list[x] for x in sent_ranks]
        
        return summary
    
    
    def summarizeSumBasic(self, text, num_sentences=1):
        
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summary = self.sumBasicSummarizer(parser.document, num_sentences)
        summary = [str(x) for x in summary]
        
        return summary
    
    
    def summarizeTextRank(self, text, ratio, total_sentences):
        
        summary = gensim_summarize(text, ratio=ratio, split=True)
        if not summary:
            summary = gensim_summarize(text, ratio=1/total_sentences, split=True)
        
        return summary


    def extractSummary(self, text, num_sentences=None, ratio=0.3, method='textrank', return_type='str'):
        """Creates a summary from a single text using extractive summarization. 
        The choice of summarization algorithm can be modified.

        Arguments
        ---------
            input_text {str} -- The text from which a summary will be created.
                Recommended that punctuation is not completely removed from the text, so that sentences can be parsed.
        
        Keyword Arguments
        -----------------
            num_sentences {int, None} -- The number of sentences to return in the summary.
                If None, this argument will be ignored and ratio will be used. If int, ratio will be ignored. 
                It must be a positive integer. If the argument is larger than the number of sentences in the input text, 
                the entire original text will be returned (default: None).
            ratio {float, None} -- The ratio determining the length of the summary. The number of sentences in the summary 
                is the ratio * the number of sentences in the original text. ratio must be a positive float <= 1.0.
                If num_sentences is not None, ratio will be ignored. However, num_sentences and ratio cannot both be None
                (default: 0.3).
            method {str} -- The name of the algorithm used to create a summary.
                Currently supported methods: 'klsum', 'lexrank', 'lsa', 'sumbasic', 'textrank' (default: 'textrank').
            output_type {str} -- Indicator of output format. return_type = 'str' will return a summary as a string. 
                return_type = 'list' will return a list of sentences chosen to be included in the summary, ranked 
                in descending order of importance (default: 'str').
        
        Raises
        ------
            TypeError -- If text, num_sentences, ratio, method, or return_type are not of the appropriate types.
            ValueError -- If num_sentences, ratio, or return_type are not in the possible range of values.
            Exception -- If method called is not a supported method (see above for supported methods).

        Returns
        -------
            str, list -- Variable return type depending on return_type (default: str)
        """     
        
        self.errorChecking(text)
        
        total_sentences = len(sent_tokenize(text))
        if total_sentences < 2:
            return text
        
        num_sentences, ratio = self.calculateNumSentences(num_sentences=num_sentences, ratio=ratio, total_sentences=total_sentences)
        
        if isinstance(method, str):
            method = method.lower()
            if method == 'klsum':
                summary_list = self.summarizeKLSum(text, num_sentences=num_sentences)
            elif method == 'lexrank':
                summary_list = self.summarizeLexRank(text, num_sentences=num_sentences)
            elif method == 'lsa':
                summary_list = self.summarizeLSA(text, num_sentences=num_sentences)
            elif method == 'sumbasic':
                summary_list = self.summarizeSumBasic(text, num_sentences=num_sentences)
            elif method == 'textrank':
                summary_list = self.summarizeTextRank(text, ratio=ratio, total_sentences=total_sentences)
            else:
                raise Exception('Method should be one of the following:\n- klsum\n- lexrank\n- lsa\n- sumbasic\n- textrank')
        else:
            raise TypeError('Method should be of type str')
        
        if isinstance(return_type, str):
            return_type = return_type.lower()
            if return_type == 'str':
                summary = ' '.join(x for x in summary_list)
                return summary
            elif return_type == 'list':
                return summary_list
            else:
                raise ValueError("return_type should be one of the following:\n- 'str'\n- 'list'")
        else:
            raise TypeError('return_type should be of type str')
    
