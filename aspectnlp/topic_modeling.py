# -*- coding: utf-8 -*-
"""

"""

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel, Phrases
import lda
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import RegexpTokenizer
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class TopicModeler:
    
    def __init__(self, stopwords='nltk', phrases=True, remove_numbers=True, verbose=True, invalid_list=['na', 'nan', 'none']):

        self.invalid_list = invalid_list
        self.models_available = {'lda':'Latent Dirichlet Allocation',
                                 'lda_lda': 'Latent Dirichlet Allocation (implemented with LDA python package)', 
                                 'lsa':'Latent Semantic Analysis'}
        self.random_state = 17
        self.phrases = phrases
        self.remove_numbers = remove_numbers
        self.stopwords = stopwords
        self.tokenizer = RegexpTokenizer("[\w']+")
        self.verbose = verbose
        self.__error_checking_initialization__()


    def __error_checking__(self, df, target_column, model, num_topics, tfidf, min_k, max_k, step_size):

        if isinstance(df, pd.DataFrame) == False:
            raise TypeError('Input df must be of type pd.DataFrame')
        if isinstance(target_column, str) == False:
            raise TypeError('Input target_column must be of type str')
        if target_column not in list(df.columns):
            raise ValueError('Input target_column must be a column in input df')
        if isinstance(model, str) == False:
            raise TypeError('Input model must be of type str')
        else:
            if model.lower() not in self.models_available.keys():
                raise ValueError('Input model must be one of "lda, lsa, lda_lda"')
        if isinstance(tfidf, bool) == False:
            raise TypeError('Input tf_idf must be of type bool (True or False)')
        if isinstance(num_topics, bool):
            raise TypeError('Input "num_topics" must be of type int or str')
        elif isinstance(num_topics, int):
            if num_topics <= 0:
                raise ValueError('Input "num_topics" must be a positive integer')
        elif isinstance(num_topics, str):
            if num_topics.lower() not in ['auto']:
                raise ValueError('Input "num_topics" must be "auto", or of type int')
            if isinstance(min_k, bool):
                raise TypeError('Input "min_k" must be of type int')
            if isinstance(min_k, int) == False:
                raise TypeError('Input "min_k" must be of type int')
            if isinstance(max_k, bool):
                raise TypeError('Input "max_k" must be of type int')            
            if isinstance(max_k, int) == False:
                raise TypeError('Input "max_k" must be of type int')
            if isinstance(step_size, bool):
                raise TypeError('Input "step_size" must be of type int')
            if isinstance(step_size, int) == False:
                raise TypeError('Input "step_size" must be of type int')
            if min_k < 1:
                raise ValueError('Input "min_k" must be a positive integer')
            if max_k < 1:
                raise ValueError('Input "max_k" must be a positive integer')
            if step_size < 1:
                raise ValueError('Input "step_size" must be a positive integer')
            if min_k > max_k:
                raise ValueError('Input "min_k" cannot be greater than "max_k"')
            if min_k == max_k:
                num_topics = min_k
            if min_k + step_size > max_k:
                num_topics = min_k       
        else:
            raise TypeError('Input "num_topics" must be of type int or str')

        return num_topics


    def __error_checking_initialization__(self):

        if self.stopwords:
            if isinstance(self.stopwords, str):
                if self.stopwords.lower() == 'nltk':
                    self.stopwords = set(nltk_stopwords.words('english'))
                else:
                    raise ValueError('Input stopwords must be of type list, set, or str. If str, must be "nltk"')
            elif isinstance(self.stopwords, list):
                self.stopwords = set(self.stopwords)
                if not all(isinstance(x, str) for x in self.stopwords):
                    raise TypeError('All elements in input stopwords must be of type str')
            elif isinstance(self.stopwords, set):
                if not all(isinstance(x, str) for x in self.stopwords):
                    raise TypeError('All elements in input stopwords must be of type str')
            else:
                raise TypeError('Input stopwords must be of type list, set, or str. If str, must be "nltk"')
            self.stopwords = list(self.stopwords)
        if isinstance(self.phrases, bool) == False:
            raise TypeError('Input "phrases" must be of type bool (True or False)')
        if isinstance(self.remove_numbers, bool) == False:
            raise TypeError('Input "remove_numbers" must be of type bool (True or False)')
        if isinstance(self.verbose, bool) == False:
            raise TypeError('Input "verbose" must be of type bool (True or False)')


    def __assign_top_topic__(self, input_df):

        top_topics = self.get_top_topics(n_top_topics=1, min_weight=None, return_type='df')
        top_topics = top_topics.rename(columns={'Topic':'Top_Topic', 'Weight':'Top_Weight'})
        input_df = pd.merge(input_df, top_topics, left_on='RowId', right_on='RowId')
        
        return input_df


    def __generate_phrases__(self, corpus, min_count=10):

        phrase_model = Phrases(corpus, min_count=min_count)
        if self.stopwords:
            for i in range(len(corpus)):
                for token in phrase_model[corpus[i]]:
                    if '_' in token and token not in corpus[i]:
                        token_list = token.split('_')
                        if len([x for x in token_list if x in self.stopwords]) == 0:
                            corpus[i].append(token)
        else:
            for i in range(len(corpus)):
                for token in phrase_model[corpus[i]]:
                    if '_' in token and token not in corpus[i]:
                        corpus[i].append(token)

        return corpus


    def __is_valid_screen__(self, text_list, min_chars=None, min_words=None, invalid_corpus=[]):

        if len(text_list) == 0:
            return False
        if min_words and min_words > 0:
            if len(text_list) < min_words:
                return False
        if min_chars and min_chars > 0:
            if len(text_list) < min_chars:
                text = ' '.join(x for x in text_list)
                if len(text) < min_chars:
                    return False
        if invalid_corpus:
            if text_list in invalid_corpus:
                return False
        
        return True


    def __prepare_corpus_gensim__(self, corpus, tfidf=False, min_df=5, max_df=0.5):
        
        # Create dictionary
        vocab_dict = Dictionary(corpus)
        
        # Filter invalid words
        vocab_dict.filter_extremes(no_below=min_df, no_above=max_df)
        
        if tfidf:
            print('FIX')
        else:
            corpus_dict = [vocab_dict.doc2bow(x) for x in corpus]
        
        return corpus_dict, vocab_dict


    def __train_model_lda_gensim__(self, corpus, corpus_dict, vocab_dict, num_topics=20, passes=20, num_iter=1000):
        
        temp = vocab_dict[0]
        id2word = vocab_dict.id2token
        lda_model = LdaModel(corpus=corpus_dict, 
                            id2word=id2word, 
                            num_topics=num_topics,
                            passes=passes,
                            iterations=num_iter, 
                            eval_every=1,
                            alpha='auto', 
                            eta='auto')
        
        coherence_model_lda = CoherenceModel(model=lda_model, texts=corpus, dictionary=vocab_dict, coherence='c_v')
        coherence = coherence_model_lda.get_coherence()

        return lda_model, coherence

    def __train_lda__(self, corpus, num_topics=20, min_df=5, max_df=0.5, min_k=5, max_k=40, step_size=5, tfidf=False):
        
        corpus_dict, vocab_dict = self.__prepare_corpus_gensim__(corpus=corpus, tfidf=tfidf, min_df=min_df, max_df=max_df)
        
        if isinstance(num_topics, int):
            lda_model, coherence = self.__train_model_lda_gensim__(corpus=corpus, corpus_dict=corpus_dict, vocab_dict=vocab_dict, num_topics=num_topics, passes=20, num_iter=1000)
            self.model_info['num_topics'] = num_topics
            self.model_info['coherence'] = coherence
            self.trained_model = lda_model
        elif num_topics.lower() == 'auto':
            if self.verbose:
                print('\tSearching for optimal number of topics...')           
            num_topics_list = list(range(min_k, max_k+1, step_size))
            model_list = []
            score_list = []
            for k in num_topics_list:
                lda_model, coherence = self.__train_model_lda_gensim__(corpus=corpus, corpus_dict=corpus_dict, vocab_dict=vocab_dict, num_topics=k, passes=20, num_iter=1000)
                model_list.append(lda_model)
                score_list.append(coherence)
                if self.verbose:
                    print('\t\tTrained LDA model with %s topics' %k)
            best_k = score_list.index(max(score_list))
            self.model_info['num_topics'] = num_topics_list[best_k]
            self.model_info['coherence'] = score_list[best_k]
            self.trained_model = model_list[best_k]
            if self.verbose:
                print('\tOptimal number of topics chosen\n')
        
        self.trained_results = list(self.trained_model[corpus_dict])      
        self.model_info['log_perplexity'] = self.trained_model.log_perplexity(corpus_dict)


    def __train_lsa__(self, df, tfidf=False, n_topics=20, min_df=5, n_iter=100):

        lsa_df = df.copy().reset_index(drop=True)

        # Fit Vectorizer
        if tfidf:
            countvec = TfidfVectorizer(min_df=min_df)
        else:
            countvec = CountVectorizer(min_df=min_df)        
        lsa_train = countvec.fit_transform(lsa_df[self.target_column])
        self.vocab = countvec.get_feature_names()
        
        # Fit LSA
        lsa_model = TruncatedSVD(n_components=n_topics, algorithm='randomized', n_iter=n_iter)
        lsa_model.fit(lsa_train)
        self.trained_model = lsa_model

        # Get fitted document-topic matrix
        lsa_results = lsa_model.transform(lsa_train)
        self.trained_results = lsa_results     


    def get_top_topics(self, n_top_topics=5, min_weight=None, return_type='df', return_weights=True):

        if isinstance(n_top_topics, bool):
            raise TypeError('Parameter n_top_topics must be of type int')
        if not isinstance(n_top_topics, int):
            raise TypeError('Parameter n_top_topics must be of type int')
        if n_top_topics <= 0:
            raise ValueError('Parameter n_top_topics must be a positive integer')
        if min_weight:
            if not isinstance(min_weight, float):
                raise TypeError('Argument "min_weight" must be None or of type float')
        if not isinstance(return_type, str):
            raise TypeError('Argument "return_type" must be of type str')
        if return_type.lower() not in ['df', 'dataframe', 'list']:
            raise ValueError('Argument "return_type" must be one of: "df", "dataframe", "list"')
        if not isinstance(return_weights, bool):
            raise TypeError('Argument "return_weights" must be of type bool')

        if self.model_info['model'] == 'lda':
            top_clusters = []
            for doc, y in enumerate(self.trained_results):
                if len(y) == 1:
                    if return_type == 'list':
                        if return_weights:
                            top_clusters.append([y[0]])
                        else:
                            top_clusters.extend([tuple((doc, y[0][0]))])
                    else:
                        top_clusters.append([doc] + list(y[0]))
                else:
                    y.sort(key=lambda x:x[1], reverse=True)
                    y_top = y[:n_top_topics]
                    if return_type == 'list':
                        if return_weights:    
                            top_clusters.append(y_top)
                        else:
                            top_clusters.extend([tuple((doc, x[0])) for x in y_top])
                    else:
                        y_top = [tuple([doc] + list(x)) for x in y_top]
                        top_clusters.extend(y_top)
        elif self.model_info['model'] == 'lda_lda' or self.model_info['model'] == 'lsa':
            if n_top_topics == 1:
                topics = np.argmax(self.trained_results, axis=1)
                weights = np.max(self.trained_results, axis=1)
                top_clusters = zip(range(len(topics)), topics, weights)
            else:
                topics_list = []
                weights_list = []
                doc_ids = []
                for doc, topic_dist in enumerate(self.trained_results):
                    topics = topic_dist.argsort()[:-(n_top_topics+1):-1]
                    topics_list.extend(topics)
                    weights_list.extend(topic_dist[topics])
                    doc_ids.extend([doc] * len(topics))
                top_clusters = zip(doc_ids, topics_list, weights_list)
        
        if return_type.lower() in ['df', 'dataframe']:
            top_clusters = pd.DataFrame(top_clusters, columns=['RowId', 'Topic', 'Weight']) 
            if min_weight:
                if not isinstance(min_weight, float):
                    raise TypeError('Parameter min_weight must be None or of type float')
                if min_weight <= 0.0:
                    raise ValueError('Parameter min_weight must be a positive number')
                top_clusters = top_clusters[top_clusters['Weight'] >= min_weight].reset_index(drop=True)
        
        return top_clusters


    def getTopNClusters(self, output_type='topics', n_top_clusters=5):

        top_clusters_df = pd.DataFrame(columns=['Topic', 'Weight'])
        if output_type == 'topics':
            n_top_clusters = 1
        for doc, topic_dist in enumerate(self.trained_results):
            temp_df = pd.DataFrame(index=[doc]*n_top_clusters)
            temp_df['Topic'] = topic_dist.argsort()[:-(n_top_clusters+1):-1]
            temp_df['Weight'] = topic_dist[topic_dist.argsort()[:-(n_top_clusters+1):-1]]
            top_clusters_df = top_clusters_df.append(temp_df)
        
        top_clusters_df = top_clusters_df.reset_index().rename(columns={'index':'Id'})      
        
        if output_type == 'weights':
            out_dict = {}
            for i in range(len(top_clusters_df)):
                if top_clusters_df.loc[i, 'Id'] not in out_dict:
                    out_dict[top_clusters_df.loc[i, 'Id']] = []
                out_dict[top_clusters_df.loc[i, 'Id']].append(((top_clusters_df.loc[i, 'Topic'], top_clusters_df.loc[i, 'Weight'])))
                
            out_list = []
            for k in range(len(out_dict)):
                if k in out_dict:
                    out_list.append(out_dict[k])
            return out_list
        elif output_type == 'topics':
            out_list = []
            for i in range(len(top_clusters_df)):
                out_list.append((i, top_clusters_df.loc[i, 'Topic']))
            return out_list


    def get_topic_docs(self, n_top_docs=10):
        
        if not isinstance(n_top_docs, int):
            raise TypeError('Parameter n_top_docs must be of type int')
        if n_top_docs < 1:
            raise ValueError('Parameter n_top_docs must be a positive integer')

        all_topic_docs = pd.DataFrame()
        top_topics_df = self.get_top_topics(n_top_topics=10)
        for i in range(self.model_info['num_topics']):
            temp = top_topics_df[top_topics_df['Topic'] == i]
            if len(temp) > 0:
                temp = temp.sort_values(by=['Weight'], ascending=False).reset_index(drop=True)
                temp = temp[0:n_top_docs]
                all_topic_docs = all_topic_docs.append(temp)
        
        df = self.df[['RowId', self.target_column]]
        topic_docs = pd.merge(all_topic_docs, df, left_on='RowId', right_on='RowId')
        
        return topic_docs


    def get_topic_words(self, n_top_words=20):

        if isinstance(n_top_words, bool):
            raise TypeError('Argmunet n_top_words must be of type int')
        if isinstance(n_top_words, int) == False:
            raise TypeError('Argmunet n_top_words must be of type int')
        if n_top_words < 1:
            raise ValueError('Argument n_top_words must be a positive integer')
        
        topic_term_df = pd.DataFrame()
        if self.model_info['model'] == 'lda':
            for k in range(self.model_info['num_topics']):
                terms = pd.DataFrame(self.trained_model.show_topic(k, topn=n_top_words))
                terms['Topic'] = k
                topic_term_df = topic_term_df.append(terms)
            topic_term_df = topic_term_df.rename(columns={0:'Term', 1:'Weight'})
            topic_term_df = topic_term_df.reset_index(drop=True)
            return topic_term_df
        elif self.model_info['model'] == 'lsa':
            topic_word_dist = self.trained_model.components_
        elif self.model_info['model'] == 'lda_lda':
            topic_word_dist = self.trained_model.topic_word_
        for topic, topic_dist in enumerate(topic_word_dist):
            terms_dist = zip(self.vocab, topic_dist)
            terms = sorted(terms_dist, key=lambda x:x[1], reverse=True)[:n_top_words]
            terms = pd.DataFrame(terms).rename(columns={0:'Term', 1:'Weight'})
            terms['Topic'] = topic
            topic_term_df = topic_term_df.append(terms)
        
        topic_term_df = topic_term_df.reset_index(drop=True)
        
        return topic_term_df


    def preprocess_dataframe(self, df, target_column, min_chars=None, min_words=None):

        if self.verbose:
            print('Starting Preprocessing.......\n')
        
        # Create ID column
        df = df.reset_index().rename(columns={'index':'original_id'})
        original_len = len(df)
        original_df = df.copy()
        
        # Identify null rows       
        df = df[~df[target_column].isnull()].reset_index(drop=True)
        num_null = original_len - len(df)
        if num_null == original_len:
            raise ValueError('All rows in dataframe are null')
        if self.verbose:
            if num_null == 0:
                print('\tNo null rows found')
            else:
                print('\t%s null rows removed' %(num_null))

         # Convert target column to strings
        df[target_column] = df[target_column].apply(lambda x: str(x))

        # Tokenize text
        corpus = []
        for i in range(len(df)):
            corpus.append(self.tokenizer.tokenize(df.loc[i, target_column].lower()))
        
        # Initial IsValid screen
        invalid_tracking = []
        if self.invalid_list:
            invalid_corpus = []
            for i in range(len(self.invalid_list)):
                invalid_corpus.append(self.tokenizer.tokenize(self.invalid_list[i].lower()))
            for i in range(len(corpus)):
                if corpus[i] in invalid_corpus:
                    invalid_tracking.append(i)            

        # Identify phrases
        if self.phrases:
            corpus = self.__generate_phrases__(corpus, min_count=5)
        
        # Remove stopwords
        if self.stopwords:
            corpus = [[x for x in doc if x not in self.stopwords] for doc in corpus]
        
        # Remove numbers
        if self.remove_numbers:
            corpus = [[x for x in doc if not x.isnumeric()] for doc in corpus]

        # Full IsValid screen
        for i in range(len(corpus)):
            is_valid = self.__is_valid_screen__(corpus[i], min_chars=min_chars, min_words=min_words, invalid_corpus=invalid_corpus)
            if not is_valid and i not in invalid_tracking:
                invalid_tracking.append(i)

        # Remove invalid texts
        if len(invalid_tracking) == 0:
            if self.verbose:
                print('\tNo invalid rows found')
        elif len(invalid_tracking) == len(df):
            raise ValueError('All rows in dataframe are invalid or null')
        else:
            for i in sorted(invalid_tracking, reverse=True):
                del corpus[i]
            df = df.drop(invalid_tracking).reset_index(drop=True)
            if self.verbose:
                print('\t%s additional invalid rows removed' %len(invalid_tracking))

        # Convert text back to strings
        target_column = target_column + '_preprocessed'
        df[target_column] = ''
        for i in range(len(df)):
            df.loc[i, target_column] = ' '.join(x for x in corpus[i])

        df = df.reset_index(drop=True)

        if self.verbose:
            print('\nPreprocessing completed\n')
        
        return corpus, df, target_column


    def selectK(self, input_df, min_k=10, max_k=115, divisor=25):
    
        k = int(len(input_df) / divisor)
        
        if k < min_k:
            k = min_k
        if k > max_k:
            k = max_k
        
        return k
        

    def __modelPipelineLDA__(self, df, tfidf=False, n_topics=20, min_df=5, n_iter=1500):
        
        lda_df = df.copy().reset_index(drop=True)

        # Fit Vectorizer
        if tfidf:
            countvec = TfidfVectorizer(min_df=min_df)
        else:
            countvec = CountVectorizer(min_df=min_df)        
        lda_train = countvec.fit_transform(lda_df[self.target_column])
        self.vocab = countvec.get_feature_names()

        # Fit LDA
        lda_model = lda.LDA(n_topics=n_topics, n_iter=n_iter, random_state=self.random_state)
        lda_model.fit(lda_train)
        self.trained_model = lda_model
        
        # Get fitted document-topic matrix
        lda_results = lda_model.transform(lda_train)
        self.trained_results = lda_results


    def extract_topics(self, input_df, target_column, model='lda', num_topics=20, tfidf=False, min_k=5, max_k=40, step_size=5):

        num_topics = self.__error_checking__(df=input_df, target_column=target_column, model=model, num_topics=num_topics, tfidf=tfidf, min_k=min_k, max_k=max_k, step_size=step_size)
        corpus, df, target_column = self.preprocess_dataframe(df=input_df, target_column=target_column)
        self.target_column = target_column
        self.model_info = {}

        if self.verbose:
            print('Starting Topic Modeling.......\n')
        model = model.lower()
        if model == 'lda':
            self.model_info['model'] = 'lda'
            self.__train_lda__(corpus=corpus, num_topics=num_topics, min_df=5, max_df=0.5, tfidf=tfidf)
        elif model == 'lda_lda':
            self.model_info['model'] = 'lda_lda'
            self.__modelPipelineLDA__(df=df)
        elif model == 'lsa':
            self.model_info['model'] = 'lsa'
            self.__train_lsa__(df=df)
        
        # Assign top topic to each text
        df = df.reset_index().rename(columns={'index':'RowId'})
        self.df = df
        df_results = self.__assign_top_topic__(input_df=df)
        
        return df_results