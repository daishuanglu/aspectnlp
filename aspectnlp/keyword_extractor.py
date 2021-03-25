# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 12:44:13 2019

@author: adshan


"""
from collections import Counter, namedtuple
import csv
import hashlib
import json
import os
import networkx as nx
import nltk
from nltk import ngrams
from nltk.tokenize import PunktSentenceTokenizer, RegexpTokenizer
import numpy as np #
import re
import spacy
import string

_STOPWORDS_CSV_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/stopwords_keywordextractor.csv')

class ExtractorRake:
    
    def __init__(self, stopwords=None, stopword_csv_path=_STOPWORDS_CSV_FILE_PATH):
        if stopwords:
            self.stopwords = stopwords
        else:
            with open(stopword_csv_path, 'r') as f:
                reader = csv.reader(f)
                try:
                    self.stopwords = [stopword for sublist in reader for stopword in sublist] # Flatten the nested list of csv.reader(f)
                except ValueError as e:
                    raise ValueError('The stopword lookup file is not properly formatted.  All rows should contain '
                        'exactly one stopword per row.  Make sure there are no blank lines at the end of the file.') from e
        self.word_delimiters = '!"#$%&\'()*+,./:;<=>?@[\\]^`{|}~'
        self.sentence_tokenizer = re.compile(r'[%s]|\s[-]|[-]\s'%self.word_delimiters)
        self.adjoining_keywords = True
        self.candidate_tokenizer = re.compile('|'.join([r'(?<![-])(\b%s(?![\w-]))'%x for x in self.stopwords]))
        self.whitespace_patterns = re.compile(r'(^ )|( $)|(^[-])')
            
    def getAdjoiningKeywords(self, all_tokens):
        
        adjoined_keywords = []
        for sent in all_tokens:
            if len(sent) > 2:
                idxs = [idx for idx, x in enumerate(sent) if x not in self.stopwords]
                for i in range(len(idxs)-1):
                    if idxs[i+1] - idxs[i] > 1:
                        adjoined_keywords.append(tuple(sent[idxs[i]:idxs[i+1]+1]))
        if adjoined_keywords:
            adjoined_keywords = {tuple([tuple(x.split()) for x in k]):0.0 for k, v in dict(Counter(adjoined_keywords)).items() if v > 1}
            return adjoined_keywords
        else:
            return {}
    
    def createCandidateKeywords(self, text):
        
        text = text.lower()
        sentence_tokens = self.sentence_tokenizer.split(text)
        sentence_tokens = [self.candidate_tokenizer.split(x) for x in sentence_tokens if x not in [' ', '']]
        for i in range(len(sentence_tokens)):
            sentence_tokens[i] = [self.whitespace_patterns.sub('', x) for x in sentence_tokens[i] if x if x not in [' ', '']]
        
        if self.adjoining_keywords:
            adjoined_keywords = self.getAdjoiningKeywords(all_tokens=sentence_tokens)
        else:
            adjoined_keywords = {}
        candidate_tokens = [tuple(y.split()) for x in sentence_tokens for y in x if y not in self.stopwords]
        candidate_tokens = dict(Counter(candidate_tokens))
        
        return candidate_tokens, adjoined_keywords
    
    def calculateWordScores(self, candidate_tokens):
        
        word_scores = {}
        for candidate in candidate_tokens.items():
            for x in candidate[0]:
                if x in word_scores:
                    word_scores[x]['Frequency'] += 1 * candidate[1]
                    word_scores[x]['Degree'] += len(candidate[0]) * candidate[1]
                else:
                    word_scores[x] = {}
                    word_scores[x]['Degree'] = len(candidate[0]) * candidate[1]
                    word_scores[x]['Frequency'] = 1 * candidate[1]
        
        word_scores = {k:v['Degree']/v['Frequency'] for k,v in word_scores.items()}
        
        return word_scores
    
    def extractKeywords(self, text):
        
        candidate_tokens, adjoining_tokens = self.createCandidateKeywords(text)
        word_dict = self.calculateWordScores(candidate_tokens)
        
        results_dict = {k:0.0 for k in candidate_tokens.keys()}
        for candidate in results_dict.keys():
            for word in candidate:
                results_dict[candidate] += word_dict[word]
        
        if adjoining_tokens:
            for k,v in adjoining_tokens.items():
                output_tuple = []
                for i in k:
                    output_tuple.extend(i)
                    if i in results_dict:
                        v += results_dict[i]
                results_dict[tuple(output_tuple)] = v
        
        results_dict = {' '.join(k):v for k,v in results_dict.items()}
        results_dict = sorted(results_dict.items(), key=lambda x: x[1], reverse=True)
        
        return dict(results_dict)


class ExtractorNgram:
    
    def __init__(self, stopwords=None, stopword_csv_path=_STOPWORDS_CSV_FILE_PATH, sentence_tokenizer=None, word_tokenizer=None):
        if stopwords:
            self.stopwords = stopwords
        else:
            with open(stopword_csv_path, 'r') as f:
                reader = csv.reader(f)
                try:
                    self.stopwords = [stopword for sublist in reader for stopword in sublist] # Flatten the nested list of csv.reader(f)
                except ValueError as e:
                    raise ValueError('The stopword lookup file is not properly formatted.  All rows should contain '
                        'exactly one stopword per row.  Make sure there are no blank lines at the end of the file.') from e
        if sentence_tokenizer:
            self.sentence_tokenizer = sentence_tokenizer
        else:
            self.sentence_tokenizer = PunktSentenceTokenizer()
        if word_tokenizer:
            self.word_tokenizer = word_tokenizer
        else:
            self.word_tokenizer = RegexpTokenizer("[\w']+")
    
    def tokenizeText(self, text, pos_tag=True, noun_list=[]):
        tokenized_sentences = []
        if pos_tag:    
            sentences = self.sentence_tokenizer.tokenize(text)
            for sentence in sentences:
                tokenized_sentence = nltk.pos_tag(self.word_tokenizer.tokenize(sentence))
                tokenized_sentence = [(x[0].lower(), x[1]) for x in tokenized_sentence]
                tokenized_sentence = [(x[0], 'NN') if x[0] in noun_list else (x[0], x[1]) for x in tokenized_sentence]
                tokenized_sentences.append(tokenized_sentence)
        else:
            text = text.lower()
            sentences = self.sentence_tokenizer.tokenize(text)
            for sentence in sentences:
                tokenized_sentences.append(self.word_tokenizer.tokenize(sentence))
        
        return tokenized_sentences

    def removeNgramStopwords(self, input_key, length=1, pos_tag=False, removal_type='All'):
        if length == 1:
            if isinstance(input_key, str):
                if input_key in self.stopwords:
                    return False
            else:
                if input_key[0] in self.stopwords:
                    return False
        else:
            if removal_type == 'All':
                if pos_tag:
                    for i in range(len(input_key)):
                        if input_key[i][0] in self.stopwords:
                            return False
                else:
                    for i in range(len(input_key)):
                        if input_key[i] in self.stopwords:
                            return False
            elif removal_type == 'Outside':
                if pos_tag:
                    if input_key[0][0] in self.stopwords or input_key[len(input_key)-1][0] in self.stopwords:
                        return False
                else:
                    if input_key[0] in self.stopwords or input_key[len(input_key)-1] in self.stopwords:
                        return False
        return True

    def extractNgrams(self, text, length=1, num_returned=None, pos_tag=False, removal_type=None, noun_list=[]):
        
        sentences = self.tokenizeText(text=text, pos_tag=pos_tag, noun_list=noun_list)
        
        ngrams_list = []
        if length == 1:
            [ngrams_list.extend(x) for x in sentences]
        else:
            for sentence in sentences:
                ngrams_list.extend([x for x in ngrams(sentence, length)])
                
        ngrams_list = dict(Counter(ngrams_list))
        
        if removal_type == 'All':
            ngrams_list = {k:v for k, v in ngrams_list.items() if self.removeNgramStopwords(k, length=length, pos_tag=pos_tag, removal_type='All')}
        elif removal_type == 'Outside':
            ngrams_list = {k:v for k, v in ngrams_list.items() if self.removeNgramStopwords(k, length=length, pos_tag=pos_tag, removal_type='Outside')}
               
        return ngrams_list
        
    def extractKeywords(self, text, length=1, num_returned=None, pos_tag=False, removal_type=None):
        
        ngrams_list = self.extractNgrams(text=text, length=length, num_returned=num_returned, pos_tag=pos_tag, removal_type=removal_type)
        if num_returned:
            return dict(sorted(ngrams_list.items(), key=lambda x: x[1], reverse=True)[:num_returned])
        else:
            return ngrams_list
    
        
class ExtractorDikpe(ExtractorNgram):
    
    def __init__(self, stopwords=None, stopword_csv_path=_STOPWORDS_CSV_FILE_PATH, sentence_tokenizer=None, word_tokenizer=None):
        if stopwords:
            self.stopwords = stopwords
        else:
            with open(stopword_csv_path, 'r') as f:
                reader = csv.reader(f)
                try:
                    self.stopwords = [stopword for sublist in reader for stopword in sublist] # Flatten the nested list of csv.reader(f)
                except ValueError as e:
                    raise ValueError('The stopword lookup file is not properly formatted.  All rows should contain '
                        'exactly one stopword per row.  Make sure there are no blank lines at the end of the file.') from e
        if sentence_tokenizer:
            self.sentence_tokenizer = sentence_tokenizer
        else:
            self.sentence_tokenizer = PunktSentenceTokenizer()
        if word_tokenizer:
            self.word_tokenizer = word_tokenizer
        else:
            self.word_tokenizer = RegexpTokenizer("[\w']+")
    
    def createNestedDict(self, input_dict, length=1):
        ngrams_dict = {}
        if length == 1:
            for phrase in input_dict.keys():
                ngrams_dict[phrase[0]] = {}
                ngrams_dict[phrase[0]]['Pos'] = []
                ngrams_dict[phrase[0]]['Pos'].append(phrase[1])
                ngrams_dict[phrase[0]]['Count'] = input_dict[phrase]
        else:
            for phrase in input_dict.keys():
                phrase_string = []
                pos_tags = []
                for word in phrase:
                    phrase_string.append(word[0])
                    pos_tags.append(word[1])
                phrase_string = '; '.join([x for x in phrase_string])
                ngrams_dict[phrase_string] = {}
                ngrams_dict[phrase_string]['Pos'] = pos_tags
                ngrams_dict[phrase_string]['Count'] = input_dict[phrase]
        
        return ngrams_dict
    
    def filterNgramList(self, pos_list):
        for pos in pos_list:
            if pos.startswith('NN') or pos.startswith('VB') or pos.startswith('JJ'):
                return True
        return False
    
    def calculatePosValue(self, pos_list):
        count = 0
        for pos in pos_list:
            if pos.startswith('NN'):
                count += 1
        if count == 0:
            return 0.25
        else:
            return count / len(pos_list)
    
    def calculateOccurences(self, token_list, input_list):
        idxs = []
        for i in range(len(token_list)-len(input_list)+1):
            if token_list[i] == input_list[0]:
                if token_list[i:i+len(input_list)] == input_list:
                    idxs.append(i)
                    
        depth = 1 - idxs[0] / len(token_list) 
        last_occurrence = idxs[-1] / len(token_list)
        if len(idxs) == 1:
            lifespan = 0
        else:
            lifespan = (idxs[-1] - idxs[0]) / len(token_list)
            
        return [depth, last_occurrence, lifespan]
    
    def calculateFeatures(self, text, ngrams_dict):
        token_list = []
        [token_list.extend(x) for x in self.tokenizeText(text, pos_tag=False)]
        
        total_occurrences = sum(ngrams_dict[x]['Count'] for x in ngrams_dict)
        for phrase in ngrams_dict:
            ngrams_dict[phrase]['Features'] = []
            ngrams_dict[phrase]['Features'].append(ngrams_dict[phrase]['Count'] / total_occurrences)
            ngrams_dict[phrase]['Features'].append(self.calculatePosValue(ngrams_dict[phrase]['Pos']))
            ngrams_dict[phrase]['Features'].extend(self.calculateOccurences(token_list=token_list, input_list=phrase.split('; ')))
            ngrams_dict[phrase]['Keyphraseness'] = sum(ngrams_dict[phrase]['Features']) / len(ngrams_dict[phrase]['Features'])

        return ngrams_dict

    def extractKeywords(self, text, num_returned=10, noun_list=[]):
        unigrams_dict = self.extractNgrams(text=text, length=1, pos_tag=True, removal_type='Outside', noun_list=noun_list)
        bigrams_dict = self.extractNgrams(text=text, length=2, pos_tag=True, removal_type='Outside', noun_list=noun_list)
        trigrams_dict = self.extractNgrams(text=text, length=3, pos_tag=True, removal_type='Outside', noun_list=noun_list)        
        
        unigrams_dict = self.createNestedDict(input_dict=unigrams_dict, length=1)
        bigrams_dict = self.createNestedDict(input_dict=bigrams_dict, length=2)
        trigrams_dict = self.createNestedDict(input_dict=trigrams_dict, length=3)
        
        unigrams_dict = {k:v for k,v in unigrams_dict.items() if self.filterNgramList(v['Pos'])}

        unigrams_dict = self.calculateFeatures(text=text, ngrams_dict=unigrams_dict)
        bigrams_dict = self.calculateFeatures(text=text, ngrams_dict=bigrams_dict)
        trigrams_dict = self.calculateFeatures(text=text, ngrams_dict=trigrams_dict)

        unigrams_dict = sorted({k:v['Keyphraseness'] for k,v in unigrams_dict.items()}.items(), key=lambda x: x[1], reverse=True)
        bigrams_dict = sorted({re.sub(r'; ', ' ', k):v['Keyphraseness'] for k,v in bigrams_dict.items()}.items(), key=lambda x: x[1], reverse=True)
        trigrams_dict = sorted({re.sub(r'; ', ' ', k):v['Keyphraseness'] for k,v in trigrams_dict.items()}.items(), key=lambda x: x[1], reverse=True)
        
        full_results = []
        full_results.extend([x for x in unigrams_dict if x[1] >= unigrams_dict[int(round(len(unigrams_dict)*0.4))][1]])
        full_results.extend([x for x in bigrams_dict if x[1] >= bigrams_dict[int(round(len(bigrams_dict)*0.4))][1]])
        full_results.extend([x for x in trigrams_dict if x[1] >= trigrams_dict[int(round(len(trigrams_dict)*0.2))][1]])
        
        return dict(full_results)


class ExtractorTextrank:
    
    def __init__(self, stopwords=None, stopword_csv_path=_STOPWORDS_CSV_FILE_PATH):
        if stopwords:
            self.stopwords = stopwords
        else:
            with open(stopword_csv_path, 'r') as f:
                reader = csv.reader(f)
                try:
                    self.stopwords = [stopword for sublist in reader for stopword in sublist] # Flatten the nested list of csv.reader(f)
                except ValueError as e:
                    raise ValueError('The stopword lookup file is not properly formatted.  All rows should contain '
                        'exactly one stopword per row.  Make sure there are no blank lines at the end of the file.') from e
        self.PAT_FORWARD = re.compile("\n\-+ Forwarded message \-+\n")
        self.PAT_REPLIED = re.compile("\nOn.*\d+.*\n?wrote\:\n+\>")
        self.PAT_UNSUBSC = re.compile("\n\-+\nTo unsubscribe,.*\nFor additional commands,.*")
        self.PAT_PUNCT = re.compile(r'^\W+$')
        self.PAT_SPACE = re.compile(r'\_+$')
        self.POS_KEEPS = ['v', 'n', 'j']
        self.POS_LEMMA = ['v', 'n']
        self.UNIQ_WORDS = { ".": 0 }
        self.spacy_nlp = spacy.load("en")
        self.ParsedGraf = namedtuple('ParsedGraf', 'id, sha1, graf')
        self.WordNode = namedtuple('WordNode', 'word_id, raw, root, pos, keep, idx')
        self.RankedLexeme = namedtuple('RankedLexeme', 'text, rank, ids, pos, count')
        self.SummarySent = namedtuple('SummarySent', 'dist, idx, text')

    def split_grafs(self, lines):
        graf = []
        for line in lines:
            line = line.strip()
            if len(line) < 1:
                if len(graf) > 0:
                    yield "\n".join(graf)
                    graf = []
            else:
                graf.append(line)
        if len(graf) > 0:
            yield "\n".join(graf)

    def filter_quotes (self, text, is_email=True):
        if is_email:
            text = filter(lambda x: x in string.printable, text)
            m = self.PAT_FORWARD.split(text, re.M)
            if m and len(m) > 1:
                text = m[0]
            m = self.PAT_REPLIED.split(text, re.M)
            if m and len(m) > 1:
                text = m[0]
            m = self.PAT_UNSUBSC.split(text, re.M)
            if m:
                text = m[0]
        lines = []
        for line in text.split("\n"):
            if line.startswith(">"):
                lines.append("")
            else:
                lines.append(line)
    
        return list(self.split_grafs(lines))
    
    def is_not_word(self, word):
        return self.PAT_PUNCT.match(word) or self.PAT_SPACE.match(word)

    def fix_microsoft (self, foo):
        #fix special case for `c#`, `f#`, etc.; thanks Microsoft
        i = 0
        bar = []    
        while i < len(foo):
            text, lemma, pos, tag = foo[i]
            if (text == "#") and (i > 0):
                prev_tok = bar[-1]
                prev_tok[0] += "#"
                prev_tok[1] += "#"
                bar[-1] = prev_tok
            else:
                bar.append(foo[i])
            i += 1
    
        return bar
    
    def fix_hyphenation(self, foo):
        i = 0
        bar = []
        while i < len(foo):
            text, lemma, pos, tag = foo[i]
            if (tag == "HYPH") and (i > 0) and (i < len(foo) - 1):
                prev_tok = bar[-1]
                next_tok = foo[i + 1]
    
                prev_tok[0] += "-" + next_tok[0]
                prev_tok[1] += "-" + next_tok[1]
    
                bar[-1] = prev_tok
                i += 2
            else:
                bar.append(foo[i])
                i += 1
    
        return bar

    def get_word_id(self, root):   
        if root not in self.UNIQ_WORDS:
            self.UNIQ_WORDS[root] = len(self.UNIQ_WORDS)
    
        return self.UNIQ_WORDS[root]

    def parse_graf (self, doc_id, graf_text, base_idx):
        markup = []
        new_base_idx = base_idx
        doc = self.spacy_nlp(graf_text)
    
        for span in doc.sents:
            graf = []
            digest = hashlib.sha1()
            word_list = []
            for tag_idx in range(span.start, span.end):
                token = doc[tag_idx]    
                word_list.append([token.text, token.lemma_, token.pos_, token.tag_])
            corrected_words = self.fix_microsoft(self.fix_hyphenation(word_list))    
            for tok_text, tok_lemma, tok_pos, tok_tag in corrected_words:
                word = self.WordNode(word_id=0, raw=tok_text, root=tok_text.lower(), pos=tok_tag, keep=0, idx=new_base_idx)
                if self.is_not_word(tok_text) or (tok_tag == "SYM"):
                    pos_family = '.'
                    word = word._replace(pos=pos_family)
                else:
                    pos_family = tok_tag.lower()[0]
                if pos_family in self.POS_LEMMA:
                    word = word._replace(root=tok_lemma)
                if pos_family in self.POS_KEEPS:
                    word = word._replace(word_id=self.get_word_id(word.root), keep=1)
                digest.update(word.root.encode('utf-8'))
                graf.append(list(word))
                new_base_idx += 1
            markup.append(self.ParsedGraf(id=doc_id, sha1=digest.hexdigest(), graf=graf))
        
        return markup, new_base_idx

    def parse_doc(self, text):
        base_idx = 0
        
        for graf_text in self.filter_quotes(text, is_email=False):
            grafs, new_base_idx = self.parse_graf("777", graf_text, base_idx)
            base_idx = new_base_idx
            for graf in grafs:
                yield graf

    def get_tiles(self, graf, size=3):
        keeps = list(filter(lambda w: w.word_id > 0, graf))
        keeps_len = len(keeps)
        for i in iter(range(0, keeps_len - 1)):
            w0 = keeps[i]
            for j in iter(range(i + 1, min(keeps_len, i + 1 + size))):
                w1 = keeps[j]
                if (w1.idx - w0.idx) <= size:
                    yield (w0.root, w1.root,)

    def build_graph(self, inp_iter):
        graph = nx.DiGraph()
        for meta in inp_iter:
            for pair in self.get_tiles(map(self.WordNode._make, meta["graf"])):
                for word_id in pair:
                    if not graph.has_node(word_id):
                        graph.add_node(word_id)
    
                if "edge" in dir(graph):
                    graph.edge[pair[0]][pair[1]]["weight"] += 1.0
                else:
                    graph.add_edge(pair[0], pair[1], weight=1.0)
        
        return graph

    def find_chunk_sub(self, phrase, n_p, i):
        for j in iter(range(0, len(n_p))):
            p = phrase[i + j]
            if p.text != n_p[j]:
                return None
    
        return phrase[i:i + len(n_p)]

    def find_chunk(self, phrase, n_p):
        for i in iter(range(0, len(phrase))):
            parsed_np = self.find_chunk_sub(phrase, n_p, i)
            if parsed_np:
                return parsed_np

    def enumerate_chunks(self, phrase):
        if (len(phrase) > 1):
            found = False
            text = " ".join([rl.text for rl in phrase])
            doc = self.spacy_nlp(text.strip())
            for n_p in doc.noun_chunks:
                if n_p.text != text:
                    found = True
                    yield n_p.text, self.find_chunk(phrase, n_p.text.split(" "))
            if not found and all([rl.pos[0] != "v" for rl in phrase]):
                yield text, phrase

    def collect_keyword(self, sent, ranks, stopwords):
        for w in sent:
            if (w.word_id > 0) and (w.root in ranks) and (w.pos[0] in "NV") and (w.root not in stopwords):
                rl = self.RankedLexeme(text=w.raw.lower(), rank=ranks[w.root]/2.0, ids=[w.word_id], pos=w.pos.lower(), count=1)
    
                yield rl

    def find_entity(self, sent, ranks, ent, i):
        if i >= len(sent):
            return None, None
        else:
            for j in iter(range(0, len(ent))):
                w = sent[i + j]
    
                if w.raw != ent[j]:
                    return self.find_entity(sent, ranks, ent, i + 1)
            w_ranks = []
            w_ids = []
            for w in sent[i:i + len(ent)]:
                w_ids.append(w.word_id)
                if w.root in ranks:
                    w_ranks.append(ranks[w.root])
                else:
                    w_ranks.append(0.0)
    
            return w_ranks, w_ids

    def collect_entities(self, sent, ranks):
        sent_text = " ".join([w.raw for w in sent])
        for ent in self.spacy_nlp(sent_text).ents:
            if (ent.label_ not in ["CARDINAL"]) and (ent.text.lower() not in self.stopwords):
                w_ranks, w_ids = self.find_entity(sent, ranks, ent.text.split(" "), 0)
                if w_ranks and w_ids:
                    rl = self.RankedLexeme(text=ent.text.lower(), rank=w_ranks, ids=w_ids, pos="np", count=1)
                    yield rl

    def collect_phrases(self, sent, ranks):
        tail = 0
        last_idx = sent[0].idx - 1
        phrase = []
        while tail < len(sent):
            w = sent[tail]
            if (w.word_id > 0) and (w.root in ranks) and ((w.idx - last_idx) == 1):
                rl = self.RankedLexeme(text=w.raw.lower(), rank=ranks[w.root], ids=w.word_id, pos=w.pos.lower(), count=1)
                phrase.append(rl)
            else:
                for text, p in self.enumerate_chunks(phrase):
                    if p:
                        id_list = [rl.ids for rl in p]
                        rank_list = [rl.rank for rl in p]
                        np_rl = self.RankedLexeme(text=text, rank=rank_list, ids=id_list, pos="np", count=1)
                        yield np_rl
                phrase = []
            last_idx = w.idx
            tail += 1
    
    def calc_rms(self, values):
        #return math.sqrt(sum([x**2.0 for x in values])) / float(len(values))
        # take the max() which works fine
        return max(values)

    def normalize_key_phrases(self, path, ranks, skip_ner=True):
        single_lex = {}
        phrase_lex = {}
        for meta in path:
            sent = [w for w in map(self.WordNode._make, meta["graf"])]
            for rl in self.collect_keyword(sent, ranks, self.stopwords):
                id = str(rl.ids)
                if id not in single_lex:
                    single_lex[id] = rl
                else:
                    prev_lex = single_lex[id]
                    single_lex[id] = rl._replace(count = prev_lex.count + 1)
            if not skip_ner:
                for rl in self.collect_entities(sent, ranks):
                    id = str(rl.ids)
    
                    if id not in phrase_lex:
                        phrase_lex[id] = rl
                    else:
                        prev_lex = phrase_lex[id]
                        phrase_lex[id] = rl._replace(count = prev_lex.count + 1)
            for rl in self.collect_phrases(sent, ranks):
                id = str(rl.ids)
                if id not in phrase_lex:
                    phrase_lex[id] = rl
                else:
                    prev_lex = phrase_lex[id]
                    phrase_lex[id] = rl._replace(count = prev_lex.count + 1)
    
        rank_list = [rl.rank for rl in single_lex.values()]
        repeated_roots = {}
        for rl in sorted(phrase_lex.values(), key=lambda rl: len(rl), reverse=True):
            rank_list = []
            for i in iter(range(0, len(rl.ids))):
                id = rl.ids[i]
                if not id in repeated_roots:
                    repeated_roots[id] = 1.0
                    rank_list.append(rl.rank[i])
                else:
                    repeated_roots[id] += 1.0
                    rank_list.append(rl.rank[i] / repeated_roots[id])
            phrase_rank = self.calc_rms(rank_list)
            single_lex[str(rl.ids)] = rl._replace(rank = phrase_rank)
        sum_ranks = sum([rl.rank for rl in single_lex.values()])
        for rl in sorted(single_lex.values(), key=lambda rl: rl.rank, reverse=True):
            if sum_ranks > 0.0:
                rl = rl._replace(rank=rl.rank / sum_ranks)
            elif rl.rank == 0.0:
                rl = rl._replace(rank=0.1)
            rl = rl._replace(text=re.sub(r"\s([\.\,\-\+\:\@])\s", r"\1", rl.text))
            yield rl

    def pretty_print (self, obj, indent=False):
        if indent:
            return json.dumps(obj, sort_keys=True, indent=2, separators=(',', ': '))
        else:
            return json.dumps(obj, sort_keys=True)
    
    def extractKeywords(self, text):
        self.UNIQ_WORDS = { ".": 0 }
        y = '['
        for graf in self.parse_doc(text):
            y = y + "%s" % self.pretty_print(graf._asdict())
            y = y + ','
        y = y[:len(y)-1]
        y = y + ']'
        graph = self.build_graph(json.loads(y))
        ranks = nx.pagerank(graph)
        results = []
        for r1 in self.normalize_key_phrases(json.loads(y), ranks):
            z2 = r1._asdict()
            results.append((z2['text'], z2['rank']))
        
        return dict(results)


class KeywordExtractor:
    """Class to extract keywords from a text. After class KeywordExtractor is initialized,
    keywords can be extracted using the function extractKeywords(text).
    
    """
    def __init__(self, stopwords=None, stopword_csv_path=_STOPWORDS_CSV_FILE_PATH, sentence_tokenizer=None, word_tokenizer=None):
        """Initializes a keyword extractor class. 
        
        Keyword Arguments
        -----------------
            stopwords {None, list} -- A list of stopwords to be used in keyword extraction.
                If stopwords=None (recommended), the nltk english stopwords list will be used (default: None)
            sentence_tokenizer {tokenizer-type Object} -- A tokenizer used for splitting text into sentences.
                If sentences_tokenizer=None (recommended), PunktSentenceTokenizer() will be used (default: None)
            word_tokenizer {tokenizer-type Object} -- A tokenizer used for splitting texts into words.
                If word_tokenizer=None (recommended), RegexpTokenizer("[\w']+") will be used (default: None)
        """
        
        if stopwords:
            self.stopwords = stopwords
        else:
            with open(stopword_csv_path, 'r') as f:
                reader = csv.reader(f)
                try:
                    self.stopwords = [stopword for sublist in reader for stopword in sublist] # Flatten the nested list of csv.reader(f)
                except ValueError as e:
                    raise ValueError('The stopword lookup file is not properly formatted.  All rows should contain '
                        'exactly one stopword per row.  Make sure there are no blank lines at the end of the file.') from e
        if sentence_tokenizer:
            self.sentence_tokenizer = sentence_tokenizer
        else:
            self.sentence_tokenizer = PunktSentenceTokenizer()
        if word_tokenizer:
            self.word_tokenizer = word_tokenizer
        else:
            self.word_tokenizer = RegexpTokenizer("[\w']+")
        self.Extractor_dikpe = ExtractorDikpe(stopwords=self.stopwords, sentence_tokenizer=self.sentence_tokenizer, word_tokenizer=self.word_tokenizer)
        self.Extractor_ngram = ExtractorNgram(stopwords=self.stopwords, sentence_tokenizer=self.sentence_tokenizer, word_tokenizer=self.word_tokenizer)
        self.Extractor_rake = ExtractorRake(stopwords=self.stopwords)
        self.Extractor_textrank = ExtractorTextrank(stopwords=self.stopwords)
        self.remove_websites = True
        if self.remove_websites == True:
            self.compileWebsitePatterns()

    def errorChecking(self, text, weights):
        
        if isinstance(text, str) == False:
            raise ValueError('Input should be of type str.')
        if isinstance(weights, list):
            if not weights:
                raise ValueError('weights cannot be empty')
            try:
                sum_weights = sum(weights)
            except:
                raise ValueError('Items in weights must be of type float or int')
            for x in weights:
                if x < 0.0:
                    raise ValueError('Items in weights cannot be negative')
            if sum_weights == 0.0:
                raise ValueError('Items in weights cannot sum to 0.0')
        else:
            raise ValueError('Weights must be of type list')
    
    def checkEdgeCases(self, text):
        # Check for single word
        word_list = self.word_tokenizer.tokenize(text)
        if not word_list:
            return [('',1.0)]
        elif len(set(word_list)) == 1:
            return [(word_list[0],1.0)]
        else:
            return None
    
    def compileWebsitePatterns(self):
        pattern_list = []
        url_chars = "[-.~:/?#\[\]@!$&'()*+,:;_=a-zA-Z0-9]"
        website_starts = ['https:', 'http:', 'www\.', 'aka\.m']
        domain_exts = ['\.com', '\.net', '\.org', '\.co', '\.us', '\.de', '\.uk', '\.info']
        pattern_list.append("((" + "|".join(x for x in website_starts) + ")(" + url_chars + ")+)")
        pattern_list.append("(" + url_chars + "+(" + "|".join(x for x in domain_exts) + ")[/]" + url_chars + "+)")
        pattern_list.append("(" + url_chars + "+(" + "|".join(x for x in domain_exts) + ")[/])")
        pattern_list.append("(" + url_chars + "+(" + "|".join(x for x in domain_exts) + "))")
        
        final_patterns = '|'.join(x for x in pattern_list)
        self.website_patterns = re.compile(r'(%s)'%final_patterns)
    
    def findWebsites(self, text):
        websites_mapping = {}
        websites_list = self.website_patterns.findall(text)
        if not websites_list:
            return text, None
        else:
            for x in websites_list:
                website_string = re.sub(r"[-.~:/?#\[\]@!$&'()*+,:;_=]", '', x[0])
                websites_mapping[website_string] = x[0]
                site = re.sub(r'[?.,\[\]*+&]', '[\g<0>]', x[0])
                text = re.sub(r'%s(?!/)'%site, website_string, text)
                 
        return text, websites_mapping
    
    def majorityVoting(self, input_dict, ensemble='Average', weights=None):
        
        if ensemble == 'Average':
            if not weights or len(set(weights)) == 1:
                return {k:np.mean(v['Weights']) for k,v in input_dict.items()}
            else:
                return {k:sum(x*y for x,y in zip(v['Weights'],weights))/sum(weights) for k,v in input_dict.items()}

    def normalizeKeywordWeights(self, input_dict, norm_type='MinMax0'):
        
        dict_max = max(input_dict.values())
        if norm_type == 'MinMax0':
            input_dict = {k: v/dict_max for k, v in input_dict.items()}
        elif norm_type == 'MinMax':
            dict_min = min(input_dict.values())
            input_dict = {k: (v-dict_min)/(dict_max-dict_min) for k, v in input_dict.items()}
        
        return input_dict

    def populateWeights(self, input_list, normalize_weights=True):
        
        results_dict = {}
        
        for i in range(len(input_list)):
            top_rank = None
            top_weight = None
            for x, y in enumerate(sorted(input_list[i].items(), key=lambda z: z[1], reverse=True)):
                if y[0] in results_dict:
                    results_dict[y[0]]['Weights'][i] = y[1]
                    if y[1] == top_weight:
                        results_dict[y[0]]['Ranks'][i] = top_rank
                    else:
                        top_weight = y[1]
                        top_rank = x
                        results_dict[y[0]]['Ranks'][i] = x
                else:
                    results_dict[y[0]] = {}
                    results_dict[y[0]]['Weights'] = [0.0 for z in range(len(input_list))]
                    results_dict[y[0]]['Ranks'] = [np.nan for z in range(len(input_list))]
                    results_dict[y[0]]['Weights'][i] = y[1]
                    if y[1] == top_weight:
                        results_dict[y[0]]['Ranks'][i] = top_rank
                    else:
                        top_weight = y[1]
                        top_rank = x
                        results_dict[y[0]]['Ranks'][i] = x
                        
        return results_dict
    
    def createKeywordList(self, text, method='rake', noun_list=[]):
        if method == 'rake':
            keyword_list = self.Extractor_rake.extractKeywords(text)
        elif method == 'dikpe':
            keyword_list = self.Extractor_dikpe.extractKeywords(text=text, noun_list=noun_list)
        elif method == 'textrank':
            keyword_list = self.Extractor_textrank.extractKeywords(text=text)
        elif method == 'unigram':
            keyword_list = self.Extractor_ngram.extractKeywords(text=text, length=1)
        elif method == 'bigram':
            keyword_list = self.Extractor_ngram.extractKeywords(text=text, length=2)
        elif method == 'trigram':
            keyword_list = self.Extractor_ngram.extractKeywords(text=text, length=3)        
        else:
            raise Exception("Method '%s' does not exist. Please choose from the following methods:\ndikpe, rake, textrank, unigram, bigram, trigram" %method)
        return keyword_list

    def removeDuplicatesKeyword(self, input_keyword):
        
        output_list = input_keyword.split()
        if not output_list:
            return input_keyword
        if len(set(output_list)) == 1:
            return output_list[0]
        if len(output_list) > 2 and len(output_list) % 2 == 0:
            temp = []
            for i in range(0, len(output_list), 2):
                temp.append((output_list[i], output_list[i+1]))
            if len(set(temp)) == 1:
                return output_list[0] + ' ' + output_list[1]
        if len(output_list) > 3 and len(output_list) % 3 == 0:
            temp = []
            for i in range(0, len(output_list), 3):
                temp.append((output_list[i], output_list[i+1], output_list[i+2]))
            if len(set(temp)) == 1:
                return ' '.join([x for x in output_list[0:3]])
        if len(output_list) > 4 and len(output_list) % 4 == 0:
            temp = []
            for i in range(0, len(output_list), 4):
                temp.append((output_list[i], output_list[i+1], output_list[i+2], output_list[i+3]))
            if len(set(temp)) == 1:
                return ' '.join([x for x in output_list[0:4]])
        return input_keyword
    
    def removeDuplicatesDict(self, input_dict):
        
        new_dict = {}
        for phrase in input_dict.keys():
            new_phrase = self.removeDuplicatesKeyword(input_keyword=phrase)
            if new_phrase in new_dict:
                new_dict[new_phrase] = max(new_dict[new_phrase], input_dict[phrase])
            else:
                new_dict[new_phrase] = input_dict[phrase]
        return new_dict

    def prepareOutput(self, keyword_list, threshold=None, max_return=None, scores=None, return_type=None):
        
        if threshold:
            keyword_list = [x for x in keyword_list if x[1] >= threshold]
        if max_return:
            keyword_list = keyword_list[:max_return]
        keyword_list = dict(keyword_list)
        keyword_list = self.removeDuplicatesDict(input_dict=keyword_list)
        keyword_list = sorted(keyword_list.items(), key=lambda x: x[1], reverse=True)
        if return_type == 'dict':
            return dict(keyword_list)
        elif return_type == 'list':
            if scores == False:
                return [x[0] for x in keyword_list]
            else:
                return keyword_list
        else:
            raise ValueError("return_type must be 'dict' or 'list'")
        return keyword_list

                    
    def extractKeywords(self, text, method=['rake', 'textrank'], weights=[0.6, 0.4], threshold=0.4, max_return=7, scores=False, return_type='list'):
        """Extracts keywords from a text. The choice of keyword extraction algorithm can be modified.

        Arguments
        ---------
            input_text {str} -- The text from which keywords will be extracted.
                Recommended that stopwords and punctuation are not removed from the text.
        
        Keyword Arguments
        -----------------
            method {str, list of str} -- The name(s) of the algorithm(s) used to extract keywords.
                If str, keywords will be extracted for the single algorithm identified in method. If list, 
                keywords will be extracted for each of the algorithms identified in method, and will
                be aggregated into single output.
                Currently supported methods: 'dikpe', 'rake', 'textrank', 'unigram', 'bigram', 'trigram'
                Recommended usage: 'rake', 'textrank', or ['rake', 'textrank', 'dikpe'] (default: ['rake', 'textrank', 'dikpe'])
            weights {list of float, list of int} -- Weight importance for the algorithm(s) used to extract keywords.
                Not used if method is of type str. Each position in weights corresponds to the same position in method.
                Each number in weights must be non-negative, and the sum of weights cannot be 0.0 (default: [0.4, 0.3, 0.3])
            threshold {float, None} -- Threshold for determining output keywords, after keyword scores are normalized
                between 0.0 - 1.0. Only keywords with a score >= threshold will be outputted (default: 0.4).
                threshold = None will ignore any score filtering.
            max_return {int} -- Maximum number of keywords to return (default: 7).
            scores {bool} -- Flag to determine if keyword scores should be included in output
            return_type {'list', 'dict'} -- Indicator of output format. return_type = 'dict' will return a dictionary
                of keywords with corresponding scores. return_type = 'list' will return a list of keywords. If 
                return_type = 'list' and scores = True, will return a list of tuples (keyword, score). If 
                return_type = 'list' and scores = False, will return a list of keywords (default: 'list').
        
        Raises
        ------
            ValueError -- If text, method, weights, or return_type are not the appropriate types
            Exception -- If method called is not a supported method (see above for supported methods)

        Returns
        -------
            list, dict -- Variable return type depending on return_type (default: list)
        """
        
        self.errorChecking(text=text, weights=weights)
        edge_cases = self.checkEdgeCases(text=text)
        if edge_cases:
            return self.prepareOutput(keyword_list=edge_cases, max_return=1, scores=scores, return_type=return_type)
        noun_list = []
        if self.remove_websites:
            try:
                text, website_mapping = self.findWebsites(text)
                if website_mapping:
                    noun_list = [x.lower() for x in website_mapping.keys()]
            except:
                pass
        if isinstance(method, str):
            keyword_list = self.createKeywordList(text=text, method=method, noun_list=noun_list)
            if len(keyword_list) > 0:
                #return self.prepareOutput(keyword_list=[])
                keyword_list = self.normalizeKeywordWeights(input_dict=keyword_list)
        elif isinstance(method, list):
            new_weights = []
            keyword_list = []
            for x in range(len(method)):
                method_results = self.createKeywordList(text=text, method=method[x], noun_list=noun_list)
                if method_results:
                    method_results = self.normalizeKeywordWeights(input_dict=method_results)
                    keyword_list.append(method_results)
                    new_weights.append(weights[x])
            
            keyword_list = self.populateWeights(keyword_list)
            keyword_list = self.majorityVoting(input_dict=keyword_list, weights=new_weights)
        else:
            raise ValueError('Method should be of type str or list')
        
        if noun_list:
            for x, y in website_mapping.items():
                keyword_list = {re.sub(r'\b(%s)\b'%x, y, k):v for k, v in keyword_list.items()}
        
        keyword_list = sorted(keyword_list.items(), key=lambda x: x[1], reverse=True)

        return self.prepareOutput(keyword_list=keyword_list, threshold=threshold, max_return=max_return, scores=scores, return_type=return_type)
