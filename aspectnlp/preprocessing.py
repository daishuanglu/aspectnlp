import csv
import functools
from gensim.parsing.preprocessing import STOPWORDS as gensim_stopwords
import itertools
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import numpy as np
import os
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
import spacy.tokenizer


_CONTRACTIONS_CSV_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/contractions.csv')
_BUSINESS_WORD_CSV_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/business_replacements.csv')
_STOPWORDS_CSV_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/stopwords_english.csv')
_SUPPORTED_STOPWORDS = {'english', 'dutch', 'french', 'german', 'italian', 'japanese', 'spanish', 'keywordextractor'}

nlp = spacy.load("en_core_web_sm")


def expand_contractions(input_text, lookup_dict=None, lookup_dict_csv_path=_CONTRACTIONS_CSV_FILE_PATH):
    """Expands all contractions found in an input string, as defined by a default or custom dictionary. 
    Clitics such as `'s -> is` or `'ll -> will` can also be included.  Capitalization case is ignored.

    For reference, the default contraction dictionary is located at `/rmls/data/contractions.csv`. For custom 
    contraction dictionaries, the user can choose to either pass in a Python dictionary directly or choose a file 
    path string pointing to a CSV with each row representing a `contraction: expansion` entry.
    
    Arguments
    ---------
        input_text {str} -- Input text to have contractions expanded

    Keyword Arguments
    -----------------
        lookup_dict {dict} -- Custom dictionary of contractions to expand
        lookup_dict_csv_path {str} -- File path for uploading a custom dictionary

    Returns
    -------
        str -- New string with all defined contractions and clitics expanded
    
    Raises
    ------
        TypeError -- If input_text is not a string
    """

    if not isinstance(input_text, str):
        raise TypeError('Input "input_text" must be of type str')
    if lookup_dict:
        if not isinstance(lookup_dict, dict):
            raise TypeError('Input "lookup_dict" must be None or of type dict')
    if not lookup_dict:
        with open(lookup_dict_csv_path, 'r', encoding = 'utf-8-sig') as f:
            reader = csv.reader(f)
            try:
                lookup_dict = dict(reader)
            except ValueError as e:
                raise ValueError('The contraction lookup file is not properly formatted.  All rows should contain '
                    'only the contraction and its expansion, separated by a comma.  Make sure there are no blank lines'
                    ' at the end of the file.') from e
    return replace_with_lookup(input_text, lookup_dict=lookup_dict)


def filter_by_pos(token_pos_tuple_list, pos, keep=True):
    """Filters a list of tokenized words according to their part of speech tag.  By default only the words with the 
    selected POS tags will be kept, but can be removed instead by toggling the `keep` parameter to `False`.
    
    Arguments
    ---------
        token_pos_tuple_list {list} -- A list of (token, pos_tag) tuples, such as the output of pos_tag()
        pos {list, str} -- A list of Penn Treebank POS tags to retain (if keep = True) or filter out (if keep = False), or a string containing a single POS tag
    
    Keyword Arguments
    -----------------
        keep {bool} -- Whether all tokens matching `pos` should be retained or filtered out (default: {True})
    
    Returns
    -------
        list -- A list of all tokens (in order) left after filtering by part of speech
    
    Raises
    ------
        TypeError -- If keep is not of type bool, or pos is not of type list or str
    """
    if not isinstance(keep, bool):
        raise TypeError('Attribute "keep" must be of type bool')
    if not isinstance(pos, list) and not isinstance(pos, str):
        raise TypeError('Attribute "pos" must be of type list or str')
    
    pos = [c[0].lower() for c in pos]
    if keep:
        return [word for word, tag in token_pos_tuple_list if tag[0].lower() in pos]
    return [word for word, tag in token_pos_tuple_list if tag[0].lower() not in pos]



def generate_vocab_list(input_text, ignore_case=True, target_column=None):
    """Generates a list of unique terms found in the corpus/text data provided. 
    A user can input a variety of corpus formats, and a list will be returned containing alphabetically sorted terms. rmls_nlp.preprocessing's word_tokenize is used to tokenize terms. If a different tokenization approach is required, it is recommended to first tokenize the data into a list (or a list of lists) and pass this in as input.

    Arguments
    ---------
    input_text {str, list, pd.DataFrame} -- Input text to have vocab list generated from

    Keyword Arguments
    -----------------
    ignore_case {bool} -- A flag indicating whether case should be ignore when compiling the list (default: {True})
    target_column {None, str} -- Used if input_text is a dataframe, otherwise ignored. Points the function to the column within the dataframe to have the vocab list generated from.

    Returns
    -------
    list -- Alphabetically sorted list of unique terms found in the data

    Raises
    ------
    TypeError -- If input_text, target_column, or ignore_case are not of the correct type (see above)
    ValueError -- If input_text is a dataframe and target_column is not a valid column name
    """
    
    if not isinstance(ignore_case, bool):
        raise TypeError('Attribute "ignore_case" must be of type bool')
    
    if isinstance(input_text, str):
        if ignore_case:        
            corpus = word_tokenize(input_text.lower())
        else:
            corpus = word_tokenize(input_text)
        corpus = list(set(corpus))
    elif isinstance(input_text, list):
        corpus = []
        if isinstance(input_text[0], str):
            if ignore_case:
                try:
                    input_text = [x.lower() for x in input_text]
                except:
                    raise TypeError('All items in input "input_text" must be of the same type (str or list)')
            for x in input_text:
                corpus.extend(word_tokenize(x))
            corpus = list(set(corpus))
        elif isinstance(input_text[0], list):
            corpus = []
            if ignore_case:
                for doc in input_text:
                    for text in doc:
                        corpus.extend(word_tokenize(text.lower()))
            else:
                for doc in input_text:
                    for text in doc:
                        corpus.extend(word_tokenize(text))
            corpus = list(set(corpus))
    elif isinstance(input_text, pd.DataFrame):
        corpus = []
        if not isinstance(target_column, str):
            raise TypeError('If "input_text" is a dataframe, target_column must be of type str')
        if target_column not in list(input_text.columns):
            raise ValueError('Attribute "target_column" must be a column in dataframe input_text')
        inp_list = list(input_text[target_column])
        if ignore_case:
            inp_list = [x.lower() for x in inp_list]
        for doc in inp_list:
            corpus.extend(word_tokenize(doc))
        corpus = list(set(corpus))
    else:
        raise TypeError('Input "input_text" must be of type str, list, or pd.DataFrame')
            
    corpus.sort()
    return corpus


def get_default_business_words():
    """Retrieves the list of business acronyms/abbreviations and their full forms used in the replace_business_words
    method. 
    
    Returns
    -------
        dict -- Dictionary with all acronyms/abbreviations and their replacements as key-value pairs
    """

    with open(_BUSINESS_WORD_CSV_FILE_PATH, 'r', encoding = 'utf-8-sig') as f:
        reader = csv.reader(f)
        lookup_dict = dict(reader)
    return lookup_dict


def get_stopwords(stopwords_source=None):
    """Retrieves the list of default stopwords that are removed from an input text in the remove_stopwords() method. 
    
    Keyword Arguments
    -----------------
        stopwords_source {None, list} -- Source for which stopwords are retrieved from (default:{None})
    
    Returns
    -------
        list -- list of default stopwords
        
    """
    if stopwords_source:
        if not isinstance(stopwords_source, str):
            raise TypeError('Attribute "stopwords_source" must be None or of type str')
        stopwords_source = stopwords_source.lower()
        if stopwords_source in _SUPPORTED_STOPWORDS:
            package_path = os.path.dirname(os.path.abspath(__file__))
            relative_csv_path = f'data/stopwords_{stopwords_source}.csv'
            stopwords_source = os.path.join(package_path, relative_csv_path)
            with open(stopwords_source, 'r', encoding = 'utf-8-sig') as f:
                reader = csv.reader(f)
                try:
                    lookup_list = [stopword for sublist in reader for stopword in sublist] # Flatten the nested list of csv.reader(f)
                except ValueError as e:
                    raise ValueError('The stopword lookup file is not properly formatted.  All rows should contain '
                        'exactly one stopword per row.  Make sure there are no blank lines at the end of the file.') from e
        elif stopwords_source == 'aml':
            with open(_STOPWORDS_CSV_FILE_PATH, 'r', encoding = 'utf-8-sig') as f:
                reader = csv.reader(f)
                lookup_list = [stopword for sublist in reader for stopword in sublist] # Flatten the nested list of csv.reader(f)
        elif stopwords_source == 'gensim':
            lookup_list = list(gensim_stopwords)
        elif stopwords_source == 'nltk':
            lookup_list = list(set(nltk_stopwords.words('english')))
        elif stopwords_source == 'spacy':
            lookup_list = list(spacy_stopwords)
        else:
            raise ValueError('Argument "stopwords_source" must be one of: \n"aml", \n"gensim", \n"nltk", \n"spacy", \n"english", \n"dutch", \n"french", \n"german", \n"italian", \n"japanese", \n"spanish",')
    else:
        lookup_list = list(spacy_stopwords)

    # Sort alphabetically
    lookup_list.sort()
    
    return lookup_list


def lemmatize(token_pos_tuples, lemmatizer=nltk.stem.WordNetLemmatizer(), output_type='list'):
    """Lemmatize a list of input words by finding their base forms. POS tags must be provided.
    
    Arguments
    ---------
        token_pos_tuples {list} -- A list of (token, POS tag) tuples
    
    Keyword Arguments
    -----------------
        lemmatizer -- Optional custom lemmatizer (default: {nltk.stem.WordNetLemmatizer()})
        output_type -- Parameter indicating whether output should be of type list or str (default: {'list'})
    
    Raises
    ------
        TypeError -- If token_pos_tuples contains a non-tuple or has invalid tokens or POS tags, or if output_out is not of type str
        ValueError -- If output_type is of type str but is not "list" or "string

    Returns
    -------
        list -- A list of the lemmatized input tokens
        str -- A concatenated list of the lemmatized input tokens
    """

    if not isinstance(output_type, str):
        raise TypeError('Attribute "output_type" must be of type str')
    if output_type.lower() == 'list':
        try:
            return [lemmatize_word(token, pos_tag, lemmatizer) for token, pos_tag in token_pos_tuples]
        except Exception as e:
            if not all(isinstance(token_pos_tuple, tuple) for token_pos_tuple in token_pos_tuples):
                raise TypeError('token_pos_tuples contains a non-tuple but needs to be a list of tuples') from e
    elif output_type.lower() == 'string' or output_type.lower() == 'str':
        try:
            lemm_list = [lemmatize_word(token, pos_tag, lemmatizer) for token, pos_tag in token_pos_tuples]
            return ' '.join(x for x in lemm_list)
        except Exception as e:
            if not all(isinstance(token_pos_tuple, tuple) for token_pos_tuple in token_pos_tuples):
                raise TypeError('token_pos_tuples contains a non-tuple but needs to be a list of tuples') from e
    else:
        raise ValueError('Attribute "output_type" must be one of: "list", "string"')


def lemmatize_word(input_word, pos_tag='NN', lemmatizer=nltk.stem.WordNetLemmatizer()):
    """Lemmatize an input word by finding its base form.  POS tags should be provided if possible. 
    
    Arguments
    ---------
        input_word {str} -- Input word to lemmatize
    
    Keyword Arguments
    -----------------
        pos_tag {str} -- Penn Treebank part-of-speech tag that assists in lemmatization  (default: {'NN'})
        lemmatizer -- Optional custom lemmatizer (default: {nltk.stem.WordNetLemmatizer()})
    
    Raises
    ------
        TypeError -- If input_text or pos_tag are not strings, or lemmatizer does not implement a lemmatize() method
    
    Returns
    -------
        str -- Lemmatized form of input_word.  The original word is returned if it is not found in WordNet (default)
    """
    if not isinstance(input_word, str):
        raise TypeError('Input "input_word" must be of type str')
    if not isinstance(pos_tag, str):
        raise TypeError('Input "pos_tag" type must be of type str')

    try:
        # Convert Penn Treebank POS tags to WordNet POS tags for the default lemmatizer to use
        if isinstance(lemmatizer, nltk.stem.wordnet.WordNetLemmatizer):
            if pos_tag[0].lower() == 'j':
                pos_tag = 'a' 
            pos_tag = pos_tag[0].lower() if pos_tag[0].lower() in ['v', 'r'] else 'n'
        return lemmatizer.lemmatize(input_word, pos_tag)
    except AttributeError:
        raise TypeError('Attribute "lemmatizer" must implement a lemmatize() method')


def lowercase(input_text):
    """Converts all characters/strings in a string, list or other iterable to lower case.
    
    Arguments
    ---------
        input_text {str, list} -- Input string or list, set, tuple, etc. of string-only elements
    
    Returns
    -------
        str, list -- New string/iterable with all strings converted to lower case
    
    Raises
    ------
        TypeError -- If input_text is not of type list or str
    """

    if isinstance(input_text, str):
        return input_text.lower()
    elif isinstance(input_text, list):
        return [x.lower() for x in input_text]
    else:
        raise TypeError('Input "input_text" must be of type str or list')


def pos_tag(token_list, pos_tagger='spacy'):
    """Assigns a Penn Treebank part-of-speech tag to each word-level token in a list.  Uses spaCy's tagger from the
    en_core_web_sm model by default.
    
    Arguments
    ---------
        token_list {list} -- List of word-level tokens
    
    Keyword Arguments
    -----------------
        pos_tagger -- Optional alternative POS tagger. (default: {'spacy'})
            if 'spacy', uses SpaCy's implementation (see above)
            if 'nltk', uses nltk's Perceptron tagger
            otherwise, attempts to tag using a custom tagger

    Raises
    ------
        TypeError -- If pos_tagger is not 'spacy' or 'nltk' and does not implement a tag() method.
        TypeError -- If token_list is not of type list, or contains non-str items
    
    Returns
    -------
        list -- New list of (token, pos_tag) tuples for each token in token_list
    """

    if not isinstance(token_list, list):
        raise TypeError('Input "token_list" must be of type list')
    if isinstance(pos_tagger, str):
        if pos_tagger.lower() == 'spacy':
            try:
                return [(token.text, token.tag_) for token in nlp(' '.join(token_list))]
            except:
                raise TypeError('All items in input "token_list" must be of type str')
        elif pos_tagger == 'nltk':
            try:
                return nltk.PerceptronTagger().tag(token_list)
            except:
                raise TypeError('All items in input "token_list" must be of type str')
        else:
            raise TypeError('Attribute "pos_tagger" must be one of "spacy", "nltk", or a custom tagger')
    else:
        try:
            return pos_tagger.tag(token_list)
        except AttributeError as e:
            raise TypeError('pos_tagger type is {0!r} but needs to implement str or nltk.tag.api.TaggerI'.format(
                pos_tagger.__class__.__name__))	from e


def remove_numbers(input_text, remove_all=False):
    """Removes numbers from an input text or list. User has the option of removing all numbers, or just standalone numbers.

    Arguments
    ---------
        input_text {str, list} -- Input string or list of string-only elements

    Keyword Arguments
    -----------------
        remove_all {bool} -- Flag indicating whether all numbers should be removed (default: {False})
            if True, all numbers are removed
            if False, only numbers that are not attached to other characters are removed

    Raises
    ------
        TypeError -- If input_text is not of type list/str, or remove_all is not of type bool

    Returns
    -------
        str, list -- New string/iterable with numbers removed
    """
    if not isinstance(remove_all, bool):
        raise TypeError('Attribute "remove_all" must be of type bool')
        
    if isinstance(input_text, list):
        if remove_all:
            input_text = [re.sub(r'\d', '', x) for x in input_text]
            return [x for x in input_text if x != '']
        else:
            return [x for x in input_text if not x.isnumeric()]
    elif isinstance(input_text, str):
        if remove_all:
            input_text = re.sub(r'\d', ' ', input_text)
        else:
            input_text = re.sub(r'\b\d+\b', ' ', input_text)
        input_text = remove_whitespace(input_text)
        return input_text
    else:
        raise TypeError('Input "input_text" must be of type list or str')


def remove_punctuation(input_text, lookup_list=list(string.punctuation), replacement=' ', keep_url=False):
    """Removes all punctuation from a string or list of strings, as defined by string.punctuation or a custom list.
    URLs and email addresses will be kept intact by default, but can also be removed completely.  
    
    For reference, Python's string.punctuation is equivalent to ``'!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~'```
    
    Arguments
    ---------
        input_text {str, list} -- Input text to have punctuation removed from
    
    Keyword Arguments
    -----------------
        lookup_list {list} -- Custom list of punctuation to remove (default: {list(string.punctuation)})
        replacement {str} -- Replacement value for punctuation (default: {' '})
        keep_url {bool} -- Whether to keep URLs and their punctuation intact in the input list (default: {True})
    
    Returns
    -------
        str -- New string with all punctuation removed or a new list with punctuation-only elements removed

    Raises
    ------
    TypeError -- If input_text, lookup_list, replacement, or keep_url are not of the correct type
    """

    if not isinstance(lookup_list, list):
        raise TypeError('Attribute "lookup_list" must be of type list')
    if not isinstance(replacement, str):
        raise TypeError('Attribute "replacement" must be of type str')
    if not isinstance(keep_url, bool):
        raise TypeError('Attribute "keep_url" must be of type bool')

    lookup_list = [re.escape(x) for x in lookup_list]
    lookup_dict = dict(zip(lookup_list, itertools.repeat(replacement))) # Map all punctuation to the empty string
    
    if isinstance(input_text, str):
        input_text = replace_with_lookup(input_text, lookup_dict=lookup_dict, exact_match=False)
        if replacement != '':
            input_text = remove_whitespace(input_text)
        return input_text
    elif isinstance(input_text, list):
        non_punct_tokens = [] # Avoid modifying the input token list
        url_regex = get_url_regex()
        for token in input_text:
            if keep_url and re.fullmatch(url_regex, token):
                non_punct_tokens.append(token)
            else:
                non_punct_tokens.append(replace_with_lookup(token, lookup_dict = lookup_dict, exact_match = True))
        return [token for token in non_punct_tokens if token != ''] # Remove empty list elements
    else:
        raise TypeError('Input "input_text" must be of type str or list')


def remove_stopwords(input_text, stopwords_source=None, lookup_list=None, ignore_case=True, strip_whitespace=True):
    """Removes all stopwords from either a pre-tokenized text list or a string.

    For custom stopword dictionaries, the user can choose to either pass in a Python list directly or choose a file 
    path string pointing to a CSV with one stopword per row.  Non-English stopword lists are also available.
    
    Arguments
    ---------
        input_text {str, list} -- Input text to have stopwords removed from
    
    Keyword Arguments
    -----------------
        stopwords_source {None, str} -- File path for uploading a custom stopword list (default: {None})
        lookup_list {None, list} -- Custom list of stopwords to remove (default: {None})
        ignore_case {bool} -- Flag to replace the input pattern regardless of capitalization (default: {True})
        strip_whitespace {bool} -- Flag to remove whitespace in addition to stopwords (default: {True})
    
    Returns
    -------
        str -- Original text with stopwords removed (if input_text is of type str)
        list -- New tokenized list with all matching stopwords removed (if input_text is of type list)

    Raises
    ------
    TypeError -- If input_text, lookup_list, ignore_case, or strip_whitespace is not of the correct type
    """
    if not isinstance(strip_whitespace, bool):
        raise TypeError('Attribute "strip_whitespace" must be of type bool')
    if lookup_list:
        if not isinstance(lookup_list, list):
            raise TypeError('Attribute "lookup_list" must be None or of type list')
    else:
        lookup_list = get_stopwords(stopwords_source=stopwords_source)
    if isinstance(input_text, list):
        if ignore_case:
            return [token for token in input_text if token.lower() not in lookup_list]
        else:
            return [token for token in input_text if token not in lookup_list]
    elif isinstance(input_text, str):
        for term in lookup_list:
            input_text = re.sub(r'\b%s\b'%term, ' ', input_text, flags=re.IGNORECASE if ignore_case else False)
        if strip_whitespace:
            input_text = remove_whitespace(input_text)
        return input_text
    else:
        raise TypeError('Attribute "input_text" must be of type list or str'.format(input_text.__class__.__name__))


def remove_whitespace(input_text, remove_duplicate_whitespace=True):
    """Removes leading, trailing, and (optionally) duplicated whitespace.
    
    Arguments
    ---------
        input_text {str} -- Input string (word, sentence, corpus, etc.)
    
    Keyword Arguments
    -----------------
        remove_duplicate_whitespace {bool} -- Flag to remove duplicate whitespace characters (default: {True})
    
    Raises
    ------
        TypeError -- If input_text is not a string-like type or remove_duplicate_whitespace is not a Boolean
    
    Returns
    -------
        str -- New string with input text with the specified whitespace removed
    """
    if not isinstance(remove_duplicate_whitespace, bool):
        raise TypeError('Attribute remove_duplicate_whitespace must be of type bool')
    try:
        if remove_duplicate_whitespace:
            return " ".join(re.split(r'\s+', input_text.strip(), flags=re.UNICODE))
        return input_text.strip()
    except AttributeError as e:
        raise TypeError("input_text type is {0!r} but needs to be a string".format(
            input_text.__class__.__name__)) from e


def replace_business_words(input_text, lookup_dict_csv_path=_BUSINESS_WORD_CSV_FILE_PATH, lookup_dict=None, 
                           ignore_case=True):
    """Replaces all business-specific acronyms and abbreviations (such as as msft -> Microsoft or sfb->Skype for 
    Business) from an input text.  Any acronyms or abbreviations can be replaced by using a custom lookup 
    dictionary, but the default replacements are business-focused.

    For custom replacement dictionaries, the user can choose to either pass in a Python dictionary directly or choose a
    file path string pointing to a CSV with each row representing a `business word: replacement` entry.
    
    Arguments
    ---------
        input_text {str} -- Input string (word, sentence, corpus, etc.)
    
    Keyword Arguments
    -----------------
        lookup_dict_path {str} -- File path for uploading a custom stopword list
        lookup_dict {list} -- Custom dictionary of {business word : replacement} key-value pairs (default: {None})
        ignore_case {bool} -- Flag to replace the input pattern regardless of capitalization (default: {True})
    
    Returns
    -------
        str -- New string with all business words replaced
    """
    if not lookup_dict:
        with open(lookup_dict_csv_path, 'r', encoding = 'utf-8-sig') as f:
            reader = csv.reader(f)
            try:
                lookup_dict = dict(reader)
            except ValueError as e:
                raise ValueError('The business word lookup file is not properly formatted.  All rows should contain '
                    'only the business acronym/abbreviation and its replacement, separated by a comma.  Make sure '
                    'there are no blank lines at the end of the file.') from e
    regex_lookup_dict = { rf'\b{key}\b' : value for key, value in lookup_dict.items() }

    return replace_with_lookup(input_text, lookup_dict=regex_lookup_dict, ignore_case=ignore_case, exact_match=False)


def replace_with_lookup(input_text, lookup_dict, ignore_case=True, exact_match=True):
    """Replaces all instances of a regex pattern or substring with a given replacement string, in the order provided.
    
    Arguments
    ---------
        input_text {str} -- Input string (word, sentence, corpus, etc.)
        lookup_dict {dict} -- Dictionary of {input_pattern : output_string} key-value (regex/string) pairs to replace
    
    Keyword Arguments
    -----------------
        ignore_case {bool} -- Flag to replace the input pattern regardless of capitalization (default: {True})
        exact_match {bool} -- Whether lookup_dict input patterns replace exact matches (default: {True})
    
    Raises
    ------
        TypeError -- If input_text, lookup_dict, ignore_case, or exact_match are not the appropriate types
    
    Returns
    -------
        str -- New string with all input patterns of input_text replaced with their corresponding output strings
    """
    try:
        if exact_match:
            for input_pattern, output_string in lookup_dict.items():
                input_text = re.sub(r'\b%s\b'%input_pattern, output_string, input_text, flags=re.IGNORECASE if ignore_case else False)
        else:
            for input_pattern, output_string in lookup_dict.items():
                input_text = re.sub(input_pattern, output_string, input_text, flags=re.IGNORECASE if ignore_case else False)
        return input_text
    except TypeError as e:
        raise TypeError("input_text type is {0!r} but needs to be a string".format(
            input_text.__class__.__name__)) from e
    except AttributeError as e:
        raise TypeError("lookup_dict type is {0!r} but needs to be a dict".format(
            lookup_dict.__class__.__name__)) from e


def tfidf_vectorize(corpus, target_column=None, stopwords=None, ignore_case=True, min_freq=1, max_freq=1.0, max_terms=None, return_vectorizer=False):

    if stopwords:
        if isinstance(stopwords, set):
            stopwords = list(stopwords)
        elif not isinstance(stopwords, list):
            raise TypeError('Attribute "stopwords" must be of type list or set')
    if not isinstance(ignore_case, bool):
        raise TypeError('Attribute ignore_case must be of type bool')
    if not isinstance(return_vectorizer, bool):
        raise TypeError('Attribute return_vectorizer must be of type bool')
    if isinstance(min_freq, int):
        if min_freq < 0:
            raise ValueError('Attribute "min_freq" must be positive')
    elif isinstance(min_freq, float):
        if min_freq < 0.0 or min_freq > 1.0:
            raise ValueError('Attribute "min_freq" must be a value in the range [0.0, 1.0]')
    else:
        raise TypeError('Attribute "min_freq" must be of type int or float')
    if isinstance(max_freq, int):
        if max_freq < 0:
            raise ValueError('Attribute "max_freq" must be positive')
    elif isinstance(max_freq, float):
        if max_freq < 0.0 or max_freq > 1.0:
            raise ValueError('Attribute "max_freq" must be a value in the range [0.0, 1.0]')
    else:
        raise TypeError('Attribute "max_freq" must be of type int or float')
    if type(min_freq) == type(max_freq):
        if max_freq < min_freq:
            raise ValueError('Attribute "min_freq" cannot be larger than "max_freq"')
    
    countvec = TfidfVectorizer(lowercase=ignore_case, 
                               stop_words=stopwords, 
                               min_df=min_freq, 
                               max_df=max_freq, 
                               max_features=max_terms)
    
    if isinstance(corpus, pd.DataFrame):
        if not isinstance(target_column, str):
            raise TypeError('If corpus is a dataframe, target_column must be of type str')
        if target_column not in list(corpus.columns):
            raise ValueError('Attribute target_column must be a valid column name in corpus')
        transformed = countvec.fit_transform(corpus[target_column])
    elif isinstance(corpus, list):
        if not all(isinstance(x, str) for x in corpus):
            raise TypeError('If corpus is a list, all items must be of type str')
        transformed = countvec.fit_transform(corpus)
    
    transformed = transformed.toarray()
    vocab = countvec.get_feature_names()
    
    if return_vectorizer:
        return transformed, vocab, countvec
    else:
        return transformed, vocab


def sent_tokenize(input_text, tokenizer=nltk.tokenize.PunktSentenceTokenizer()):
    """Converts a string or text into a list of sentence tokens, with the option to choose a custom tokenizer.
    
    Arguments
    ---------
        input_text {str} -- Input string (sentence, paragraph, corpus, etc.) to tokenize
    
    Keyword Arguments
    -----------------
        tokenizer -- Object that implements tokenize() to define token boundaries (default: {PunktSentenceTokenizer()})
    
    Raises
    ------
        TypeError -- If input_text is not a string or tokenizer does not implement a tokenize() method
    
    Returns
    -------
        list -- New list of sentence strings created from tokenizing the input text
    """
    if isinstance(tokenizer, str):
        tokenizer = nltk.RegexpTokenizer(tokenizer)
    try:
        return tokenizer.tokenize(input_text)
    except TypeError as e:
        raise TypeError('input_text type is {0!r} but needs to be a string'.format(
            input_text.__class__.__name__)) from e
    except AttributeError as e:
        raise TypeError('tokenizer type is {0!r} but needs to implement nltk.tokenize.api.TokenizerI'.format(
            tokenizer.__class__.__name__)) from e


def stem(input_text, stemmer='porter', language='english'):
    """Stem a list of words by removing any inflected endings.
    
    Arguments
    ---------
        stemmer -- Type of stemmer to be used (default: 'porter')
        language -- Language to be used for stemmer (default: 'english') 
    
    Raises
    ------
        TypeError -- If input_text is not a string or list or stemmer does not implement a stem() method

    Returns
    -------
        list -- List of the stemmed form of each input word (if input is of type list)
        str -- String of the stemmed form of each input word (if input is of type str)
    """

    if isinstance(language, str):
        language = language.lower()
        if language not in ['danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 'italian', 'norwegian', 'portuguese', 'romanian', 'russian', 'spanish', 'swedish']:
            raise ValueError("Attribute 'language' must be one of: 'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 'italian', 'norwegian', 'portuguese', 'romanian', 'russian', 'spanish', 'swedish'")
    else:
        raise TypeError('Attribute "language" must be of type str')
    
    if not isinstance(stemmer, str):
        raise TypeError('Attribute "stemmer" must be of type str')
    else:
        if stemmer.lower() == 'porter':
            stemmer = PorterStemmer()
        elif stemmer.lower() == 'snowball':
            stemmer = SnowballStemmer(language)
        else:
            raise ValueError('Attribute "stemmer" must be one of: "porter", "snowball"')
    
    if isinstance(input_text, str):
        return stemmer.stem(input_text)
    elif isinstance(input_text, list):
        return [stemmer.stem(x) for x in input_text]
    else:
        raise TypeError('Input "input_text" must be of type list or str')



def stem_word(input_word, stemmer='porter', language='english'):
    """Stem an input word by removing any inflected endings.
    
    Arguments
    ---------
        input_word {str} -- Input word to stem
    
    Arguments
    -----------------
        stemmer -- Type of stemmer to be used (default: 'porter')
        language -- Language to be used for stemmer (default: 'english')
    
    Raises
    ------
        TypeError -- If input_word is not a string or stemmer does not implement a stem() method
    
    Returns
    -------
        str -- New string of the stemmed form of the input word
    """

    if isinstance(language, str):
        language = language.lower()
        if language not in ['danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 'italian', 'norwegian', 'portuguese', 'romanian', 'russian', 'spanish', 'swedish']:
            raise ValueError("Attribute 'language' must be one of: 'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 'italian', 'norwegian', 'portuguese', 'romanian', 'russian', 'spanish', 'swedish'")
    else:
        raise TypeError('Attribute "language" must be of type str')
    
    if not isinstance(stemmer, str):
        raise TypeError('Attribute "stemmer" must be of type str')
    else:
        if stemmer.lower() == 'porter':
            stemmer = PorterStemmer()
        elif stemmer.lower() == 'snowball':
            stemmer = SnowballStemmer(language)
        else:
            raise ValueError('Attribute "stemmer" must be one of: "porter", "snowball"')
    
    try:
        return stemmer.stem(input_word)
    except TypeError as e:
        raise TypeError('input_word type is {0!r} but needs to be a string'.format(
            input_word.__class__.__name__)) from e


def uppercase(input_text, title=False):
    """Converts all characters/strings in a string, list or other iterable to upper case.
    
    Arguments:
        input_text {str, list} -- Input string or list, set, tuple, etc. of string-only elements
    
    Keyword Arguments
    -----------------
        title {bool} -- Flag indicating whether title option should be used (default: {False})
    
    Raises
    ------
        TypeError -- If input_text is not of type str/list or title is not None and is not of type bool

    Returns
    -------
        str, list -- New string/iterable with all strings converted to upper case
    """
    if not isinstance(title, bool):
        raise TypeError('Input "title" must be of type bool')
    if isinstance(input_text, str):
        if title:
            return input_text.title()
        else:
            return input_text.upper()
    elif isinstance(input_text, list):
        if title:
            return [x.title() for x in input_text]
        else:
            return [x.upper() for x in input_text]
    else:
        raise TypeError('Input "input_text" must be of type str or list')


def vectorize(corpus, target_column=None, stopwords=None, ignore_case=True, min_freq=1, max_freq=1.0, max_terms=None, return_vectorizer=False):
    """ Vectorizes a text dataset (corpus) into a bag-of-words format, showing counts of unique terms within the corpus. 
    The user has control over a number of arguments related to how the vectorization is conducted. For more information about the vectorizer that may be returned, see sklearn's documentation on CountVectorizer.

    Arguments
    ---------
    input_text {list, pd.DataFrame} -- Input corpus to vectorize, either as a list of strings or a dataframe containing a column of strings

    Keyword Arguments
    -----------------
    target_column {None, str} -- Used if corpus is a dataframe, otherwise ignored. Points the function to the column within the dataframe to vectorize (default: {None})
    stopwords {None, list, set} -- A list of stopwords to use to ignore in the vectorization process (default: {None})
    ignore_case {bool} -- A flag indicating whether case should be ignored during vectorization
    min_freq {float, int} -- The lower-bound number of appearances for a term to be included in the result (default: {1})
        if int, this is the absolute value lower-bound
        if float, this is the percentage of documents that must contain the term for it to be included (must be in the range [0.0, 1.0])
    max_freq {float, int} -- The upper-bound number of appearances for a term to be included in the result (default: {1.0})
        if int, this is the absolute value upper-bound
        if float, this is the max percentage of documents that can contain the term for it to be included (must be in the range [0.0, 1.0])
    max_terms {None, int} -- If this is set to an int, will only include up to that number of terms
    return_vectorizer {bool} -- Flag indicating whether the trained vectorizer (sklearn.CountVectorizer) should be returned

    Returns
    -------
    array -- Vectorized array, of size (# docs, # terms in vocab)
    list -- List of vocabulary terms corresponding to columns in above array
    CountVectorizer -- sklearn.feature_extraction.text.CountVectorizer, the trained vectorizer (only returned if return_vectorizer=True)

    Raises
    ------
    TypeError -- corpus, target_column, stopwords, ignore_case, min_freq, max_freq, max_terms, or return_vectorizer is not of the correct type (see above)
    ValueError -- If corpus is a dataframe and target_column is not a valid column name
    """

    if stopwords:
        if isinstance(stopwords, set):
            stopwords = list(stopwords)
        elif not isinstance(stopwords, list):
            raise TypeError('Attribute "stopwords" must be of type list or set')
    if not isinstance(ignore_case, bool):
        raise TypeError('Attribute ignore_case must be of type bool')
    if not isinstance(return_vectorizer, bool):
        raise TypeError('Attribute return_vectorizer must be of type bool')
    if isinstance(min_freq, int):
        if min_freq < 0:
            raise ValueError('Attribute "min_freq" must be positive')
    elif isinstance(min_freq, float):
        if min_freq < 0.0 or min_freq > 1.0:
            raise ValueError('Attribute "min_freq" must be a value in the range [0.0, 1.0]')
    else:
        raise TypeError('Attribute "min_freq" must be of type int or float')
    if isinstance(max_freq, int):
        if max_freq < 0:
            raise ValueError('Attribute "max_freq" must be positive')
    elif isinstance(max_freq, float):
        if max_freq < 0.0 or max_freq > 1.0:
            raise ValueError('Attribute "max_freq" must be a value in the range [0.0, 1.0]')
    else:
        raise TypeError('Attribute "max_freq" must be of type int or float')
    if type(min_freq) == type(max_freq):
        if max_freq < min_freq:
            raise ValueError('Attribute "min_freq" cannot be larger than "max_freq"')
    
    countvec = CountVectorizer(lowercase=ignore_case, 
                               stop_words=stopwords, 
                               min_df=min_freq, 
                               max_df=max_freq, 
                               max_features=max_terms)
    
    if isinstance(corpus, pd.DataFrame):
        if not isinstance(target_column, str):
            raise TypeError('If corpus is a dataframe, target_column must be of type str')
        if target_column not in list(corpus.columns):
            raise ValueError('Attribute target_column must be a valid column name in corpus')
        transformed = countvec.fit_transform(corpus[target_column])
    elif isinstance(corpus, list):
        if not all(isinstance(x, str) for x in corpus):
            raise TypeError('If corpus is a list, all items must be of type str')
        transformed = countvec.fit_transform(corpus)
    
    transformed = transformed.toarray()
    vocab = countvec.get_feature_names()
    
    if return_vectorizer:
        return transformed, vocab, countvec
    else:
        return transformed, vocab


def word_tokenize(input_text, tokenizer="[\w']+"):
    """Converts a string or text into a list of word tokens, with the option to choose a custom tokenizer.

    The default tokenizer is a regex pattern using nltk's RegexpTokenizer.
    Another option is to use spaCy's tokenizer from the en_core_web_sm model.
    
    Arguments
    ---------
        input_text {str} -- Input string (word, sentence, corpus, etc.) to tokenize
    
    Keyword Arguments
    -----------------
        tokenizer -- Optional regex pattern or object that implements tokenize() to define token boundaries (default: {"[\w']+"})
    
    Raises
    ------
        TypeError -- If input_text is not a string or tokenizer does not implement a tokenize() method
    
    Returns
    -------
        list -- New list of word and punctuation strings created from tokenizing the input text
    """
    try:
        if tokenizer == 'spacy':
            tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab)
            return [token.text for token in tokenizer(input_text)]
        elif isinstance(tokenizer, str):
            tokenizer = RegexpTokenizer(tokenizer)
        return tokenizer.tokenize(input_text)
    except TypeError as e:
        raise TypeError('input_text type is {0!r} but needs to be a string'.format(
            input_text.__class__.__name__)) from e
    except AttributeError as e:
        raise TypeError('tokenizer type is {0!r} but needs to implement nltk.tokenize.api.TokenizerI'.format(
            tokenizer.__class__.__name__)) from e


def custom_preprocess(input_text, steps=[remove_whitespace, expand_contractions, replace_business_words, 
    remove_numbers, lowercase, word_tokenize, remove_punctuation, remove_stopwords, pos_tag, lemmatize], output_type='string'):
    """Executes a series of preprocessing functions consecutively on an input text.  The choice of preprocessing 
    functions and their execution order can be modified as desired.  To use non-default arguments in each step, the
    use of functools.partial() to create a single-parameter replacement step is strongly recommended.  See the 
    example below for usage.
    
    Arguments
    ---------
        input_text {str, list} -- Input text/corpus, but can be a tokenized list if starting with e.g. stem() or pos_tag()
    
    Keyword Arguments
    -----------------
        steps {list} -- A list of function names to be called consecutively on input_text 
            (default: {[remove_whitespace, expand_contractions, replace_business_words, remove_numbers, lowercase, 
            word_tokenize, remove_punctuation, remove_stopwords, pos_tag, lemmatize]})
    
    Returns
    -------
        str, list -- Return type depends on the last element of steps.  Lemmatized token list by default

    Examples
    --------
    Define a custom stopword dictionary inline with functools.partial(<function>, <non-default argument>)
    >>> from functools import *
    >>> custom_preprocess('  hello; world!!', steps = [remove_whitespace, expand_contractions, word_tokenize, remove_punctuation, 
    ...     partial(remove_stopwords, lookup_list = ['hello'])])
        ['world']
    """

    if not isinstance(output_type, str):
        raise TypeError('Attribute "output_type" must be of type str')
    if not isinstance(steps, list):
        raise TypeError('Attribute "steps" must be of type list')
    if isinstance(input_text, list):
        try:
            input_text = ' '.join(x for x in input_text)
        except:
            raise TypeError('All items in input "input_text" must be of type str')
    elif not isinstance(input_text, str):
        raise TypeError('Input "input_text" must be of type list or str')

    input_deep_copy = input_text[:]
    for step in steps:
        input_deep_copy = step(input_deep_copy)
    
    if output_type.lower() == 'string' or output_type.lower() == 'str':
        if isinstance(input_deep_copy, str):
            return input_deep_copy
        else:
            return ' '.join(x for x in input_deep_copy)
    elif output_type.lower() == 'list':
        if isinstance(input_deep_copy, str):
            return word_tokenize(input_deep_copy)
        else:
            return input_deep_copy
    else:
        raise ValueError('Attribute "output_type" must be one of: "string", "list"')


def preprocess_dataframe(input_df, target_column, steps=[remove_whitespace, expand_contractions, replace_business_words, 
    remove_numbers, lowercase, word_tokenize, remove_punctuation, remove_stopwords, pos_tag, lemmatize]):
    """Applies a preprocessing pipeline to all of the texts in a column of a dataframe. 
    The user passes in a dataframe and a selected column, and the original dataframe is returned with an additional column with the preprocessed version of the target column. 
    Preprocessing steps are set by default, but a user has control over the steps taken in the preprocessing if they choose.

    Arguments
    ---------
    input_df {pd.DataFrame} -- Input dataframe to preprocess
    target_column {str} -- Name of the column within input_df to apply preprocessing steps to

    Keyword Arguments
    -----------------
    steps {list} -- Ordered steps used for preprocessing (default: {[remove_whitespace, expand_contractions, replace_business_words, remove_numbers, lowercase, word_tokenize, remove_punctuation, remove_stopwords, pos_tag, lemmatize]})

    Returns
    -------
    pd.DataFrame -- Original dataframe with an additional column containing the preprocessed texts

    Raises
    ------
    TypeError -- If input_df, target_column, or steps is not of the correct type
    ValueError -- If target_column is not a valid column name or an item in steps is not a valid preprocessing step
    """

    if not isinstance(input_df, pd.DataFrame):
        raise TypeError('Input "input_df" must be of type pd.DataFrame')
    if not isinstance(target_column, str):
        raise TypeError('Input "target_column" must be of type str')
    if target_column not in list(input_df.columns):
        raise ValueError('target_column must be a valid column in input_df')
    if not isinstance(steps, list):
        raise TypeError('Attribute "steps" must be of type list')

    if expand_contractions in steps:
        with open(_CONTRACTIONS_CSV_FILE_PATH, 'r', encoding = 'utf-8-sig') as f:
            reader = csv.reader(f)
            try:
                contractions_dict = dict(reader)
            except ValueError as e:
                raise ValueError('The contraction lookup file is not properly formatted.  All rows should contain '
                    'only the contraction and its expansion, separated by a comma.  Make sure there are no blank lines'
                    ' at the end of the file.') from e
    if replace_business_words in steps:
        with open(_BUSINESS_WORD_CSV_FILE_PATH, 'r', encoding = 'utf-8-sig') as f:
            reader = csv.reader(f)
            try:
                business_list = dict(reader)
            except ValueError as e:
                raise ValueError('The business word lookup file is not properly formatted.  All rows should contain '
                    'only the business acronym/abbreviation and its replacement, separated by a comma.  Make sure '
                    'there are no blank lines at the end of the file.') from e
        business_dict = { rf'\b{key}\b' : value for key, value in business_list.items() }
    if remove_stopwords in steps:
        stopwords_list = get_stopwords()
    
    list_type = False
    input_df[target_column+'_preprocessed'] = input_df[target_column].apply(lambda x: str(x))
    for step in steps:
        if step == remove_whitespace:
            input_df[target_column+'_preprocessed'] = input_df[target_column+'_preprocessed'].apply(lambda x: remove_whitespace(x))
        elif step == expand_contractions:
            input_df[target_column+'_preprocessed'] = input_df[target_column+'_preprocessed'].apply(lambda x: replace_with_lookup(x, lookup_dict=contractions_dict))
        elif step == replace_business_words:
            input_df[target_column+'_preprocessed'] = input_df[target_column+'_preprocessed'].apply(lambda x: replace_with_lookup(x, lookup_dict=business_dict, exact_match=False))
        elif step == remove_numbers:
            input_df[target_column+'_preprocessed'] = input_df[target_column+'_preprocessed'].apply(lambda x: remove_numbers(x))
        elif step == lowercase:
            input_df[target_column+'_preprocessed'] = input_df[target_column+'_preprocessed'].apply(lambda x: x.lower())
        elif step == uppercase:
            input_df[target_column+'_preprocessed'] = input_df[target_column+'_preprocessed'].apply(lambda x: x.upper())
        elif step == word_tokenize:
            input_df[target_column+'_preprocessed'] = input_df[target_column+'_preprocessed'].apply(lambda x: word_tokenize(x))
            list_type=True
        elif step == remove_punctuation:
            input_df[target_column+'_preprocessed'] = input_df[target_column+'_preprocessed'].apply(lambda x: remove_punctuation(x))
        elif step == remove_stopwords:
            input_df[target_column+'_preprocessed'] = input_df[target_column+'_preprocessed'].apply(lambda x: remove_stopwords(x, lookup_list=stopwords_list))
        elif step == pos_tag:
            input_df[target_column+'_preprocessed'] = input_df[target_column+'_preprocessed'].apply(lambda x: pos_tag(x))
        elif step == lemmatize:
            input_df[target_column+'_preprocessed'] = input_df[target_column+'_preprocessed'].apply(lambda x: lemmatize(x))
            list_type=True
        elif step == stem:
            input_df[target_column+'_preprocessed'] = input_df[target_column+'_preprocessed'].apply(lambda x: stem(x))
        else:
            raise ValueError('Step ' + step + ' is not a valid preprocessing step')
        
    if list_type:
        input_df[target_column+'_preprocessed'] = input_df[target_column+'_preprocessed'].apply(lambda x: ' '.join(z for z in x))

    return input_df

###################################


def get_url_regex():
    """Creates a regular expression object that matches most URLs.
    
    Returns
    -------
        re.Pattern -- Regular expression object that can be used as a pattern with the re package to match, search, etc.
    """

    valid_chars = r"[\w\[\].*+!?@#$%&()-=:;,'/]" # \w character class, square brackets, and other valid characters
    url_starts = r'(https?:|www\.|aka\.m)' # Can add option to require
    domain_exts = r'(\.(com|net|org|co|us|uk|de|info))'

    url_pattern = rf'({url_starts}?{valid_chars}+{domain_exts}{valid_chars})|({url_starts}{valid_chars}+)*'
    return re.compile(url_pattern)
