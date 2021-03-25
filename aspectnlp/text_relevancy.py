# -*- coding: utf-8 -*-

# Imports
from collections import Counter
from Levenshtein import distance, hamming, jaro, jaro_winkler
import json
import re
import requests
import string


class TextMatcher:
    
    def __init__(self, punctuation=None, authorization_key=None):
        
        if punctuation:
            if isinstance(punctuation, str) == False:
                raise TypeError('punctuation must be of type str')
            self.punctuation_patterns = re.compile(r'[%s]'%punctuation)
        else:
            self.punctuation_patterns = re.compile(r'[%s]'%string.punctuation)
        self.whitespace_patterns = re.compile(r'(^ )|( $)')
        if authorization_key:
            if isinstance(authorization_key, str):
                self.authorization_key = authorization_key
            else:
                raise TypeError('authorization_key must be of type str or None')
        else:
            self.authorization_key = None
        self.web_service_url = 'http://52.151.3.252/api/v1/service/text-relevance/score'
    
    def __error_checking__(self, strings, ignore_case, remove_punctuation, authorization_key=None):
        
        if isinstance(strings[0], str) == False or isinstance(strings[1], str) == False:
            raise TypeError('Inputs should be of type str.')
        if len(strings[0]) == 0 or len(strings[1]) == 0:
            raise ValueError('Inputs cannot be blank.')
        if isinstance(ignore_case, bool) == False:
            raise TypeError('ignore_case must be True or False.')
        if isinstance(remove_punctuation, bool) == False:
            raise TypeError('remove_punctuation must be True or False')
        if authorization_key:
            if isinstance(authorization_key, str) == False:
                raise TypeError('authorization_key must be of type str')

    def __find_common_subsequences__(self, list1, list2):
        
        sequences1 = []
        for i in range(len(list1)+1):
            for j in range(i+1, len(list1)+1):
                sequences1.append(tuple(list1[i:j]))
    
        sequences2 = []
        for i in range(len(list2)+1):
            for j in range(i+1, len(list2)+1):
                sequences2.append(tuple(list2[i:j]))
        
        intersection = set(sequences1) & set(sequences2)
        
        return intersection
            
    def __preprocess__(self, strings, ignore_case=True, remove_punctuation=True):
        
        if ignore_case:
            strings = [x.lower() for x in strings]
        if remove_punctuation:
            strings = [self.punctuation_patterns.sub(' ', x) for x in strings]
        strings = [re.sub(' +', ' ', x) for x in strings]
        strings = [self.whitespace_patterns.sub('', x) for x in strings]       
            
        return strings
    
    def __split_string__(self, input_string):
        
        return input_string.split()

    def char_distance(self, strings, char_type='jaro', normalized=True):
        
        if char_type == 'jaro':
            return jaro(strings[0], strings[1])
        elif char_type == 'jarowinkler':
            return jaro_winkler(strings[0], strings[1])
        elif char_type == 'levenshtein':
            lev = distance(strings[0], strings[1])
            if normalized:
                lev = 1 - lev / max(len(strings[0]), len(strings[1]))
            return lev
        
    def cosine_similarity(self, strings):
        
        fd1 = Counter(self.__split_string__(strings[0]))
        fd2 = Counter(self.__split_string__(strings[1]))
        
        intersection = set(fd1.keys()) & set(fd2.keys())
        intersection_sum = sum(fd1[x] * fd2[x] for x in intersection)
        total1 = sum([fd1[x]**2 for x in fd1.keys()])
        total2 = sum([fd2[x]**2 for x in fd2.keys()])
        divisor = total1**0.5 * total2**0.5
        if intersection_sum == 0.0:
            return 0.0
        else:
            return intersection_sum / divisor
    
    def dice_coefficient(self, strings):

        list1 = self.__split_string__(strings[0])
        list2 = self.__split_string__(strings[1])
        fd1 = Counter(list1)
        fd2 = Counter(list2)
    
        intersection = set(fd1.keys()) & set(fd2.keys())
        if not intersection:
            return 0.0
        overlap = 0
        for key in intersection:
            overlap += min(fd1[key], fd2[key])
        
        dice = 2 * overlap / (len(list1) + len(list2))
        return dice

    def euclidean_distance(self, strings, normalized=True):
        
        fd1 = Counter(self.__split_string__(strings[0]))
        fd2 = Counter(self.__split_string__(strings[1]))
        
        intersection = set(fd1.keys()) & set(fd2.keys())
        if not intersection:
            return 0.0
        sq_diff = sum((fd1[x] - fd2[x])**2 for x in intersection)
        for key in fd1:
            if key not in intersection:
                sq_diff += fd1[key]**2             
        for key in fd2:
            if key not in intersection:
                sq_diff += fd2[key]**2
        
        euclidean = sq_diff ** 0.5
        if normalized:
            euclidean = 1 / (1 + euclidean)
        
        return euclidean

    def jaccard_similarity(self, strings):
        
        set1 = set(self.__split_string__(strings[0]))
        set2 = set(self.__split_string__(strings[1]))
        
        intersection = len(set1 & set2)
        union = len(set1.union(set2))
        
        return intersection / union
    
    def overlap_coefficient(self, strings):
        
        list1 = self.__split_string__(strings[0])
        list2 = self.__split_string__(strings[1])
        
        intersection = self.__find_common_subsequences__(list1=list1, list2=list2)
        
        if not intersection:
            return 0.0
        lcs = len(max(intersection, key=len))
        min_set = min(len(set(list1)), len(set(list2)))
        overlap = lcs / min_set
        
        return overlap        
    
    def evaluate_strings(self, string1, string2, method='jarowinkler', ignore_case=True, remove_punctuation=True):
        
        self.__error_checking__(strings=((string1, string2)), ignore_case=ignore_case, remove_punctuation=remove_punctuation)
        strings = self.__preprocess__(strings=[string1, string2], ignore_case=ignore_case)
        
        if isinstance(method, str):
            method = method.lower()
            if method == 'cosine':
                return self.cosine_similarity(strings=strings)
            elif method == 'dice':
                return self.dice_coefficient(strings=strings)
            elif method == 'euclidean':
                return self.euclidean_distance(strings=strings)
            elif method == 'jaccard':
                return self.jaccard_similarity(strings=strings)
            elif method == 'jaro':
                self.char_distance(strings, char_type='jaro')
            elif method == 'jarowinkler':
                self.char_distance(strings, char_type='jarowinkler')
            elif method == 'levenshtein':
                return self.char_distance(strings, char_type='levenshtein')
            elif method == 'overlap':
                return self.overlap_coefficient(strings)
            elif method == 'chars':
                result1 = self.char_distance(strings, char_type='jarowinkler')
                result2 = self.char_distance(strings, char_type='jaro')
                result3 = self.char_distance(strings, char_type='levenshtein')
                return (result1 + result2 + result3) / 3
            elif method == 'terms':
                result1 = self.cosine_similarity(strings=strings)
                result2 = self.overlap_coefficient(strings)
                result3 = self.jaccard_similarity(strings=strings)
                return (result1 + result2 + result3) / 3
            else:
                raise ValueError("method '%s' does not exist. Please choose from the following methods: \nCosine\nEuclidean\nJaccard\nJaro\nJarowinkler\nLevenshtein\nOverlap\nChars\nTerms" %method)
        else:
            raise TypeError('method should be of type str.')

    def evaluate_strings_all_methods(self, string1, string2, ignore_case=True, remove_punctuation=True, verbose=True):
        
        self.__error_checking__(strings=((string1, string2)), ignore_case=ignore_case, remove_punctuation=remove_punctuation)
        strings = self.__preprocess__(strings=[string1, string2], ignore_case=ignore_case)
        
        results_dict = {}
        results_dict['Cosine Similarity'] = self.cosine_similarity(strings=strings)
        results_dict['Dice Coefficient'] = self.dice_coefficient(strings=strings)
        results_dict['Euclidean Distance'] = self.euclidean_distance(strings=strings)
        results_dict['Jaccard Similarity'] = self.jaccard_similarity(strings=strings)
        results_dict['Jaro'] = self.char_distance(strings=strings, char_type='jaro')
        results_dict['Jaro-Winkler'] = self.char_distance(strings=strings, char_type='jarowinkler')
        results_dict['Levenshtein'] = self.char_distance(strings=strings, char_type='levenshtein')
        results_dict['Overlap Coefficient'] = self.overlap_coefficient(strings=strings)
        results_dict['Char-Based'] = (results_dict['Jaro-Winkler'] + results_dict['Jaro'] + results_dict['Levenshtein'] ) / 3
        results_dict['Term-Based'] = (results_dict['Cosine Similarity'] + results_dict['Overlap Coefficient'] + results_dict['Jaccard Similarity']) / 3
        
        if verbose == True:
            for x, y in enumerate(results_dict):
                print('%s : %.2f' % (y, results_dict[y]))
        
        return results_dict

    def evaluate_strings_character_based(self, string1, string2, ignore_case=True, remove_punctuation=True, verbose=True):
        
        self.__error_checking__(strings=((string1, string2)), ignore_case=ignore_case, remove_punctuation=remove_punctuation)
        strings = self.__preprocess__(strings=[string1, string2], ignore_case=ignore_case)
        
        results_dict = {}
        results_dict['Jaro'] = self.char_distance(strings=strings, char_type='jaro')
        results_dict['Jaro-Winkler'] = self.char_distance(strings=strings, char_type='jarowinkler')
        results_dict['Levenshtein'] = self.char_distance(strings=strings, char_type='levenshtein')
        results_dict['Char-Based'] = (results_dict['Jaro-Winkler'] + results_dict['Jaro'] + results_dict['Levenshtein'] ) / 3
        
        if verbose == True:
            for x, y in enumerate(results_dict):
                print('%s : %.2f' % (y, results_dict[y]))
        
        return results_dict


    def evaluate_strings_term_based(self, string1, string2, ignore_case=True, remove_punctuation=True, verbose=True):
        
        self.__error_checking__(strings=((string1, string2)), ignore_case=ignore_case, remove_punctuation=remove_punctuation)
        strings = self.__preprocess__(strings=[string1, string2], ignore_case=ignore_case)
        
        results_dict = {}
        results_dict['Cosine Similarity'] = self.cosine_similarity(strings=strings)
        results_dict['Dice Coefficient'] = self.dice_coefficient(strings=strings)
        results_dict['Euclidean Distance'] = self.euclidean_distance(strings=strings)
        results_dict['Jaccard Similarity'] = self.jaccard_similarity(strings=strings)
        results_dict['Overlap Coefficient'] = self.overlap_coefficient(strings=strings)
        results_dict['Term-Based'] = (results_dict['Cosine Similarity'] + results_dict['Overlap Coefficient'] + results_dict['Jaccard Similarity']) / 3
        
        if verbose == True:
            for x, y in enumerate(results_dict):
                print('%s : %.2f' % (y, results_dict[y]))
        
        return results_dict


    def evaluate_strings_bert(self, string1, string2, authorization_key=None):

        self.__error_checking__(strings=((string1, string2)), ignore_case=True, remove_punctuation=True, authorization_key=authorization_key)
        data = {'text_a': string1, 'text_b': string2 }
        input_data = json.dumps(data)

        if authorization_key:
            headers =  {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + authorization_key}
        elif self.authorization_key:
            headers =  {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.authorization_key}
        else:
            print('Please provide an authorization_key')
            return None

        try: 
            r = requests.post(self.web_service_url, input_data, headers=headers)
            result = json.loads(r.text) 
        except: 
            raise ValueError('Incorrect output format. Result cant not be parsed: ' + r.text)
        
        out = float(result[0].split()[1])
        return out
