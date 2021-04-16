#standard librarires
import re
import os
import time
import csv
import requests
from fasttext import load_model


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def keep_alphanumeric_symb(s):
    return re.sub(r"[^A-Za-z0-9/.' ]+", ' ', s)


def keep_alphanumeric(s):
    return re.sub(r"[^A-Za-z0-9.' ]+", ' ', s)


def load_pretrained_embedding():
    import aspectnlp
    rmls_nlp_dir = os.path.dirname(aspectnlp.__file__)
    emb_file = os.path.join(rmls_nlp_dir, "absa_embedding",'custom.vec.bin')
    if not os.path.isfile(emb_file):
        print("downloading pretrained embedding model ...")
        download_file_from_google_drive('1mQPKHoa4SQr-skCO5XpzWpOxGB5z02-U',emb_file)

    return load_model(emb_file)


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
