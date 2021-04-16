# standard libraries
import os
import time
import hashlib

# Third-party libraries
import dill as pkl
import shutil

class Logger(object):
    def __init__(self):
        self._logger = None

    def init(self, logdir, name='log'):
        if self._logger is None:
            import logging
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            log_file = os.path.join(logdir, name)
            if os.path.exists(log_file):
                os.remove(log_file)
            self._logger = logging.getLogger()
            self._logger.setLevel('INFO')
            fh = logging.FileHandler(log_file)
            ch = logging.StreamHandler()
            self._logger.addHandler(fh)
            self._logger.addHandler(ch)

    def info(self, str_info):
        self.init('/tmp', 'tmp.log')
        self._logger.info(str_info)

def parse_data_file(data_file):
    f=open(data_file, 'r')
    lines=f.readlines()
    f.close()
    return [line.strip() for line in lines if line.strip()[0]!="#"]

def parse_attr_file(attr_file):
    f = open(attr_file, 'r')
    lines = f.readlines()
    f.close()
    return [tuple(line.strip().split(',')) for line in lines if line.strip()[0]!="#"]

def ensure_dir(path, erase=False):
    if os.path.exists(path) and erase:
        print("Removing old folder {}".format(path))
        shutil.rmtree(path)
    if not os.path.exists(path):
        print("Creating folder {}".format(path))
        os.makedirs(path)

def load_pickle(path):
    begin_st = time.time()
    with open(path, 'rb') as f:
        print("Loading pickle object from {}".format(path))
        v = pkl.load(f)
    print("=> Done ({:.4f} s)".format(time.time() - begin_st))
    return v

def dump_pickle(obj, path):
    with open(path, 'wb') as f:
        print("Dumping pickle object to {}".format(path))
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)

def expand_user(path):
    return os.path.abspath(os.path.expanduser(path))

def md5(s):
    m = hashlib.md5()
    m.update(s)
    return m.hexdigest()