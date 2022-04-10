import io
import json
import os

import numpy as np
from transformers import (
    BertTokenizer,
    BertModel,
    RobertaTokenizer,
    RobertaModel,
    XLNetTokenizer,
    XLNetModel,
    BartTokenizer,
    BartModel,
)


class IMDBDataset:
    """IMDB dataset"""

    def __init__(self, path='./data/aclImdb', valid_size=0.2):
        """
        Create the IMDB dataset
        :param path: the directory of the IMDB dataset
        :param valid_size: percentage of the training set to use as a validation set
        """
        self._path = path
        self._valid_size = valid_size
        self.num_labels = 2
        (self.train_text, self.train_text_pair, self.train_y), (self.valid_text, self.valid_text_pair, self.valid_y), \
        (self.test_text, self.test_text_pair, self.test_y) = self.load_imdb(self._path, self._valid_size)

    def read_text(self, path):
        """
        Read text from IMDB training or test directory
        :param path: the directory of train or test data
        :return: a list of texts and a list of their labels
        """
        pos_path = path + '/pos'
        neg_path = path + '/neg'
        pos_files = [pos_path + '/' + x for x in sorted(os.listdir(pos_path)) if x.endswith('.txt')]
        neg_files = [neg_path + '/' + x for x in sorted(os.listdir(neg_path)) if x.endswith('.txt')]

        pos_list = [io.open(x, 'r', encoding='utf-8', errors='ignore').read().replace('<br />', '') for x in pos_files]
        neg_list = [io.open(x, 'r', encoding='utf-8', errors='ignore').read().replace('<br />', '') for x in neg_files]
        data_list = pos_list + neg_list
        labels_list = [1] * len(pos_list) + [0] * len(neg_list)

        return data_list, labels_list

    def load_imdb(self, path, valid_size):
        """
        Load the IMDB dataset from the pre-downloaded IMDB dataset directory
        :param path: the directory of the IMDB dataset
        :param valid_size: percentage of the training set to use as a validation set
        :return: IMDB training, validation, and test sets
        """
        test_path = path + '/test'
        train_path = path + '/train'
        test_text, test_y = self.read_text(test_path)
        text, y = self.read_text(train_path)

        # split original training set to a new training set and a validation set
        num_train = len(y)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        train_text = [text[i] for i in train_idx]
        train_text_pair = [None] * len(train_idx)
        train_y = [y[i] for i in train_idx]
        valid_text = [text[i] for i in valid_idx]
        valid_text_pair = [None] * len(valid_idx)
        valid_y = [y[i] for i in valid_idx]

        test_text_pair = [None] * len(test_y)

        return (train_text, train_text_pair, train_y), \
               (valid_text, valid_text_pair, valid_y), \
               (test_text, test_text_pair, test_y)


class MnliDataset:
    """Multi-Genre Natural Language Inference (MultiNLI/Mnli) dataset"""

    def __init__(self, path='./data/multinli_1.0', valid_size=0.2):
        """
        Create the MultiNLI dataset
        :param path: the directory of the MultiNLI dataset
        :param valid_size: percentage of the training set to use as a validation set
        """
        self._path = path
        self._valid_size = valid_size
        self.num_labels = 3
        (self.train_text, self.train_text_pair, self.train_y), (self.valid_text, self.valid_text_pair, self.valid_y), \
        (self.test_text, self.test_text_pair, self.test_y) = self.load_mnli(self._path, self._valid_size)

    def read_json(self, path):
        """
        Read jsonl file from MultiNLI training or dev file
        :param path: the directory of MultiNLI train or dev file
        :return: lists of texts, text pairs, and labels
        """
        data_info = []
        for line in open(path, 'r', encoding='utf-8', errors='ignore'):
            data_info.append(json.loads(line))

        labels = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

        text = [data['sentence1'] for data in data_info if data['gold_label'] != '-']
        text_pair = [data['sentence2'] for data in data_info if data['gold_label'] != '-']
        y = [labels[data['gold_label']] for data in data_info if data['gold_label'] != '-']

        return text, text_pair, y

    def load_mnli(self, path, valid_size):
        """
        load the MultiNLI dataset from the pre-downloaded MultiNLI dataset directory
        :param path: the directory of the MultiNLI dataset
        :param valid_size: percentage of the training set to use as a validation set
        :return: MultiNLI training, validation, and test sets
        """
        test_path = path + '/multinli_1.0_dev_mismatched.jsonl'
        train_path = path + '/multinli_1.0_train.jsonl'

        test_text, test_text_pair, test_y = self.read_json(test_path)
        text, text_pair, y = self.read_json(train_path)

        # split original training set to a new training set and a validation set
        num_train = len(y)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        train_text = [text[i] for i in train_idx]
        train_text_pair = [text_pair[i] for i in train_idx]
        train_y = [y[i] for i in train_idx]
        valid_text = [text[i] for i in valid_idx]
        valid_text_pair = [text_pair[i] for i in valid_idx]
        valid_y = [y[i] for i in valid_idx]

        return (train_text, train_text_pair, train_y), \
               (valid_text, valid_text_pair, valid_y), \
               (test_text, test_text_pair, test_y)


# set models' parameters
bert_params = {
    'cls_pos': 0,
    'learning_rate': 5e-5,
    'model_class': BertModel,
    'tokenizer_class': BertTokenizer,
    'pretrained_model_name': 'bert-base-cased'
}

roberta_params = {
    'cls_pos': 0,
    'learning_rate': 1e-5,
    'model_class': RobertaModel,
    'tokenizer_class': RobertaTokenizer,
    'pretrained_model_name': 'roberta-base'
}

xlnet_params = {
    'cls_pos': -1,
    'learning_rate': 2e-5,
    'model_class': XLNetModel,
    'tokenizer_class': XLNetTokenizer,
    'pretrained_model_name': 'xlnet-base-cased'
}

bart_params = {
    'cls_pos': -1,
    'learning_rate': 5e-6,
    'model_class': BartModel,
    'tokenizer_class': BartTokenizer,
    'pretrained_model_name': 'facebook/bart-base'
}
