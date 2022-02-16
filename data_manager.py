# data_manager.py: Loads and preprocesses data
# Copyright 2016 Ramon Vinas
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import html
import re
import string

class DataManager(object):
    def __init__(self, data_arr, stopwords_file=None, sequence_len=None, n_samples=None):
        """
        Initiallizes data manager. DataManager provides an interface to load, preprocess and split data into train,
        validation and test sets
        :param data_dir: Data directory containing the dataset file 'data.csv' with columns 'SentimentText' and
                         'Sentiment'
        :param stopwords_file: Optional. If provided, discards each stopword from original data
        :param sequence_len: Optional. Let m be the maximum sequence length in the dataset. Then, it's required that
                          sequence_len >= m. If sequence_len is None, then it'll be automatically assigned to m.
        :param n_samples: Optional. Number of samples to load from the dataset (useful for large datasets). If n_samples
                          is None, then the whole dataset will be loaded (be careful, if dataset is large it may take a
                          while to preprocess every sample)
        """
        self._stopwords_file = stopwords_file
        self._n_samples = n_samples
        self.sequence_len = 23
        self.data_arr = data_arr
        self._tensors = None
        self._lengths = None
        self._vocab = None
        self.vocab_size = None
        self.__preprocess()

    def __preprocess(self):
        """
        Preprocesses each sample loaded and stores intermediate files to avoid
        preprocessing later.
        """
        # Load data
        samples = self.data_arr

        # Cleans samples text
        samples = self.__clean_samples(samples)

        # Prepare vocabulary dict
        vocab = dict()
        vocab[''] = (0, len(samples))  # add empty word
        for sample in samples:
            sample_words = sample.split()
            for word in list(set(sample_words)):  # distinct words in list
                value = vocab.get(word)
                if value is None:
                    vocab[word] = (-1, 1)
                else:
                    encoding, count = value
                    vocab[word] = (-1, count + 1)

        # Remove the most uncommon words (they're probably grammar mistakes), encode samples into tensors and
        # store samples' lengths
        sample_lengths = []
        tensors = []
        word_count = 1
        for sample in samples:
            sample_words = sample.split()
            encoded_sample = []
            for word in list(set(sample_words)):  # distinct words in list
                value = vocab.get(word)
                if value is not None:
                    encoding, count = value
                    if count / len(samples) > 0.0001:
                        if encoding == -1:
                            encoding = word_count
                            vocab[word] = (encoding, count)
                            word_count += 1
                        encoded_sample += [encoding]
                    else:
                        del vocab[word]
            tensors += [encoded_sample]
            sample_lengths += [len(encoded_sample)]

        self.vocab_size = len(vocab)
        self._vocab = vocab
        self._lengths = np.array(sample_lengths)

        # Pad each tensor with zeros according self.sequence_len
        self.sequence_len, self._tensors = self.__apply_to_zeros(tensors, self.sequence_len)

    def __clean_samples(self, samples):
        """
        Cleans samples.
        :param samples: Samples to be cleaned
        :return: cleaned samples
        """
        print('Cleaning samples ...')
        # Prepare regex patterns
        ret = []
        reg_punct = '[' + re.escape(''.join(string.punctuation)) + ']'
        if self._stopwords_file is not None:
            stopwords = self.__read_stopwords()
            sw_pattern = re.compile(r'\b(' + '|'.join(stopwords) + r')\b')

        # Clean each sample
        for sample in samples:
            # Restore HTML characters
            text = html.unescape(sample)

            # Remove @users and urls
            words = text.split()
            words = [word for word in words if not word.startswith('@') and not word.startswith('http://')]
            text = ' '.join(words)

            # Transform to lowercase
            text = text.lower()

            # Remove punctuation symbols
            text = re.sub(reg_punct, ' ', text)

            # Replace CC(C+) (a character occurring more than twice in a row) for C
            text = re.sub(r'([a-z])\1{2,}', r'\1', text)

            # Remove stopwords
            if stopwords is not None:
                text = sw_pattern.sub('', text)
            ret += [text]

        return ret

    def __apply_to_zeros(self, lst, sequence_len=None):
        """
        Pads lst with zeros according to sequence_len
        :param lst: List to be padded
        :param sequence_len: Optional. Let m be the maximum sequence length in lst. Then, it's required that
                          sequence_len >= m. If sequence_len is None, then it'll be automatically assigned to m.
        :return: padding_length used and numpy array of padded tensors.
        """
        # Find maximum length m and ensure that m>=sequence_len
        inner_max_len = max(map(len, lst))
        if sequence_len is not None:
            if inner_max_len > sequence_len:
                raise Exception('Error: Provided sequence length is not sufficient')
            else:
                inner_max_len = sequence_len

        # Pad list with zeros
        result = np.zeros([len(lst), inner_max_len], np.int32)
        for i, row in enumerate(lst):
            for j, val in enumerate(row):
                result[i][j] = val
        return inner_max_len, result

    def __read_stopwords(self):
        """
        :return: Stopwords list
        """
        if self._stopwords_file is None:
            return None
        with open(self._stopwords_file, mode='r') as f:
            stopwords = f.read().splitlines()
        return stopwords
    
    def data(self, original_text):
        """Gets the data from the text file."""
        if original_text:
            data = self.data_arr

        return data, self._tensors, self._lengths