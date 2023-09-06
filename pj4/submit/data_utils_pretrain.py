# -*- encoding: utf-8 -*-
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from sklearn.model_selection import train_test_split
from gensim import utils
import gensim.parsing.preprocessing as gsp


def collate_fn(batch):
	"""pad the sequences so that they have the same length.
		batch: (sentence_vector, label)
	"""
	sentences, labels = zip(*batch)
	sentences = [torch.LongTensor(sentence) for sentence in sentences]
    # pad_sequence(sequences (list[Tensor]) – list of variable length sequences
	sentences = pad_sequence(sentences, batch_first=True)
	labels = torch.stack(labels)
	return sentences, labels


# padding is important for the RNN model, because the RNN model needs to process the sequences with the same length.

class MyDataset(nn.Module):
    '''Customized Dataset for the dataset'''
    def __init__(self, data: pd.DataFrame, vocab_set=None):
        super(MyDataset, self).__init__()
        phrases = data['clean_sentence'].values
        sentiments = data['Sentiment'].values

        sentences = [phrase.split() for phrase in phrases]

        if vocab_set is None:
            word2ind, ind2word = get_vocab_token(data['cleaned_sentence'])
            self.word2ind = word2ind
            self.ind2word = ind2word
            self.vocab_size = len(word2ind)
        else:
            self.word2ind = vocab_set[0]
            self.ind2word = vocab_set[1]
            self.vocab_size = len(self.word2ind)

        self.sentences, self.targets = [], []   # target是longtensor, sentences是list
        for idx, sentence in enumerate(sentences):
            sentence = [self.word2ind[word] for word in sentence if word in self.word2ind] # NOTE: if word in self.word2ind 是为了预训练模型
            if len(sentence) > 0:
                self.sentences.append(torch.LongTensor(sentence))
                self.targets.append(sentiments[idx])
        self.targets = torch.LongTensor(self.targets)

    def __len__(self):
        return len(self.sentences)  # self.targets.size(0)

    def __getitem__(self, idx):
        return self.sentences[idx], self.targets[idx]


def tsvReader(filepath):
    '''read in .tsv files, and leave the sentence ID'''
    df = pd.read_csv(filepath, sep="\t")
    return df[['SentenceId','PhraseId','Phrase','Sentiment']] # 只保留phrase ID，句子，label

filters = [
            gsp.strip_tags,
            gsp.strip_punctuation,
            gsp.strip_multiple_whitespaces,
            gsp.remove_stopwords,   
            gsp.strip_short,
            # gsp.stem_text   # NOTE: 导入pretrain的时候，记得把这个去掉
        ]

def clean_text(sentence):
    '''clean the text, including lower case, remove stopwords, stemming, etc.'''
    sentence = sentence.lower()
    sentence = utils.to_unicode(sentence)
    for func in filters:
        sentence = func(sentence)
    return sentence

def add_clean(data):
    '''add a new column to the data, which is the cleaned sentence'''
    data['clean_sentence'] = data['Phrase'].apply(clean_text)
    return data


def get_vocab_token(clean_text_data):
    '''get vocabulary and tokens mapping from data
        input: text_data, cleaned sentences list
    '''
    splited_text_data = [sentence.split() for sentence in clean_text_data]
    vocab = set(chain(*splited_text_data))

    word2ind = dict()
    ind2word = dict()
    for i, word in enumerate(vocab):
        word2ind[word] = i
        ind2word[i] = word
    ind2word[len(vocab)] = 'UNK'
    word2ind['UNK'] = len(word2ind)

    return word2ind, ind2word


def train_dev_test_spliter(dataset, train_propor=0.8, test=True):   
    '''split data into train: development: test = 8:1:1 '''
    train_ind, test_ind = train_test_split(np.unique(dataset['SentenceId']), test_size=1-train_propor, random_state=42)
    if test:
        test_ind, dev_ind = train_test_split(test_ind, test_size=0.5, random_state=42)
        train_data = dataset[dataset['SentenceId'].isin(train_ind)][['clean_sentence', 'Sentiment']]
        test_data = dataset[dataset['SentenceId'].isin(test_ind)][['clean_sentence', 'Sentiment']]
        dev_data = dataset[dataset['SentenceId'].isin(dev_ind)][['clean_sentence', 'Sentiment']]

        return train_data, test_data, dev_data
    else:
        train_data = dataset[dataset['SentenceId'].isin(train_ind)][['clean_sentence', 'Sentiment']]
        test_data = dataset[dataset['SentenceId'].isin(test_ind)][['clean_sentence', 'Sentiment']]
        return train_data, test_data

# #%%
# def load_glove_model(file):
# 	print("Loading Glove Model...")
# 	f = open(file, 'r', encoding='utf-8')
# 	vocab = set()
# 	embeddings = []
# 	for line in f:
# 		splitLine = line.split()
# 		word = splitLine[0]
# 		embedding = [float(val) for val in splitLine[1:]]
# 		vocab.add(word)
# 		embeddings.append(embedding)
# 	embeddings = torch.tensor(embeddings, dtype=torch.float32)
# 	print(f"Done! {len(vocab)} words loaded!")
# 	return vocab, embeddings



# #%%
# glove_path = 'D:/universityWorks/thirdYear/Spring/DISC-NLP/pj3/data/glove.6B/'
# vocab, glove_embeddings = load_glove_model(glove_path + 'glove.6B.300d.txt')

# print(type(glove_embeddings))

#%%
################################################
if __name__ == '__main__':
    # read in 
    filename = "./dataset/pj2/train.tsv"
    dataset = tsvReader(filename)
    dataset = add_clean(dataset)

    # splitting the dataset into Training, Developing and Testing Data
    train_data, test_data, dev_data = train_dev_test_spliter(dataset, train_propor=0.8, test=True)
    word2ind, ind2word = get_vocab_token(train_data['clean_sentence'])

    train_data = MyDataset(train_data, vocab_set=(word2ind, ind2word))
    dev_data = MyDataset(dev_data, vocab_set=(train_data.word2ind, train_data.ind2word))
    test_data = MyDataset(test_data, vocab_set=(train_data.word2ind, train_data.ind2word))
