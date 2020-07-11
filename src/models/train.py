import logging
import pickle

import numpy as np

from src.data.split import split_data
from src.features.dictionary import create_dictionary
from src.data.prepare_data import read_sample, create_classes
from src.features.tokenize import tokenize_classes




def train():
    df = read_sample()
    document_classes = create_classes(df)

    word_classes = tokenize_classes(document_classes, False)

    negative_words = [item for sublist in word_classes['NEG'] for item in sublist]
    positive_words = [item for sublist in word_classes['POS'] for item in sublist]

    dictionary = create_dictionary([negative_words, positive_words])

    negative_split = split_data(negative_words, (1, 0.0, 0.0))
    positive_split = split_data(positive_words, (1, 0.0, 0.0))

    negative_bow = dictionary.doc2bow(negative_split['train'])
    positive_bow = dictionary.doc2bow(positive_split['train'])


    total_negative = len(negative_split['train']) + len(negative_bow)
    total_positive = len(positive_split['train']) + len(positive_bow)

    negative_prob = np.log(len(negative_split['train']) / (len(negative_split['train']) + len(positive_split['train'])))
    positive_prob = np.log(len(positive_split['train']) / (len(negative_split['train']) + len(positive_split['train'])))

    negative_word_probs = {}
    for id, count in negative_bow:
        negative_word_probs[dictionary[id]] = {
            'id': id,
            'logprob': np.log((count + 1) / total_negative),
        }

    negative_word_probs[-1] = {
        'id': -1,
        'logprob': np.log(1 / total_negative)
    }

    positive_word_probs = {}
    for id, count in positive_bow:
        positive_word_probs[dictionary[id]] = {
            'id': id,
            'logprob': np.log((count + 1) / total_positive),
        }

    positive_word_probs[-1] = {
        'id': -1,
        'logprob': np.log(1 / total_positive)
    }

    model = {
        'POS_PROB': positive_prob,
        'NEG_PROB': negative_prob,
        'COND_POS_PROBS': positive_word_probs,
        'COND_NEG_PROBS': negative_word_probs,
    }

    with open("models/model.pkl", "wb") as output_file:
        pickle.dump(model, output_file)
    logging.info('Model saved to artifact model.pkl')