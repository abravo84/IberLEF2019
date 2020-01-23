import csv, spacy, os, sys,csv, nltk, re, pickle, html
sys.path.append("/home/abravo/PycharmProjects/IberLEF2019")

import xml.etree.ElementTree as ET
from random import shuffle

from spacymoji import Emoji
import numpy as np

from utils.utils import preprocess_tweet, get_word_index, write_word_index, save_obj, load_obj, get_embedding_matrix, \
    get_spacy_nlp, get_task_data_for_class_task, EMBEDDING_CHAR, create_data_comb, get_task_data_for_class_task_haha


def train2txt(train_path, output_path):

    ofile = open(output_path, "w")

    with open(train_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            row[1] = row[1].replace("\n", " ")
            ofile.write("\t".join(row) + "\n")

    ofile.close()



if __name__ == '__main__':

    train_path = "/home/upf/corpora/IberLEF2019/HAHA/haha_2019_train.csv"

    output_path = "/home/upf/corpora/IberLEF2019/HAHA/preprocessed_data"

    preproc_train_path = os.path.join(output_path, "haha_2019_train.txt")

    #train2txt(train_path, preproc_train_path)

    test_path = "/home/upf/corpora/IberLEF2019/HAHA/haha_2019_test.csv"

    preproc_test_path = os.path.join(output_path, "haha_2019_test.txt")

    #train2txt(test_path, preproc_test_path)



    comb_path = os.path.join(output_path, "haha_2019_train_test.txt")
    #create_data_comb([preproc_train_path, preproc_test_path], True, comb_path)


    SAVE_WORD_INDEX = False
    SAVE_CHAR_INDEX = False
    SAVE_EMB_MATRIX = False
    SAVE_DATA = True
    SAVE_CHAR_DATA = False

    nlp = None

    tweet_col= 1
    label_col= 2



    if SAVE_WORD_INDEX:
        if not nlp:
            nlp = get_spacy_nlp('es_core_news_md', True)
        all_files = [preproc_train_path, preproc_test_path]

        word_index = get_word_index(nlp, all_files, True, 1)
        word_index_all_path = os.path.join(output_path, "word_index_all.txt")
        write_word_index(word_index, word_index_all_path)

        word_index_all_path = os.path.join(output_path, "word_index_all.pkl")
        save_obj(word_index, word_index_all_path)

    word_index_path = os.path.join(output_path, "word_index_all.pkl")
    word_index = load_obj(word_index_path)



    if SAVE_CHAR_INDEX:
        if not nlp:
            nlp = get_spacy_nlp('es_core_news_md', True)

        all_files = [preproc_train_path]

        char_index = get_word_index(nlp, all_files, True, tweet_col, True)
        char_index_all_path = os.path.join(output_path, "char_index_train.txt")
        write_word_index(char_index, char_index_all_path)
        char_index_all_path = os.path.join(output_path, "char_index_train.pkl")
        save_obj(char_index, char_index_all_path)
        print("CHAR INDEX PROCESSED!")

    #char_index_path = os.path.join(output_path, "char_index_train.pkl")
    #char_index = load_obj(char_index_path)


    if SAVE_DATA:
        if not nlp:
            nlp = get_spacy_nlp('es_core_news_md', True)

        data, labels, scores = get_task_data_for_class_task_haha(nlp, word_index, preproc_train_path, True, True, tweet_col, [2,-1], False)
        data_path = os.path.join(output_path, "data_train.pkl")
        labels_path = os.path.join(output_path, "labels_train.pkl")
        scores_path = os.path.join(output_path, "scores_train.pkl")
        save_obj(data, data_path)
        save_obj(labels, labels_path)
        save_obj(scores, scores_path)

        data, labels, scores = get_task_data_for_class_task_haha(nlp, word_index, preproc_test_path, True, False, tweet_col,
                                                    None, False)
        data_path = os.path.join(output_path, "data_test.pkl")
        save_obj(data, data_path)


    if SAVE_CHAR_DATA:
        if not nlp:
            nlp = get_spacy_nlp('es_core_news_md', True)

        data, labels = get_task_data_for_class_task(nlp, char_index, preproc_train_path, True, True, tweet_col, label_col, True)
        data_path = os.path.join(output_path, "char_data_train.pkl")
        save_obj(data, data_path)
