import spacy, os, sys,csv, nltk, re, pickle, html
from spacymoji import Emoji
import numpy as np

from utils.utils import get_word_index, write_word_index, save_obj, load_obj, get_embedding_matrix, \
    get_spacy_nlp, get_task_data_for_class_task, EMBEDDING_CHAR, EMBEDDING_WORD, split_file_train_test


def train2txt(train_path, preproc_train_path):
    ofile = open(preproc_train_path, "w")

    tweet_text_col = 3
    with open(train_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            row[tweet_text_col] = row[tweet_text_col].replace("\n", " ")
            ofile.write("\t".join(row) + "\n")

    ofile.close()

if __name__ == '__main__':

    SAVE_WORD_INDEX = True
    SAVE_EMB_MATRIX = False
    SAVE_DATA = True
    SAVE_CHAR_DATA = True
    SAVE_TRAIN_DEV_DATA = False

    SAVE_TRAIN_DEV_CHAR_DATA = False

    train_path = "/home/upf/corpora/IberLEF2019/IroSva/IberLEF19-IroSvA-training-20190331"
    output_path = "/home/upf/corpora/IberLEF2019/IroSva/preprocessed_data"

    for filename in os.listdir(train_path):
        if not filename.endswith(".csv"):
            continue
        preproc_train_path = os.path.join(output_path, filename.replace(".csv", ".txt"))
        #train2txt(os.path.join(train_path, filename), preproc_train_path)

    test_path = "/home/upf/corpora/IberLEF2019/IroSva/IberLEF19-IroSvA-test-20190420"

    for filename in os.listdir(test_path):
        if not filename.endswith(".csv"):
            continue
        preproc_test_path = os.path.join(output_path, filename.replace(".csv", ".txt"))
        #train2txt(os.path.join(test_path, filename), preproc_test_path)


    nlp = None
    tweet_col = 3
    label_col = 2
    files_by_lang = {}
    files_by_lang["all"] = []
    files_by_lang["cu"] = []
    files_by_lang["es"] = []
    files_by_lang["mx"] = []

    if SAVE_WORD_INDEX:
        if not nlp:
            nlp = get_spacy_nlp('es_core_news_md', True)
        # WORD
        input_path = os.path.join(output_path, "irosva.all.train_and_test.txt")

        word_index = get_word_index(nlp, [input_path], True, tweet_col)
        word_index_all_path = os.path.join(output_path, "word_index_all.txt")
        write_word_index(word_index, word_index_all_path)
        word_index_all_path = os.path.join(output_path, "word_index_all.pkl")
        save_obj(word_index, word_index_all_path)

        # CHAR
        char_index = get_word_index(nlp, [input_path], True, tweet_col, True)
        char_index_all_path = os.path.join(output_path, "char_index_all.txt")
        write_word_index(char_index, char_index_all_path)
        char_index_all_path = os.path.join(output_path, "char_index_all.pkl")
        save_obj(char_index, char_index_all_path)

    if SAVE_DATA:
        if not nlp:
            nlp = get_spacy_nlp('es_core_news_md', True)
        header = True
        random = True

        word_index_all_path = os.path.join(output_path, "word_index_all.pkl")
        word_index = load_obj(word_index_all_path)

        for k in files_by_lang:

            preproc_train_path = os.path.join(output_path, "irosva." + k + ".training.txt")

            data, labels = get_task_data_for_class_task(nlp, word_index, preproc_train_path, header, random,
                                                           tweet_col, label_col)

            data_path = os.path.join(output_path, "data_" + k + "_train.pkl")
            labels_path = os.path.join(output_path, "labels_" + k + "_train.pkl")
            save_obj(data, data_path)
            save_obj(labels, labels_path)

            #TEST

            if k == "all":
                continue
            preproc_train_path = os.path.join(output_path, "irosva." + k + ".test.txt")
            data, labels = get_task_data_for_class_task(nlp, word_index, preproc_train_path, header, False,
                                                        tweet_col, None, False)
            data_path = os.path.join(output_path, "data_" + k + "_test.pkl")
            save_obj(data, data_path)



    if SAVE_CHAR_DATA:
        if not nlp:
            nlp = get_spacy_nlp('es_core_news_md', True)
        header = True
        random = True

        word_index_all_path = os.path.join(output_path, "char_index_all.pkl")
        word_index = load_obj(word_index_all_path)

        for k in files_by_lang:


            preproc_train_path = os.path.join(output_path, "irosva." + k + ".training.txt")

            data, labels = get_task_data_for_class_task(nlp, word_index, preproc_train_path, header, random,
                                                        tweet_col, None, True)
            data_path = os.path.join(output_path, "char_data_" + k + "_train.pkl")
            save_obj(data, data_path)


            if k == "all":
                continue

            preproc_train_path = os.path.join(output_path, "irosva." + k + ".test.txt")

            data, labels = get_task_data_for_class_task(nlp, word_index, preproc_train_path, header, False,
                                                           tweet_col, None, True)
            data_path = os.path.join(output_path, "char_data_" + k + "_test.pkl")
            save_obj(data, data_path)



    if SAVE_TRAIN_DEV_DATA:
        """
        header = True
        random = True
        for k in files_by_lang:
            preproc_train_path = os.path.join(output_path, "irosva." + k + ".training.txt")
            split_file_train_test(preproc_train_path, output_path, "irosva." + k, header, random, test_perc=0.15)
        """

        if not nlp:
            nlp = get_spacy_nlp('es_core_news_md', True)
        header = True
        random = True

        word_index_all_path = os.path.join(output_path, "word_index_all.pkl")
        word_index = load_obj(word_index_all_path)

        for k in files_by_lang:

            for set_name in ["trainset", "devset"]:
                preproc_train_path = os.path.join(output_path, "irosva." + k + "_" + set_name + ".txt")
                data, labels = get_task_data_for_class_task(nlp, word_index, preproc_train_path, header, random,
                                                               tweet_col, label_col)
                data_path = os.path.join(output_path, "data_" + k + "_" + set_name + ".pkl")
                labels_path = os.path.join(output_path, "labels_" + k + "_" + set_name + ".pkl")
                save_obj(data, data_path)
                save_obj(labels, labels_path)

    if SAVE_TRAIN_DEV_CHAR_DATA:
        if not nlp:
            nlp = get_spacy_nlp('es_core_news_md', True)
        header = True
        random = True

        word_index_all_path = os.path.join(output_path, "char_index_all.pkl")
        word_index = load_obj(word_index_all_path)

        for k in files_by_lang:
            for set_name in ["trainset", "devset"]:
                preproc_train_path = os.path.join(output_path, "irosva." + k + "_" + set_name + ".txt")
                data, labels = get_task_data_for_class_task(nlp, word_index, preproc_train_path, header, random,
                                                               tweet_col, None, True)
                data_path = os.path.join(output_path, "char_data_" + k + "_" + set_name + ".pkl")
                save_obj(data, data_path)