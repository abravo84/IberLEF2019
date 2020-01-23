import csv, spacy, os, sys,csv, nltk, re, pickle, html


from utils.utils import preprocess_tweet, get_word_index, write_word_index, save_obj, load_obj, get_embedding_matrix, \
    get_spacy_nlp, get_task_data_for_class_task, EMBEDDING_CHAR


def train2txt(train_path, label_path, preproc_train_path):



    ofile = open(preproc_train_path, "w")
    if label_path:
        ofile.write("\t".join(["TWEET", "LABEL"])+"\n")


        labels = []
        for line in open(label_path):
            labels.append(line.strip())

        i=0
        for line in open(train_path):
            ofile.write(line.strip() + "\t" + labels[i]+"\n")
            i+=1
    else:
        ofile.write("TWEET\n")
        i = 0
        for line in open(train_path):
            ofile.write(line.strip()+ "\n")
            i += 1

    ofile.close()





if __name__ == '__main__':

    train_path = "/home/upf/corpora/IberLEF2019/MEX-A3T/train/AggressiveDetection_train.txt"
    label_path = "/home/upf/corpora/IberLEF2019/MEX-A3T/train/AggressiveDetection_train_solution.txt"
    output_path = "/home/upf/corpora/IberLEF2019/MEX-A3T/preprocessed_data"
    preproc_train_path = os.path.join(output_path, "AggressiveDetection_train.txt")

    #train2txt(train_path, label_path, preproc_train_path)

    test_path = "/home/upf/corpora/IberLEF2019/MEX-A3T/test/AggressiveDetection_test.txt"
    preproc_test_path = os.path.join(output_path, "AggressiveDetection_test.txt")
    #train2txt(test_path, None, preproc_test_path)


    SAVE_WORD_INDEX = True
    SAVE_EMB_MATRIX = False
    SAVE_DATA = True

    SAVE_CHAR_INDEX = True
    SAVE_CHAR_DATA = True

    nlp = None

    tweet_col = 0
    label_col = 1

    if SAVE_WORD_INDEX:

        if not nlp:
            nlp = get_spacy_nlp('es_core_news_md', True)

        all_files = [preproc_train_path, preproc_test_path]

        word_index = get_word_index(nlp, all_files, True, tweet_col, False)
        word_index_all_path = os.path.join(output_path, "word_index_all.txt")
        write_word_index(word_index, word_index_all_path)

        word_index_all_path = os.path.join(output_path, "word_index_all.pkl")
        save_obj(word_index, word_index_all_path)
        print("WORD INDEX PROCESSED!")

    word_index_all_path = os.path.join(output_path, "word_index_all.pkl")
    word_index = load_obj(word_index_all_path)


    if SAVE_CHAR_INDEX:
        if not nlp:
            nlp = get_spacy_nlp('es_core_news_md', True)

        all_files = [preproc_train_path, preproc_test_path]

        char_index = get_word_index(nlp, all_files, True, tweet_col, True)
        char_index_all_path = os.path.join(output_path, "char_index_all.txt")
        write_word_index(char_index, char_index_all_path)

        char_index_all_path = os.path.join(output_path, "char_index_all.pkl")
        save_obj(char_index, char_index_all_path)
        print("CHAR INDEX PROCESSED!")

    char_index_all_path = os.path.join(output_path, "char_index_all.pkl")
    char_index = load_obj(char_index_all_path)

    labels_path = os.path.join(output_path, "labels_train.pkl")
    data_path = os.path.join(output_path, "data_train.pkl")
    data_test_path = os.path.join(output_path, "data_test.pkl")

    if SAVE_DATA:
        if not nlp:
            nlp = get_spacy_nlp('es_core_news_md', True)
        header = True

        data, labels = get_task_data_for_class_task(nlp, word_index, preproc_train_path, header, True,
                                                           tweet_col, label_col, False)
        save_obj(data, data_path)
        save_obj(labels, labels_path)

        #TEST
        data_test, labels = get_task_data_for_class_task(nlp, word_index, preproc_test_path, header, False,
                                                    tweet_col, None, False)
        save_obj(data_test, data_test_path)

        print("WORD DATA PROCESSED!")

    data_char_path = os.path.join(output_path, "char_data_train.pkl")
    data_char_test_path = os.path.join(output_path, "char_data_test.pkl")

    if SAVE_CHAR_DATA:
        if not nlp:
            nlp = get_spacy_nlp('es_core_news_md', True)
        header = True


        datac, labelsc = get_task_data_for_class_task(nlp, char_index, preproc_train_path, header, True,
                                                    tweet_col, label_col, True)
        save_obj(datac, data_char_path)


        #TEST
        data_test, labelsc = get_task_data_for_class_task(nlp, char_index, preproc_test_path, header, False,
                                                    tweet_col, None, True)
        save_obj(data_test, data_char_test_path)
        print("CHAR DATA PROCESSED!")