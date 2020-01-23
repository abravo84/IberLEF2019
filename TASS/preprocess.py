import spacy, os, sys,csv, nltk, re, pickle, html
import xml.etree.ElementTree as ET
from spacymoji import Emoji
import numpy as np

from utils.utils import preprocess_tweet, get_word_index, write_word_index, save_obj, load_obj, get_embedding_matrix, \
    get_spacy_nlp, create_data_comb, get_task_data_for_class_task, EMBEDDING_CHAR


def get_cat2num():
    cat2num = {}
    cat2num["P"]=1
    cat2num["N"]=2
    cat2num["NONE"]=0
    cat2num["NEU"]=3
    return cat2num

def get_num2_cat():
    cat2num = {}
    cat2num[1]="P"
    cat2num[2]="N"
    cat2num[0]="NONE"
    cat2num[3]="NEU"

    return cat2num

def extract_info_from_xml(input_path, output_path, test=False):

    base_tree = ET.parse(input_path)
    base_root = base_tree.getroot()

    ofile = open(output_path, "w")
    header = ["TWEET_ID", "TWEET_TEXT", "LABEL"]
    ofile.write("\t".join(header) + "\n")

    cat2num = get_cat2num()

    for tweet in base_root.iter(tag='tweet'):

        t_id = tweet.find("tweetid").text
        t_user = tweet.find("user").text
        t_content = tweet.find("content").text.strip()
        t_date = tweet.find("date").text
        t_lang = tweet.find("lang").text
        t_label = "0"
        if not test:
            t_label = tweet.find("sentiment/polarity/value").text
            t_label = str(cat2num[t_label])

        ofile.write("\t".join([t_id, t_content, t_label])+"\n")

    ofile.close()


def xml2txt(input_path, output_path, test= False):
    if test:
        for filename in os.listdir(input_path):
            if not filename.endswith(".xml"):
                continue
            filename_path = os.path.join(input_path, filename)
            output_p_path = os.path.join(output_path, filename.replace(".xml", ".txt"))
            print(filename_path, output_p_path)
            extract_info_from_xml(filename_path, output_p_path, True)
    else:
        for folder in os.listdir(input_path):
            folder_lng = os.path.join(input_path, folder)
            for filename in os.listdir(folder_lng):
                if not filename.endswith(".xml"):
                    continue
                filename_path = os.path.join(folder_lng, filename)
                output_p_path = os.path.join(output_path, filename.replace(".xml", ".txt"))
                print(filename_path, output_p_path)
                extract_info_from_xml(filename_path, output_p_path)



def preproc(input_path, input_test_path, output_path):
    # Convert XML to TXT
    xml2txt(input_path, output_path)
    xml2txt(input_test_path, output_path, True)
    header = 1

    file_list = []
    file_list.append(os.path.join(output_path, "intertass_mx_train.txt"))
    file_list.append(os.path.join(output_path, "intertass_cr_train.txt"))
    file_list.append(os.path.join(output_path, "intertass_pe_train.txt"))
    file_list.append(os.path.join(output_path, "intertass_es_train.txt"))
    file_list.append(os.path.join(output_path, "intertass_uy_train.txt"))
    new_file_train = os.path.join(output_path, "intertass_all_train.txt")

    create_data_comb(file_list, header, new_file_train)

    file_list_dev = []
    file_list_dev.append(os.path.join(output_path, "intertass_mx_dev.txt"))
    file_list_dev.append(os.path.join(output_path, "intertass_cr_dev.txt"))
    file_list_dev.append(os.path.join(output_path, "intertass_pe_dev.txt"))
    file_list_dev.append(os.path.join(output_path, "intertass_es_dev.txt"))
    file_list_dev.append(os.path.join(output_path, "intertass_uy_dev.txt"))
    new_file_dev = os.path.join(output_path, "intertass_all_dev.txt")

    create_data_comb(file_list_dev, header, new_file_dev)

    new_file_train_dev = os.path.join(output_path, "intertass_all_train_dev.txt")

    create_data_comb([new_file_train, new_file_dev], header, new_file_train_dev)

    file_list_test = []
    file_list_test.append(os.path.join(output_path, "intertass_mx_test.txt"))
    file_list_test.append(os.path.join(output_path, "intertass_cr_test.txt"))
    file_list_test.append(os.path.join(output_path, "intertass_pe_test.txt"))
    file_list_test.append(os.path.join(output_path, "intertass_es_test.txt"))
    file_list_test.append(os.path.join(output_path, "intertass_uy_test.txt"))
    new_file_test = os.path.join(output_path, "intertass_all_test.txt")

    create_data_comb(file_list_dev, header, new_file_test)

    file_all_train_dev_test = os.path.join(output_path, "intertass_all_train_dev_test.txt")

    create_data_comb(file_list + file_list_dev + file_list_test, header, file_all_train_dev_test)

    files_by_lang = {}
    # files_by_lang["all"] = []
    files_by_lang["uy"] = []
    files_by_lang["es"] = []
    files_by_lang["pe"] = []
    files_by_lang["cr"] = []
    files_by_lang["mx"] = []

    for k in files_by_lang:
        set_files = []
        preproc_train_path = os.path.join(output_path, "intertass_" + k + "_train.txt")
        set_files.append(preproc_train_path)
        preproc_train_path = os.path.join(output_path, "intertass_" + k + "_dev.txt")
        set_files.append(preproc_train_path)
        preproc_train_path = os.path.join(output_path, "intertass_" + k + "_all_train_dev.txt")
        create_data_comb(set_files, header, preproc_train_path)

    files_sets = []
    for i in range(5):
        list_n = list(files_by_lang.keys())
        list_n.remove(list_n[i])
        print(sorted(list_n))
        files_sets.append(sorted(list_n))

    for s in files_sets:

        set_files_train = []
        set_files_dev = []
        set_files_all = []

        for k in s:
            preproc_train_path = os.path.join(output_path, "intertass_" + k + "_train.txt")
            set_files_train.append(preproc_train_path)
            preproc_train_path = os.path.join(output_path, "intertass_" + k + "_dev.txt")
            set_files_dev.append(preproc_train_path)

        set_files_all = set_files_dev + set_files_train

        k = "_".join(s)
        preproc_train_path = os.path.join(output_path, "intertass_" + k + "_train.txt")
        create_data_comb(set_files_train, header, preproc_train_path)
        preproc_train_path = os.path.join(output_path, "intertass_" + k + "_dev.txt")
        create_data_comb(set_files_dev, header, preproc_train_path)
        preproc_train_path = os.path.join(output_path, "intertass_" + k + "_all_train_dev.txt")
        create_data_comb(set_files_all, header, preproc_train_path)

if __name__ == '__main__':

    input_path = "/home/upf/corpora/IberLEF2019/TASS/train_dev/"

    output_path = "/home/upf/corpora/IberLEF2019/TASS/preprocessed_data"



    input_test_path = "/home/upf/corpora/IberLEF2019/TASS/test/"

    preproc(input_path, input_test_path, output_path)

    header = 1
    SAVE_WORD_INDEX = True
    SAVE_EMB_MATRIX = False
    SAVE_DATA = True
    SAVE_CHAR_DATA = True

    nlp = None

    tweet_col = 1
    label_col = 2

    files_by_lang = {}
    #files_by_lang["all"] = []
    files_by_lang["uy"] = []
    files_by_lang["es"] = []
    files_by_lang["pe"] = []
    files_by_lang["cr"] = []
    files_by_lang["mx"] = []



    files_sets = []
    for i in range(5):
        list_n = list(files_by_lang.keys())
        list_n.remove(list_n[i])
        print(sorted(list_n))
        files_sets.append(sorted(list_n))





    file_all_train_dev_test = os.path.join(output_path, "intertass_all_train_dev_test.txt")
    if SAVE_WORD_INDEX:
        if not nlp:
            nlp = get_spacy_nlp('es_core_news_md', True)

        """
        all_files = []

        for filename in os.listdir(output_path):
            if not filename.startswith("intertass_"):
                continue
            lang = filename.split("_")[1]
            filename_path = os.path.join(output_path, filename)
            all_files.append(filename_path)
            aux = files_by_lang.get(lang, [])
            aux.append(filename_path)
            files_by_lang[lang] = aux

        for k in files_by_lang:
            # WORD
            word_index = get_word_index(nlp, files_by_lang[k], True, tweet_col)
            word_index_all_path = os.path.join(output_path, "word_index_"+k+".txt")
            write_word_index(word_index, word_index_all_path)
            word_index_all_path = os.path.join(output_path, "word_index_"+k+".pkl")
            save_obj(word_index, word_index_all_path)

            # CHAR
            char_index = get_word_index(nlp, files_by_lang[k], True, tweet_col, True)
            char_index_all_path = os.path.join(output_path, "char_index_" + k + ".txt")
            write_word_index(char_index, char_index_all_path)
            char_index_all_path = os.path.join(output_path, "char_index_" + k + ".pkl")
            save_obj(char_index, char_index_all_path)
        """
        word_index = get_word_index(nlp, [file_all_train_dev_test], True, tweet_col)
        word_index_all_path = os.path.join(output_path, "word_index_all.txt")
        write_word_index(word_index, word_index_all_path)
        word_index_all_path = os.path.join(output_path, "word_index_all.pkl")
        save_obj(word_index, word_index_all_path)

        # CHAR
        char_index = get_word_index(nlp, [file_all_train_dev_test], True, tweet_col, True)
        char_index_all_path = os.path.join(output_path, "char_index_all.txt")
        write_word_index(char_index, char_index_all_path)
        char_index_all_path = os.path.join(output_path, "char_index_all.pkl")
        save_obj(char_index, char_index_all_path)

    files_sets_str = []
    for i in range(5):
        list_n = list(files_by_lang.keys())
        list_n.remove(list_n[i])
        print(sorted(list_n))
        files_sets_str.append("_".join(sorted(list_n)))

    if SAVE_DATA:
        if not nlp:
            nlp = get_spacy_nlp('es_core_news_md', True)
        header = True
        random = True

        word_index_all_path = os.path.join(output_path, "word_index_all.pkl")
        word_index = load_obj(word_index_all_path)

        for k in list(files_by_lang.keys())+files_sets_str:

            for mode in ["train", "dev", "test", "all_train_dev"]:

                if mode == "test" and len(k) > 2:
                    continue
                random = True
                if mode == "test":
                    random = False



                preproc_train_path = os.path.join(output_path, "intertass_" + k + "_" + mode + ".txt")
                print("............... ", preproc_train_path)

                data, labels = get_task_data_for_class_task(nlp, word_index, preproc_train_path, header, random,
                                                               tweet_col, label_col)

                data_path = os.path.join(output_path, "data_" + k + "_"+ mode+ ".pkl")
                labels_path = os.path.join(output_path, "labels_" + k + "_"+ mode+ ".pkl")
                save_obj(data, data_path)
                save_obj(labels, labels_path)
        """
        for s in files_sets:

            for mode in ["train", "dev", "all"]:
                set_files = []

                print(mode, s)
                for k in s:
                    preproc_train_path = os.path.join(output_path, "intertass_" + k + "_" + mode + ".txt")
                    set_files.append(preproc_train_path)
                print(set_files)
                file_all_train_dev = os.path.join(output_path, "intertass_" + "_".join(s) + "_" + mode + ".txt")
                print(file_all_train_dev)
                create_data_comb(set_files, header, file_all_train_dev)

                data, labels = get_task_data_for_class_task(nlp, word_index, file_all_train_dev, header, random,
                                                            tweet_col, label_col)
                k = "_".join(s)
                data_path = os.path.join(output_path, "data_" + k + "_" + mode + ".pkl")
                labels_path = os.path.join(output_path, "labels_" + k + "_" + mode + ".pkl")
                save_obj(data, data_path)
                save_obj(labels, labels_path)
            
            k = "_".join(s)
            file_all_train_dev = os.path.join(output_path, "intertass_" + "_".join(s) + "_all.txt")
            data, labels = get_task_data_for_class_task(nlp, word_index, file_all_train_dev, header, random,
                                                        tweet_col, label_col)
            data_path = os.path.join(output_path, "data_" + k + "_all.pkl")
            labels_path = os.path.join(output_path, "labels_" + k + "_all.pkl")
            save_obj(data, data_path)
            save_obj(labels, labels_path)
            """


    if SAVE_CHAR_DATA:
        if not nlp:
            nlp = get_spacy_nlp('es_core_news_md', True)
        header = True
        random = True

        word_index_all_path = os.path.join(output_path, "char_index_all.pkl")
        word_index = load_obj(word_index_all_path)

        for k in list(files_by_lang.keys())+files_sets_str:

            for mode in ["train", "dev", "all_train_dev", "test"]:

                if mode == "test" and len(k) > 2:
                    continue
                random = True
                if mode == "test":
                    random = False


                preproc_train_path = os.path.join(output_path, "intertass_" + k + "_" + mode + ".txt")
                print("............... ", preproc_train_path)
                data, labels = get_task_data_for_class_task(nlp, word_index, preproc_train_path, header, random,
                                                               tweet_col, label_col, True)

                data_path = os.path.join(output_path, "char_data_" + k + "_"+ mode+ ".pkl")
                save_obj(data, data_path)

        """
        for s in files_sets:

            for mode in ["train", "dev"]:
                set_files = []
                print(mode, s)
                for k in s:
                    preproc_train_path = os.path.join(output_path, "intertass_" + k + "_" + mode + ".txt")
                    set_files.append(preproc_train_path)

                file_all_train_dev = os.path.join(output_path, "intertass_" + "_".join(s) + "_" + mode + ".txt")
                create_data_comb(set_files, header, file_all_train_dev)

                data, labels = get_task_data_for_class_task(nlp, word_index, file_all_train_dev, header, random,
                                                            tweet_col, label_col, True)
                k = "_".join(s)
                data_path = os.path.join(output_path, "char_data" + k + "_" + mode + ".pkl")

                save_obj(data, data_path)
            
            k = "_".join(s)
            file_all_train_dev = os.path.join(output_path, "intertass_" + k + "_all.txt")
            data, labels = get_task_data_for_class_task(nlp, word_index, file_all_train_dev, header, random,
                                                        tweet_col, label_col, True)
            data_path = os.path.join(output_path, "char_data_" + k + "_all.pkl")
            save_obj(data, data_path)
            """