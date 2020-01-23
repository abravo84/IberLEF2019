import os

from utils.utils import get_word_index_from_files, write_word_index, save_obj, load_obj, get_embedding_matrix, \
    EMBEDDING_WORD

if __name__ == '__main__':

    output_path = "/home/upf/corpora/IberLEF2019/multitask"

    irosva_path = "/home/upf/corpora/IberLEF2019/IroSva/preprocessed_data"
    mexa3t_path = "/home/upf/corpora/IberLEF2019/MEX-A3T/preprocessed_data"
    haha_path = "/home/upf/corpora/IberLEF2019/HAHA/preprocessed_data"
    tass_path = "/home/upf/corpora/IberLEF2019/TASS/preprocessed_data"


    tasks = ["irosva", "haha", "mexa3t"]

    tag = "_".join(sorted(tasks))


    word_index_files = []
    word_index_files.append(os.path.join(irosva_path, "word_index_all.txt"))
    word_index_files.append(os.path.join(haha_path, "word_index_train.txt"))
    word_index_files.append(os.path.join(mexa3t_path, "word_index_all.txt"))



    word_index = get_word_index_from_files(word_index_files)

    word_index_all_path = os.path.join(output_path, "word_index_"+tag+".txt")
    write_word_index(word_index, word_index_all_path)

    word_index_all_path = os.path.join(output_path, "word_index_"+tag+".pkl")
    save_obj(word_index, word_index_all_path)

    word_index_all_path = os.path.join(output_path, "word_index_" + tag + ".pkl")
    word_index = load_obj(word_index_all_path)

