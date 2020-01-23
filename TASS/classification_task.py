import os, sys
sys.path.append("/home/abravo/PycharmProjects/IberLEF2019")
from TASS.preprocess import get_num2_cat


from utils.networks import lstm_simple_binary, plot_model_history, model_testing, cnn_binary_with_emb_layer_, \
    get_embedding_layer_from_matrix, lstm_simple_binary_with_emb_layer, cnn, model_lstm_atten
from utils.utils import get_embedding_matrix, load_obj, save_obj, DataTask, MAX_SEQUENCE_LENGTH, \
    MAX_SEQUENCE_CHAR_LENGTH
import numpy as np

def gogo(test_lang, mono):
    abravo = True

    output_path = "/home/upf/corpora/IberLEF2019/TASS/preprocessed_data"
    if abravo:
        output_path = "/home/abravo/corpora/IberLEF2019/TASS/preprocessed_data"

    char_data = ""

    tass_languages = ["es", "mx", "cr", "uy", "pe"]

    word_index_all_path = os.path.join(output_path, "word_index_all.pkl")

    word_index = load_obj(word_index_all_path)

    w2v_path_dict = {}
    w2v_path_dict["mx"] = "/home/upf/corpora/IberLEF2019/regional_emb/es-MX-100d.vec"
    w2v_path_dict["cu"] = "/home/upf/corpora/IberLEF2019/regional_emb/es-CU-100d.vec"
    w2v_path_dict["es"] = "/home/upf/corpora/IberLEF2019/regional_emb/es-ES-100d.vec"
    w2v_path_dict["cr"] = "/home/upf/corpora/IberLEF2019/regional_emb/es-CR-100d.vec"
    w2v_path_dict["uy"] = "/home/upf/corpora/IberLEF2019/regional_emb/es-UY-100d.vec"
    w2v_path_dict["pe"] = "/home/upf/corpora/IberLEF2019/regional_emb/es-PE-100d.vec"
    if abravo:
        w2v_path_dict["mx"] = "/home/abravo/corpora/IberLEF2019/regional_emb/es-MX-100d.vec"
        w2v_path_dict["cu"] = "/home/abravo/corpora/IberLEF2019/regional_emb/es-CU-100d.vec"
        w2v_path_dict["es"] = "/home/abravo/corpora/IberLEF2019/regional_emb/es-ES-100d.vec"
        w2v_path_dict["cr"] = "/home/abravo/corpora/IberLEF2019/regional_emb/es-CR-100d.vec"
        w2v_path_dict["uy"] = "/home/abravo/corpora/IberLEF2019/regional_emb/es-UY-100d.vec"
        w2v_path_dict["pe"] = "/home/abravo/corpora/IberLEF2019/regional_emb/es-PE-100d.vec"

    MODE_TEST = True

    emb_dim = 100

    if len(char_data):
        emb_dim = 50
    emb_matrix_dict = {}

    for k in tass_languages:
        if k in emb_matrix_dict:
            continue
        file_windex = None
        if not len(char_data):
            file_windex = w2v_path_dict[k]
        emb_matrix_dict[k] = get_embedding_matrix(word_index, emb_dim, file_windex)

    if MODE_TEST:

        new_list = list(tass_languages)

        new_list.remove(test_lang)

        set_lang = "_".join(sorted(new_list))
        if mono:
            set_lang = test_lang

        data_path = os.path.join(output_path, char_data + "data_" + set_lang + "_all_train_dev.pkl")
        labels_path = os.path.join(output_path, "labels_" + set_lang + "_all_train_dev.pkl")
        data = DataTask(data_path, labels_path, 0, 0)
    else:
        tass_data = {}
        for k in tass_languages:  # ["all","es", "mx", "cu"]:
            print("TASS " + k)
            data_path = os.path.join(output_path, char_data + "data_" + k + "_train.pkl")
            labels_path = os.path.join(output_path, "labels_" + k + "_train.pkl")
            test_data_path = os.path.join(output_path, char_data + "data_" + k + "_dev.pkl")
            test_labels_path = os.path.join(output_path, "labels_" + k + "_dev.pkl")
            id = DataTask(data_path, labels_path, 0.15, 0.1, test_data_path, test_labels_path)
            tass_data[k] = id

    max_seq = MAX_SEQUENCE_LENGTH
    if len(char_data):
        max_seq = MAX_SEQUENCE_CHAR_LENGTH

    if MODE_TEST:

        model = model_lstm_atten(emb_matrix_dict[test_lang], lr=0.0001, nlabels=4, nunits=50, max_seq=50)
    else:
        for k in tass_languages:
            print("MODEL: " + k)
            embedding_layer_k = get_embedding_layer_from_matrix(embedding_matrix=emb_matrix_dict[k],
                                                                trainable=True,
                                                                max_seq=max_seq)

            model = lstm_simple_binary_with_emb_layer(embedding_layer_k, lr=0.0001, bilstm=None, do=0.5, units=100,
                                                      nlabels=4)
            # model = cnn_binary_with_emb_layer_(emb_matrix_dict[k],lr=0.001,nlabels=4, nfilters=50, max_seq=max_seq)

            # model = cnn(emb_matrix_dict[k], num_filters=50, filter_sizes=[2, 3, 4], lr=1e-4, max_seq=50, do=0.5)

            model = model_lstm_atten(emb_matrix_dict[k], lr=0.0001, nlabels=4, nunits=50, max_seq=50)
            tass_data[k].set_model(model)
    """
    lr = 0.0001
    model = lstm_simple_binary(emb_matrix_dict[lang], lr, labels.shape[1])

    lr = 0.001
    nfilters = 50
    model = cnn_binary_with_emb_layer_(emb_matrix_dict[lang], lr, labels.shape[1], nfilters)
    """

    batch_size = 130
    if MODE_TEST:
        batch_size = 130
    epochs = 25
    lang = "es"
    class_weight = {0: 4.,
                    1: 1.,
                    2: 1.,
                    3: 4.}
    if MODE_TEST:
        history = model.fit(data.get_x_train(), data.get_y_train(),
                            batch_size=batch_size,
                            epochs=epochs,
                            shuffle=True)

    else:
        history = tass_data[lang].get_model().fit(tass_data[lang].get_x_train(), tass_data[lang].get_y_train(),
                                                  batch_size=batch_size,
                                                  epochs=epochs,
                                                  validation_data=(
                                                  tass_data[lang].get_x_val(), tass_data[lang].get_y_val()),
                                                  shuffle=True)

        plot_model_history(history)

    if not MODE_TEST:
        print(model_testing(tass_data[lang].get_model(), tass_data[lang].get_x_test(), tass_data[lang].get_y_test()))


    else:
        print("TESTING IROSVA " + test_lang)

        ids_path = os.path.join(output_path, "intertass_" + test_lang + "_test" + ".txt")

        header = 1
        ides_list = []
        for line in open(ids_path):
            if header:
                header = False
                continue
            ides_list.append(line.strip().split("\t")[0])

        data_path = os.path.join(output_path, "data_" + test_lang + "_" + "test" + ".pkl")
        data_test = load_obj(data_path)

        result_path = os.path.join(output_path, "cross_" + test_lang + ".tsv")

        if mono:
            result_path = os.path.join(output_path, "mono_" + test_lang + ".tsv")
        out = open(result_path, "w")

        i = 0
        while i < data_test.shape[0]:
            prediction = model.predict(np.array([data_test[i]]))
            predicted_label = np.argmax(prediction[0])

            line = []

            line.append(ides_list[i])
            line.append(get_num2_cat()[predicted_label])

            out.write("\t".join(line) + "\n")
            i += 1

        out.close()

if __name__ == '__main__':

    test_lang = "mx"
    mono = False

    tass_languages = ["mx", "cr", "uy", "pe"]

    for test_lang in tass_languages:
        gogo(test_lang, mono)
