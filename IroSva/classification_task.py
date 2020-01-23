import os, sys

from utils.networks import lstm_simple_binary, plot_model_history, model_testing, cnn_binary_with_emb_layer_, \
    cnn_binary_with_emb_layer_char_word, cnn_binary_with_emb_layer_lstm, get_cnn_model_v2
from utils.utils import get_embedding_matrix, load_obj, save_obj, MAX_SEQUENCE_LENGTH, MAX_SEQUENCE_CHAR_LENGTH, \
    EMBEDDING_CHAR, create_all_data

if __name__ == '__main__':
    output_path = "/home/upf/corpora/IberLEF2019/IroSva/preprocessed_data"
    char_mode = False
    char_word_mode = False
    lang = "mx"

    #word_index_path = os.path.join(output_path, "word_index_train.pkl")
    #word_index = load_obj(word_index_path)

    test_data = {}
    test_label = {}

    data = None
    labels = None

    data_pathw = ""
    data_pathc = ""

    if not char_mode:
        print("WORD MODE!!!")
        #emb_matrix_filename = "emb_matrix_"
        #emb_matrix_filename = "emb_matrix_fb_"

        #emb_matrix_path = os.path.join(output_path, emb_matrix_filename+lang+".pkl")
        #emb_matrix = load_obj(emb_matrix_path)
        #####################

        word_index_all_path = os.path.join(output_path, "word_index_all.pkl")
        word_index = load_obj(word_index_all_path)

        w2v_path = "/home/upf/Downloads/model_swm_300-6-10-low_es.w2v"
        w2v_path = "/home/upf/corpora/IberLEF2019/regional_emb/es-MX-100d.vec"
        emb_dim = 100
        emb_matrix = get_embedding_matrix(word_index, emb_dim, w2v_path)


        print("EMB_MATRIX CREATED!")

        print("LOADING TRAINING DATA: ALL LANGUAGES")

        data_path = os.path.join(output_path, "data_"+lang+"_train.pkl")
        #data_path = os.path.join(output_path, "data_" + lang + "_trainset.pkl")
        data = load_obj(data_path)
        print("load", data_path)

        labels_path = os.path.join(output_path, "labels_"+lang+"_train.pkl")
        #labels_path = os.path.join(output_path, "labels_" + lang + "_trainset.pkl")
        labels = load_obj(labels_path)
        print("load", labels_path)

        """
        for l in ["es", "mx", "cu"]:
            print("LOADING TEST DATA:", l)
            data_path = os.path.join(output_path, "data_" + l + "_devset.pkl")
            datat = load_obj(data_path)
            print("load", data_path)
            test_data[l] = datat

            labels_path = os.path.join(output_path, "labels_" + l + "_devset.pkl")
            labelst = load_obj(labels_path)
            print("load", labels_path)
            test_label[l] = labelst

        """
        lr = 0.001
        nunits = 100
        #model = lstm_simple_binary(emb_matrix, lr, labels.shape[1], nunits, MAX_SEQUENCE_LENGTH)

        lr = 0.0001
        nfilters = 50

        #model = cnn_binary_with_emb_layer_(emb_matrix, lr, labels.shape[1], nfilters, MAX_SEQUENCE_LENGTH)

        model = get_cnn_model_v2(emb_matrix, lr, labels.shape[1], nfilters, MAX_SEQUENCE_LENGTH)

    elif char_word_mode:
        word_index_all_path = os.path.join(output_path, "word_index_all.pkl")
        word_indexw = load_obj(word_index_all_path)

        w2v_path = "/home/upf/Downloads/model_swm_300-6-10-low_es.w2v"
        w2v_path = "/home/upf/corpora/IberLEF2019/regional_emb/es-MX-100d.vec"
        emb_dim = 100
        emb_matrixw = get_embedding_matrix(word_indexw, emb_dim, w2v_path)

        print("EMB_MATRIX CREATED!")

        print("LOADING TRAINING DATA: ALL LANGUAGES")

        data_pathw = os.path.join(output_path, "data_" + lang + "_train.pkl")
        # data_path = os.path.join(output_path, "data_" + lang + "_trainset.pkl")
        dataw = load_obj(data_pathw)
        print("load", data_pathw)

        labels_path = os.path.join(output_path, "labels_" + lang + "_train.pkl")
        # labels_path = os.path.join(output_path, "labels_" + lang + "_trainset.pkl")
        labels = load_obj(labels_path)
        print("load", labels_path)


        #######

        word_index_all_path = os.path.join(output_path, "char_index_all.pkl")
        word_indexc = load_obj(word_index_all_path)
        emb_matrixc = get_embedding_matrix(word_indexc, EMBEDDING_CHAR)

        print("EMB_MATRIX CREATED!")
        print("LOADING TRAINING DATA: ALL LANGUAGES")
        data_pathc = os.path.join(output_path, "char_data_" + lang + "_train.pkl")
        datac = load_obj(data_pathc)
        print("load", data_pathc)

        lr = 0.0001
        nfilters = 50

        model = cnn_binary_with_emb_layer_char_word(emb_matrixw, emb_matrixc,lr,labels.shape[1], nfilters, MAX_SEQUENCE_LENGTH, MAX_SEQUENCE_CHAR_LENGTH)
    else:
        #CHAR
        print("CHAR MODE!!!")
        #emb_matrix_path = os.path.join(output_path, "emb_char_matrix_" + lang + ".pkl")
        #emb_matrix = load_obj(emb_matrix_path)
        #print("load", emb_matrix_path)

        word_index_all_path = os.path.join(output_path, "char_index_all.pkl")
        word_index = load_obj(word_index_all_path)
        emb_matrix = get_embedding_matrix(word_index, EMBEDDING_CHAR)

        print("EMB_MATRIX CREATED!")
        print("LOADING TRAINING DATA: ALL LANGUAGES")
        data_path = os.path.join(output_path, "char_data_" + lang + "_train.pkl")
        data = load_obj(data_path)
        print("load", data_path)

        labels_path = os.path.join(output_path, "labels_" + lang + "_train.pkl")
        labels = load_obj(labels_path)
        print("load", labels_path)

        """
        for l in ["es", "mx", "cu"]:
            print("LOADING TEST DATA:", l)
            data_path = os.path.join(output_path, "char_data_" + l + "_devset.pkl")
            datat = load_obj(data_path)
            print("load", data_path)
            test_data[l] = datat

            labels_path = os.path.join(output_path, "labels_" + l + "_devset.pkl")
            labelst = load_obj(labels_path)
            print("load", labels_path)
            test_label[l] = labelst
        """

        lr = 0.001
        nunits= 50
        #model = lstm_simple_binary(emb_matrix, lr,labels.shape[1],nunits, MAX_SEQUENCE_CHAR_LENGTH)

        lr = 0.001
        nfilters = 100

        model = cnn_binary_with_emb_layer_(emb_matrix,lr,labels.shape[1], nfilters, MAX_SEQUENCE_CHAR_LENGTH)



    if char_word_mode:
        batch_size = 68  # 55
        epochs = 50
        print(labels.shape[0])


        x_trainw, y_trainw, x_testw, y_testw, x_valw, y_valw = create_all_data(data_pathw, labels_path, test_perc=0.15,
                                                                         val_per=0.1, val_bias=0)

        x_trainc, y_trainc, x_testc, y_testc, x_valc, y_valc = create_all_data(data_pathc, labels_path, test_perc=0.15,
                                                                               val_per=0.1, val_bias=0)

        history = model.fit([x_trainw, x_trainc], y_trainw,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=([x_valw, x_valc], y_valw),
                            shuffle=True)

        plot_model_history(history)

        print(model_testing(model, [x_testw, x_testc], y_testw))

    else:

        batch_size = 68#55
        epochs = 50
        print(labels.shape[0])
        print("SHAPE", data.shape)

        x_train, y_train, x_test, y_test, x_val, y_val = create_all_data(data_path, labels_path, test_perc=0.15, val_per=0.1, val_bias=0)


        """
        history = model.fit(data, labels,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.10,
                            shuffle=True)
        """
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_val, y_val),
                            shuffle=True)

        plot_model_history(history)

        print(model_testing(model, x_test, y_test))

        #for l in ["es", "mx", "cu"]:
        #    print("TESTING:", l)
        #    print(model_testing(model,test_data[l], test_label[l]))
        #    print("\n\n##############################\n\n")

