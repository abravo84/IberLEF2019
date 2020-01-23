import os, sys
sys.path.append("/home/abravo/PycharmProjects/IberLEF2019")
import numpy as np
from sklearn.model_selection import train_test_split

from utils.networks import lstm_simple_binary, plot_model_history, model_testing, cnn_binary_with_emb_layer_char_word, \
    cnn_binary_with_emb_layer_, lstm_simple_binary_attent, model_testing_2inputs
from utils.utils import get_embedding_matrix, load_obj, save_obj, EMBEDDING_CHAR, MAX_SEQUENCE_CHAR_LENGTH, \
    MAX_SEQUENCE_LENGTH, create_all_data

def test_cnn(data_path, labels_path, emb_matrix, nlabels, max_seq):
    print(emb_matrix.shape, nlabels, max_seq)
    lr = 0.0001
    nfilters = 50

    lr = 0.0001
    nfilters = 100

    model = cnn_binary_with_emb_layer_(emb_matrix,lr,nlabels, nfilters, max_seq)

    batch_size = 95  # 55
    epochs = 100

    data = load_obj(data_path)
    labels = load_obj(labels_path)

    #x_train, y_train, x_test, y_test, x_val, y_val = create_all_data(data_path, labels_path, test_perc=0.15,
    #                                                                       val_per=0.1, val_bias=0)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.15)
    #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10)

    class_weight = {0: 1.,
                    1: 1.3}

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.10,
                        shuffle=True,
                        class_weight=class_weight)

    #plot_model_history(history)

    print(model_testing(model, x_test, y_test))

    return model

def test_cnn_words_chars(data_pathw, emb_matrixw, data_pathc, emb_matrixc, labels_path, nlabels):
    print("test_cnn_words_chars")
    lr = 0.001
    nfilters = 64

    model = cnn_binary_with_emb_layer_char_word(emb_matrixw, emb_matrixc, lr, nlabels, nfilters,
                                                MAX_SEQUENCE_LENGTH, MAX_SEQUENCE_CHAR_LENGTH)

    batch_size = 95  # 55
    epochs = 20

    dataw = load_obj(data_pathw)
    datac = load_obj(data_pathc)
    labels = load_obj(labels_path)

    print(labels.shape[0])
    print("WORD LEVEL")

    x_trainw, x_testw, y_trainw, y_testw = train_test_split(dataw, labels, test_size=0.15)

    print("CHAR LEVEL")
    x_trainc, x_testc, y_trainc, y_testc = train_test_split(datac, labels, test_size=0.15)
    class_weight = {0: 1.,
                    1: 1.3}
    history = model.fit([x_trainw, x_trainc], y_trainw,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.10,
                        shuffle=True,
                        class_weight=class_weight)

    #plot_model_history(history)

    print(model_testing_2inputs(model, x_testw, x_testc, y_testw))

    return model

def test_bilstm_words(emb_matrix, data_path, labels_path,max_seq):
    print("test_bilstm_words")
    lr = 0.0001
    model = lstm_simple_binary(emb_matrix, lr, nlabels=2, nunits=50, max_seq=max_seq)

    #model = lstm_simple_binary_attent(embedding_matrix=emb_matrix,lr=lr, nlabels=labels.shape[1], nunits=50, max_seq=max_seq)

    batch_size = 95  # 55
    epochs = 25

    data = load_obj(data_path)
    labels = load_obj(labels_path)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.15)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10)

    print("X_TRAIN:", x_train.shape)
    print("Y_TRAIN:", y_train.shape)
    print("X_TEST:", x_test.shape)
    print("Y_TEST:", y_test.shape)
    print("X_VAL:", x_val.shape)
    print("Y_VAL:", y_val.shape)


    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_val, y_val),
                        shuffle=True)



    #plot_model_history(history)

    print(model_testing(model, x_test, y_test))

    return model

if __name__ == '__main__':
    output_path = "/home/upf/corpora/IberLEF2019/MEX-A3T/preprocessed_data"

    word_index_path = os.path.join(output_path, "word_index_all.pkl")
    word_index = load_obj(word_index_path)

    # WORD LEVEL
    data_path = os.path.join(output_path, "data_train.pkl")
    data = load_obj(data_path)

    w2v_path = "/home/upf/corpora/IberLEF2019/regional_emb/es-MX-100d.vec"
    emb_dim = 100
    emb_matrix = get_embedding_matrix(word_index, emb_dim, w2v_path)


    # CHAR LEVEL
    char_data_path = os.path.join(output_path, "char_data_train.pkl")
    #char_data = load_obj(char_data_path)
    char_index_path = os.path.join(output_path, "char_index_all.pkl")
    char_index = load_obj(char_index_path)
    char_emb_matrix = get_embedding_matrix(char_index, EMBEDDING_CHAR)

    # LABEL
    labels_path = os.path.join(output_path, "labels_train.pkl")
    labels = load_obj(labels_path)




    model = test_bilstm_words(emb_matrix, data_path, labels_path, MAX_SEQUENCE_LENGTH)


    #model= test_cnn(data_path, labels_path, emb_matrix,labels.shape[1], MAX_SEQUENCE_LENGTH)

    #test_cnn(char_data_path, labels_path, char_emb_matrix, labels.shape[1], MAX_SEQUENCE_CHAR_LENGTH)

    #model = test_cnn_words_chars(data_path, emb_matrix, char_data_path, char_emb_matrix, labels_path, labels.shape[1])



    #TEST

    data_test_path = os.path.join(output_path, "data_test.pkl")
    data_test = load_obj(data_test_path)

    char_data_test_path = os.path.join(output_path, "char_data_test.pkl")
    char_data_test = load_obj(char_data_test_path)

    result_path = os.path.join(output_path, "result_a.txt")
    out = open(result_path, "w")
    i=1
    print("data_test.shape",data_test.shape)
    while i < data_test.shape[0]:

        prediction = model.predict(np.array([data_test[i]]))
        #prediction = model.predict([np.array([data_test[i]]), np.array([char_data_test[i]])])

        #prediction = model.predict([np.array([data_test[i]]), np.array([char_data_test[i]])])


        predicted_label = np.argmax(prediction[0])

        # print(ids_test[i] + "\t" + str(label_test_dict[predicted_label]))

        line = []
        line.append("\"aggressiveness\"")
        line.append("\"tweet-"+str(i)+"\"")
        line.append("\"" + str(predicted_label) + "\"")

        out.write("\t".join(line) + "\n")
        i += 1
    out.close()