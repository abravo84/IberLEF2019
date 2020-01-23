import os, sys

from sklearn.preprocessing import MinMaxScaler

sys.path.append("/home/abravo/PycharmProjects/IberLEF2019")

from utils.networks import lstm_simple_binary, plot_model_history, model_testing, lstm_haha, model_testing_haha, \
    cnn_haha
from utils.utils import get_embedding_matrix, load_obj, save_obj
import numpy as np

if __name__ == '__main__':
    output_path = "/home/abravo/corpora/IberLEF2019/HAHA/preprocessed_data"
    #emb_matrix_filename = "emb_matrix_"
    #emb_matrix_filename = "emb_matrix_fb_"
    #emb_matrix_path = os.path.join(output_path, emb_matrix_filename + ".pkl")
    #emb_matrix = load_obj(emb_matrix_path)



    word_index_path = os.path.join(output_path, "word_index_all.pkl")
    word_index = load_obj(word_index_path)
    dim = 100
    w2v_path = "/home/abravo/corpora/IberLEF2019/regional_emb/es-MX-100d.vec"

    #dim = 300
    #w2v_path = "/home/abravo/corpora/IberLEF2019/regional_emb/model_swm_300-6-10-low_es.w2v"
    emb_matrix = get_embedding_matrix(word_index, dim, w2v_path)
    emb_matrix_path = os.path.join(output_path, "emb_matrix" + ".pkl")
    save_obj(emb_matrix, emb_matrix_path)



    MODE_BOTH = True


    if MODE_BOTH:

        data_path = os.path.join(output_path, "data_train.pkl")
        data = load_obj(data_path)

        labels_path = os.path.join(output_path, "labels_train.pkl")
        labels = load_obj(labels_path)

        scores_path = os.path.join(output_path, "scores_train.pkl")
        scores = load_obj(scores_path)



        scores = np.reshape(scores, (-1, 1))
        scaler = MinMaxScaler()
        scores = scaler.fit_transform(scores)

        #scores = []
        #for s in scores_scaled:
        #    scores.append(s[0])
        #    print (s, s[0])


        print("dasdasda", scores.shape, data.shape, labels.shape)

        lr = 0.001
        model = lstm_haha(emb_matrix_path,
                          lr=lr, nlabels=2, nunits=75, max_seq=50, trainable=True)


        MODE_TEST = False


        batch_size = 255
        epochs = 100
        print(labels.shape[1])

        print("SHAPE", data.shape)


        if MODE_TEST:
            x_train = data
            y_train = labels
            y_score = scores
            history = model.fit(x_train, [y_train, y_score],
                                batch_size=240,
                                epochs=epochs,
                                shuffle=True)
        else:
            test_count = int(labels.shape[0] * 0.15)
            x_train = data[test_count:]
            y_train = labels[test_count:]
            y_score = scores[test_count:]

            x_val = data[:test_count]
            y_val = labels[:test_count]
            y_score_val = scores[:test_count]


            history = model.fit(x_train, [y_train,y_score],
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_split=0.10,
                                shuffle=True)

            #plot_model_history(history)

            #model_testing(model, x_val, y_val)

            model_testing_haha(model, x_val, y_val, y_score_val, scaler)


        header = True

        data_path = os.path.join(output_path, "haha_2019_test.txt")
        ids_list = []
        for line in open(data_path):

            if header:
                header = False
                continue

            id = line.strip().split("\t")[0]
            ids_list.append(id)



        data_path = os.path.join(output_path, "data_test.pkl")
        data_test = load_obj(data_path)

        output_path = os.path.join(output_path, "haha_prediction_both.cvs")
        out = open(output_path, "w")

        i = 0
        while i < data_test.shape[0]:
            prediction = model.predict(np.array([data_test[i]]))

            predicted_label = np.argmax(prediction[0])
            predicted_score= scaler.inverse_transform(prediction[1])

            #print(ids_list[i], prediction, predicted_score)

            line = []

            line.append(ids_list[i])
            line.append(str(predicted_label))
            score = "0.0"
            if str(predicted_label) == "1":
                score = str(predicted_score[0][0])

            line.append(score)

            out.write(",".join(line) + "\n")
            i += 1

        out.close()




    else:

        data_path = os.path.join(output_path, "data_train.pkl")
        data = load_obj(data_path)

        labels_path = os.path.join(output_path, "labels_train.pkl")
        labels = load_obj(labels_path)

        lr = 0.0001
        model = lstm_simple_binary(emb_matrix, lr)




        batch_size = 255
        epochs = 100
        print(labels.shape[1])
        test_count = int(labels.shape[0] * 0.15)
        print("SHAPE", data.shape)
        x_train = data[test_count:]
        y_train = labels[test_count:]
        x_val = data[:test_count]
        y_val = labels[:test_count]

        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.10,
                            shuffle=True)

        #plot_model_history(history)

        model_testing(model, x_val, y_val)



        header = True

        data_path = os.path.join(output_path, "haha_2019_test.txt")
        ids_list = []
        for line in open(data_path):

            if header:
                header = False
                continue

            id = line.strip().split("\t")[0]
            ids_list.append(id)




        data_path = os.path.join(output_path, "data_test.pkl")

        data_test = load_obj(data_path)

        output_path = os.path.join(output_path, "haha_prediction.csv")
        out = open(output_path, "w")

        out.write("id,is_humor,funniness_average\n")

        print(len(ids_list), data_test.shape)

        i = 0
        while i < data_test.shape[0]:
            prediction = model.predict(np.array([data_test[i]]))
            predicted_label = np.argmax(prediction[0])

            line = []

            line.append(ids_list[i])
            line.append(str(predicted_label))
            line.append(str(0.0))

            out.write(",".join(line) + "\n")
            i += 1

        out.close()