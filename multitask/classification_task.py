import os, random, sys
import numpy as np
sys.path.append("/home/abravo/PycharmProjects/IberLEF2019")
from utils.networks import get_embedding_layer, lstm_simple_binary_with_emb_layer, get_bilstm, model_testing, \
    get_embedding_layer_from_matrix, get_convolutions, get_CNN_last_part, get_dense_layer, \
    lstm_simple_binary_all, get_embedded_sequences, get_input_layer, lstm_simple_binary_with_emb_layer_att
from utils.utils import load_obj, create_all_data, get_embedding_matrix, DataTask, get_word_index_from_files, save_obj, \
    MAX_SEQUENCE_LENGTH, MAX_SEQUENCE_CHAR_LENGTH
from random import randint
HAHA_TASK = 0
IROSVA_TASK = 1
MEXAT_TASK = 2
TASS_TASK = 3



MODE_LSTM = 0
MODE_WORD_CNN = 1
MODE_CHAR_CNN = 2
MODE_WORD_CHAR_CNN = 3

if __name__ == '__main__':

    output_path = "/home/upf/corpora/IberLEF2019/multitask"

    irosva_path = "/home/upf/corpora/IberLEF2019/IroSva/preprocessed_data"
    mexa3t_path = "/home/upf/corpora/IberLEF2019/MEX-A3T/preprocessed_data"
    haha_path = "/home/upf/corpora/IberLEF2019/HAHA/preprocessed_data"
    tass_path = "/home/upf/corpora/IberLEF2019/TASS/preprocessed_data"

    output_path = "/home/abravo/corpora/IberLEF2019/multitask"

    irosva_path = "/home/abravo/corpora/IberLEF2019/IroSva/preprocessed_data"
    mexa3t_path = "/home/abravo/corpora/IberLEF2019/MEX-A3T/preprocessed_data"
    haha_path = "/home/abravo/corpora/IberLEF2019/HAHA/preprocessed_data"
    tass_path = "/home/abravo/corpora/IberLEF2019/TASS/preprocessed_data"

    MODE_TEST = True

    IROSVA_on = True
    MEXAT_on = True
    HAHA_on = True
    TASS_on = True
    print("MODE_TEST:", MODE_TEST)

    tasks = ["irosva", "haha", "mexa3t"]

    tag = "_".join(sorted(tasks))

    tass_languages = ["es", "mx", "cr", "uy", "pe"]
    tass_languages = ["mx"]

    # TASS DATA
    tass_data = {}

    char_data = ""
    for k in tass_languages:  # ["all","es", "mx", "cu"]:
        print("TASS " + k)
        data_path = os.path.join(tass_path, char_data + "data_" + k + "_train.pkl")
        labels_path = os.path.join(tass_path, "labels_" + k + "_train.pkl")
        test_data_path = os.path.join(tass_path, char_data + "data_" + k + "_dev.pkl")
        test_labels_path = os.path.join(tass_path, "labels_" + k + "_dev.pkl")
        if MODE_TEST:
            id = DataTask(data_path, labels_path, 0, 0, test_data_path, test_labels_path)
        else:
            id = DataTask(data_path, labels_path, 0, 0.1, test_data_path, test_labels_path)
        tass_data[k] = id




    irosva_languages = ["es", "mx", "cu"]


    #IROSVA DATA
    irosva_data= {}

    char_data = ""
    for k in irosva_languages:#["all","es", "mx", "cu"]:
        print("IROSVA "+ k)
        data_path = os.path.join(irosva_path, char_data + "data_" + k + "_train.pkl")
        labels_path = os.path.join(irosva_path,"labels_" + k + "_train.pkl")
        if MODE_TEST:
            test_data_path = os.path.join(irosva_path, char_data + "data_" + k + "_test.pkl")
            id = DataTask(data_path, labels_path, 0.15, None, test_data_path, None)
            #id = DataTask(data_path, labels_path, 0, 0)
        else:
            id = DataTask(data_path, labels_path, test_perc=0, val_per=0.10)
        irosva_data[k] = id

    #HAHA DATA
    print("HAHA")
    data_path = os.path.join(haha_path, char_data + "data_train.pkl")
    labels_path = os.path.join(haha_path, "labels_train.pkl")

    haha_data = {}

    #if MODE_TEST:
    #    test_data_path = os.path.join(haha_path, char_data + "data_test.pkl")
    #    id = DataTask(data_path, labels_path, 0.15, None, test_data_path, None)
    #else:
    id = DataTask(data_path, labels_path, test_perc=0, val_per=0)
    haha_data["mx"] = id

    # MEXA3T DATA
    print("MEXA3T")
    data_path = os.path.join(mexa3t_path, char_data + "data_train.pkl")
    labels_path = os.path.join(mexa3t_path, "labels_train.pkl")
    mexa3t_data = {}

    if MODE_TEST:
        test_data_path = os.path.join(mexa3t_path, char_data + "data_test.pkl")
        id = DataTask(data_path, labels_path, 0.15, None, test_data_path, None)
    else:
        id=DataTask(data_path, labels_path)
    mexa3t_data["mx"] = id

    lang = "mx"
    mode = MODE_LSTM



    #irosva_x_train_batches, irosva_y_train_batches = irosva_data[lang].get_batches(102)
    #haha_x_train_batches, haha_y_train_batches = haha_data[lang].get_batches(1020)
    #mexa3t_x_train_batches, mexa3t_y_train_batches = mexa3t_data[lang].get_batches(310)

    for k in tass_languages:
        tass_data[k].split_batches(254*len(tass_languages))

    for k in irosva_languages:
        b= 102
        if MODE_TEST:
            b=240
        irosva_data[k].split_batches(b*len(irosva_languages))

    b = 1020
    if MODE_TEST:
        b = 2400
    haha_data[lang].split_batches(b)

    b=310
    if MODE_TEST:
        b=770

    mexa3t_data[lang].split_batches(b)



    print(len(irosva_data[lang].get_x_train_batches()),
          len(tass_data[lang].get_x_train_batches()),
          len(haha_data[lang].get_x_train_batches()),
          len(mexa3t_data[lang].get_x_train_batches()))

    #word_index_all_path = os.path.join(output_path, "word_index_" + tag + ".pkl")
    #word_index = load_obj(word_index_all_path)
    word_index_files = []
    if len(char_data):
        word_index_files.append(os.path.join(irosva_path, "char_index_all.txt"))
        word_index_files.append(os.path.join(haha_path, "char_index_train.txt"))
        word_index_files.append(os.path.join(mexa3t_path, "char_index_all.txt"))
        word_index_files.append(os.path.join(tass_path, "char_index_all.txt"))
    else:
        word_index_files.append(os.path.join(irosva_path, "word_index_all.txt"))
        word_index_files.append(os.path.join(haha_path, "word_index_train.txt"))
        word_index_files.append(os.path.join(mexa3t_path, "word_index_all.txt"))
        word_index_files.append(os.path.join(tass_path, "char_index_all.txt"))

    word_index = get_word_index_from_files(word_index_files)

    w2v_path_dict = {}
    w2v_path_dict["mx"] = "/home/abravo/corpora/IberLEF2019/regional_emb/es-MX-100d.vec"
    w2v_path_dict["cu"] = "/home/abravo/corpora/IberLEF2019/regional_emb/es-CU-100d.vec"
    w2v_path_dict["es"] = "/home/abravo/corpora/IberLEF2019/regional_emb/es-ES-100d.vec"
    w2v_path_dict["cr"] = "/home/abravo/corpora/IberLEF2019/regional_emb/es-CR-100d.vec"
    w2v_path_dict["uy"] = "/home/abravo/corpora/IberLEF2019/regional_emb/es-UY-100d.vec"
    w2v_path_dict["pe"] = "/home/abravo/corpora/IberLEF2019/regional_emb/es-PE-100d.vec"
    emb_dim = 100

    if len(char_data):
        emb_dim = 50
    emb_matrix_dict = {}
    for k in irosva_languages+tass_languages:
        if k in emb_matrix_dict:
            continue
        file_windex = None
        if not len(char_data):
            file_windex = w2v_path_dict[k]
        emb_matrix_dict[k] = get_embedding_matrix(word_index, emb_dim, file_windex)

    max_seq = MAX_SEQUENCE_LENGTH
    if len(char_data):
        max_seq = MAX_SEQUENCE_CHAR_LENGTH

    if mode == MODE_WORD_CNN:

        sequence_input = get_input_layer(max_seq)
        embedded_sequences = {}
        for k in irosva_languages:

            embedded_sequences[k] = get_embedded_sequences(embedding_matrix=emb_matrix_dict[lang],
                                                            sequence_input=sequence_input,
                                                            max_seq=max_seq,
                                                            trainable=True)

        alpha =get_convolutions(embedded_sequences=embedded_sequences,
                                embedding_matrix=emb_matrix_dict[lang],
                                max_seq=max_seq,
                                nfilters=100,
                                grams=[5, 4, 3, 2])

        sequence_input, alpha=get_convolutions(embedding_matrix=emb_matrix_dict[lang],
                                               max_seq=MAX_SEQUENCE_LENGTH,
                                               nfilters=50,
                                               grams=[5, 4, 3, 2],
                                               trainable=True)


        irosva_data[lang].set_model(get_CNN_last_part(sequence_input, alpha,
                                                      nlabels=2,
                                                      dense_nodes=100))
        haha_data[lang].set_model(get_CNN_last_part(sequence_input, alpha,
                                                    nlabels=2,
                                                    dense_nodes=100))
        mexa3t_data[lang].set_model(get_CNN_last_part(sequence_input,alpha,
                                                      nlabels=2,
                                                      dense_nodes=100))

    elif mode == MODE_LSTM:
        embedding_layer = get_embedding_layer_from_matrix(embedding_matrix=emb_matrix_dict[lang],
                                                          trainable=True,
                                                          max_seq=max_seq)
        bilstm = get_bilstm(100)
        dense1 = get_dense_layer(50, activation='relu')
        dense2 = get_dense_layer(2)


        #bilstm = None


        if IROSVA_on:
            for k in irosva_languages:
                print("MODEL: "+ k)
                embedding_layer_k = get_embedding_layer_from_matrix(embedding_matrix=emb_matrix_dict[k],
                                                                  trainable=True,
                                                                  max_seq=max_seq)

                #irosva_data[k].set_model(lstm_simple_binary_all(embedding_layer_k, bilstm, [dense2], lr=0.0001, do=0.5))
                irosva_data[k].set_model(lstm_simple_binary_with_emb_layer_att(embedding_layer_k, lr=0.001, bilstm=bilstm, do=0.5, units=100, nlabels=2))

        if TASS_on:
            for k in tass_languages:
                print("MODEL: "+ k)
                embedding_layer_k = get_embedding_layer_from_matrix(embedding_matrix=emb_matrix_dict[k],
                                                                  trainable=True,
                                                                  max_seq=max_seq)

                #tass_data[k].set_model(lstm_simple_binary_all(embedding_layer_k, bilstm, [dense2], lr=0.0001, do=0.5))
                tass_data[k].set_model(lstm_simple_binary_with_emb_layer_att(embedding_layer_k, lr=0.001, bilstm=bilstm, do=0.5, units=100, nlabels=4))

        if HAHA_on:
            haha_data[lang].set_model(lstm_simple_binary_with_emb_layer_att(embedding_layer, lr=0.001, bilstm=bilstm, do=0.5, units=100, nlabels=2))
        if MEXAT_on:
            mexa3t_data[lang].set_model(lstm_simple_binary_with_emb_layer_att(embedding_layer, lr=0.001, bilstm=bilstm, do=0.5, units=100, nlabels=2))

    ofile = open(os.path.join(output_path, "bilstm_haha_irosva_classw_relu_100.txt"), "w")

    EPOCHS = 20
    for epoch in range(0,EPOCHS):
        print("#############################################")
        print("EPOCH:", epoch)
        print("#############################################")

        ofile.write("EPOCH: "+str(epoch)+"\n")
        ofile.write("---------------------------------------------\n")



        for k in irosva_languages:
            irosva_data[k].init_batches()
        for k in tass_languages:
            tass_data[k].init_batches()
        haha_data[lang].init_batches()
        mexa3t_data[lang].init_batches()




        total_batches = []
        if IROSVA_on:
            for k in irosva_languages:
                total_batches += irosva_data[k].get_batches()
        if TASS_on:
            for k in tass_languages:
                total_batches += tass_data[k].get_batches()
        if MEXAT_on:
            total_batches += mexa3t_data[lang].get_batches()
        if HAHA_on:
            total_batches += haha_data[lang].get_batches()



        while len(total_batches):

            class_weight = {0: 1.,
                            1: 1.3}

            task_rnd = randint(0,3)
            #task_rnd = IROSVA_TASK

            if task_rnd == IROSVA_TASK:# and len(irosva_batches):
                if not IROSVA_on:
                    continue
                total_irosva = []
                for k in irosva_languages:
                    total_irosva += irosva_data[k].get_batches()

                if not len(total_irosva):
                    continue

                k = random.choice(irosva_languages)
                #print("iros languages", k)
                while not len(irosva_data[k].get_batches()):
                    k = random.choice(irosva_languages)
                    #print("iros languages", k)
                print("Iros languages", k)
                batch = random.choice(irosva_data[k].get_batches())
                irosva_data[k].get_batches().remove(batch)
                #print("irosva_batches:", len(irosva_data[k].get_batches()))
                #print("IROSVA " + k, len(irosva_data[k].get_batches()))
                x_train = irosva_data[k].get_x_train_batches()[batch]
                y_train = irosva_data[k].get_y_train_batches()[batch]

                if not MODE_TEST:
                    x_val = irosva_data[k].get_x_val()
                    y_val = irosva_data[k].get_y_val()
                model = irosva_data[k].get_model()

            elif task_rnd == TASS_TASK:# and len(irosva_batches):
                if not TASS_on:
                    continue

                class_weight = {0: 1.,
                                1: 1.,
                                2: 1.,
                                3: 1.}
                #continue
                total_tass = []
                for k in tass_languages:
                    total_tass += tass_data[k].get_batches()

                if not len(total_tass):
                    continue

                k = random.choice(tass_languages)
                #print("iros languages", k)
                while not len(tass_data[k].get_batches()):
                    k = random.choice(tass_languages)
                    #print("iros languages", k)
                print("TASSS languages", k)
                batch = random.choice(tass_data[k].get_batches())
                tass_data[k].get_batches().remove(batch)
                #print("irosva_batches:", len(irosva_data[k].get_batches()))
                #print("IROSVA " + k, len(irosva_data[k].get_batches()))
                x_train = tass_data[k].get_x_train_batches()[batch]
                y_train = tass_data[k].get_y_train_batches()[batch]
                if not MODE_TEST:
                    x_val = tass_data[k].get_x_val()
                    y_val = tass_data[k].get_y_val()
                model = tass_data[k].get_model()



            elif task_rnd == HAHA_TASK and len(haha_data[lang].get_batches()):
                if not HAHA_on:
                    continue
                batch = random.choice(haha_data[lang].get_batches())
                haha_data[lang].get_batches().remove(batch)
                #print("haha_batches:", len(haha_data[lang].get_x_train_batches()))
                x_train = haha_data[lang].get_x_train_batches()[batch]
                y_train = haha_data[lang].get_y_train_batches()[batch]
                if not MODE_TEST:
                    x_val = haha_data[lang].get_x_val()
                    y_val = haha_data[lang].get_y_val()
                model = haha_data[lang].get_model()
                print("##### HAHA_TASK #####")# + lang, len(haha_data[lang].get_batches()))



            elif task_rnd == MEXAT_TASK and len(mexa3t_data[lang].get_batches()):
                if not MEXAT_on:
                    continue
                batch = random.choice(mexa3t_data[lang].get_batches())
                mexa3t_data[lang].get_batches().remove(batch)
                #print("mex_batches:", len(mexa3t_data[lang].get_batches()))
                x_train = mexa3t_data[lang].get_x_train_batches()[batch]
                y_train = mexa3t_data[lang].get_y_train_batches()[batch]
                if not MODE_TEST:
                    x_val = mexa3t_data[lang].get_x_val()
                    y_val = mexa3t_data[lang].get_y_val()
                model = mexa3t_data[lang].get_model()
                print("##### MEXAT_TASK #####")# + lang, len(mexa3t_data[lang].get_batches()))
            else:
                #print("TASK ERROR!", task_rnd, len(haha_batches), len(irosva_batches), len(mex_batches))

                continue


            if MODE_TEST:
                history = model.fit(x_train, y_train,
                                batch_size=x_train.shape[0],
                                epochs=1,
                                shuffle=True,
                                verbose=1,
                                class_weight=class_weight)
            else:
                history = model.fit(x_train, y_train,
                                    batch_size=x_train.shape[0],
                                    epochs=1,
                                    validation_data=(x_val, y_val),
                                    shuffle=True,
                                    verbose=1,
                                    class_weight=class_weight)

            total_batches = []
            if IROSVA_on:
                for k in irosva_languages:
                    total_batches += irosva_data[k].get_batches()
            if TASS_on:
                for k in tass_languages:
                    total_batches += tass_data[k].get_batches()
            if MEXAT_on:
                total_batches += mexa3t_data[lang].get_batches()
            if HAHA_on:
                total_batches += haha_data[lang].get_batches()

            print("total_batches", len(total_batches))


        if MODE_TEST:
            continue
        """
        if IROSVA_on:
            for k in irosva_languages:
                ofile.write("IROSVA "+k.upper()+" Eval:\n")
                ofile.write(model_testing(irosva_data[k].get_model(), irosva_data[k].get_x_test(),
                                          irosva_data[k].get_y_test()) + "\n")
        if TASS_on:
            for k in tass_languages:
                ofile.write("TASS "+k.upper()+" Eval:\n")
                ofile.write(model_testing(tass_data[k].get_model(), tass_data[k].get_x_test(),
                                          tass_data[k].get_y_test()) + "\n")

        ofile.write("---------------------------------------------\n")

        if HAHA_on:
            ofile.write("HAHA Eval:\n")
            ofile.write(model_testing(haha_data[lang].get_model(), haha_data[lang].get_x_test(),
                                      haha_data[lang].get_y_test()) + "\n")
        ofile.write("---------------------------------------------\n")
        """
        if MEXAT_on:
            ofile.write("MEX Eval:\n")
            ofile.write(model_testing(mexa3t_data[lang].get_model(), mexa3t_data[lang].get_x_test(),
                                      mexa3t_data[lang].get_y_test()) + "\n")

        ofile.write("\n\n#####################################\n\n")
        ofile.flush()



    ofile.close()

    if not MODE_TEST:
        sys.exit()

    #IROSVA

    irosva_languages = ["es", "mx", "cu"]

    for k_train in irosva_languages:

        irosva_test = list(irosva_languages)
        irosva_test.remove(k_train)

        for k in irosva_test:

            print("TESTING IROSVA " + k + " MODEL " + k_train)

            data_test = irosva_data[k].get_x_test()
            model = irosva_data[k_train].get_model()

            ids_path = os.path.join(irosva_path, "irosva." + k + ".test.txt")
            header = 1
            ides_list = []
            for line in open(ids_path):
                if header:
                    header = False
                    continue
                ides_list.append(line.strip().split("\t")[0])

            result_path = os.path.join(output_path, "irosva_" + k + "_result.txt")

            result_path = os.path.join(output_path, "LASTUS-UPF_method2_" + k_train + "_" + k + ".txt")

            out = open(result_path, "w")
            out.write("message_id,irony\n")
            i = 1
            print("data_test.shape", data_test.shape)
            while i < data_test.shape[0]:
                print("TWEET: " + ides_list[i])
                prediction = model.predict(np.array([data_test[i]]))
                # prediction = model.predict([np.array([data_test[i]]), np.array([char_data_test[i]])])

                # prediction = model.predict([np.array([data_test[i]]), np.array([char_data_test[i]])])

                predicted_label = np.argmax(prediction[0])

                # print(ids_test[i] + "\t" + str(label_test_dict[predicted_label]))

                line = []
                line.append(ides_list[i])
                line.append(str(predicted_label))

                out.write(",".join(line) + "\n")
                i += 1
            out.close()

    """
    for k in irosva_languages:
    
        print("TESTING IROSVA " + k)

        data_test = irosva_data[k].get_x_test()
        model = irosva_data[k].get_model()

        ids_path =  os.path.join(irosva_path, "irosva."+k+".test.txt")
        header = 1
        ides_list = []
        for line in open(ids_path):
            if header:
                header = False
                continue
            ides_list.append(line.strip().split("\t")[0])


        result_path = os.path.join(output_path, "irosva_"+k+"_result.txt")
        out = open(result_path, "w")
        out.write("message_id,irony\n")
        i = 1
        print("data_test.shape", data_test.shape)
        while i < data_test.shape[0]:

            print("TWEET: " + ides_list[i])
            prediction = model.predict(np.array([data_test[i]]))
            # prediction = model.predict([np.array([data_test[i]]), np.array([char_data_test[i]])])

            # prediction = model.predict([np.array([data_test[i]]), np.array([char_data_test[i]])])

            predicted_label = np.argmax(prediction[0])

            # print(ids_test[i] + "\t" + str(label_test_dict[predicted_label]))

            line = []
            line.append(ides_list[i])
            line.append(str(predicted_label))

            out.write(",".join(line) + "\n")
            i += 1
        out.close()
    

    data_test_path = os.path.join(mexa3t_path, "data_test.pkl")
    data_test = load_obj(data_test_path)

    result_path = os.path.join(output_path, "MEX_ALL_with_atten.txt")
    out = open(result_path, "w")
    i = 1
    print("data_test.shape", data_test.shape)
    while i < data_test.shape[0]:
        prediction = mexa3t_data[lang].get_model().predict(np.array([data_test[i]]))
        # prediction = model.predict([np.array([data_test[i]]), np.array([char_data_test[i]])])

        # prediction = model.predict([np.array([data_test[i]]), np.array([char_data_test[i]])])

        predicted_label = np.argmax(prediction[0])

        # print(ids_test[i] + "\t" + str(label_test_dict[predicted_label]))

        line = []
        line.append("\"aggressiveness\"")
        line.append("\"tweet-" + str(i) + "\"")
        line.append("\"" + str(predicted_label) + "\"")

        out.write("\t".join(line) + "\n")
        i += 1
    out.close()
    """