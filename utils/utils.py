import spacy, os, sys,csv, nltk, re, pickle, html
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from spacymoji import Emoji
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from random import shuffle

MAX_SEQUENCE_LENGTH=50
MAX_SEQUENCE_CHAR_LENGTH=280
URL="URL"
USER="@USUARIO"
EMOJI="@EMOJI_"

EMBEDDING_CHAR = 50
EMBEDDING_WORD = 300



class DataTask:
    def __init__(self, data_path, labels_path=None, test_perc=0.15, val_per=0.1, test_data_path=None, test_labels_path=None):
        self.data_path = data_path
        self.labels_path = labels_path
        self.test_perc = test_perc
        self.val_per = val_per
        self.history = None
        self.model = None
        self.batch_size = None
        self.x_train_batches = []
        self.y_train_batches = []
        self.batches = []
        self.scaler = MinMaxScaler()

        data = load_obj(self.data_path)
        labels = [0]*data.shape[0]
        if labels_path:
            labels = load_obj(self.labels_path)
            if "scores_train" in self.labels_path:
                labels = np.reshape(labels, (-1, 1))
                print("FIIIIIIIIIIIT!!!")
                labels = self.scaler.fit_transform(labels)





        self.x_train = data
        self.y_train = labels
        self.y_test = None
        self.x_val = None
        self.y_val = None

        if test_data_path:

            self.x_test = load_obj(test_data_path)

            if test_labels_path:
                self.y_test = load_obj(test_labels_path)
        elif test_perc:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data, labels, test_size=test_perc)


        if val_per:
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=val_per)


        print("X_TRAIN:", self.x_train.shape)
        #print("Y_TRAIN:", self.y_train.shape)

        if test_perc:
            print("X_TEST:", self.x_test.shape)
        """
        if self.y_test!=None:
            print("Y_TEST:", self.y_test.shape)
        if self.x_val:
            print("X_VAL:", self.x_val.shape)
        if self.y_val:
            print("Y_VAL:", self.y_val.shape)
        """

    def get_scaler(self):
        return self.scaler

    def get_x_train(self):
        return self.x_train

    def get_y_train(self):
        return self.y_train

    def get_x_test(self):
        return self.x_test

    def get_y_test(self):
        return self.y_test

    def get_x_val(self):
        return self.x_val

    def get_y_val(self):
        return self.y_val

    def split_batches(self, batch_size = None):
        self.batch_size = batch_size
        if not batch_size:
            batch_size = self.x_train.shape[0]
        count = 0
        x_train_batches = []
        y_train_batches = []
        while count * batch_size < len(self.y_train):
            x_train_batches.append(self.x_train[batch_size * count:batch_size * (count + 1)])
            y_train_batches.append(self.y_train[batch_size * count:batch_size * (count + 1)])
            count += 1
        self.x_train_batches = x_train_batches
        self.y_train_batches = y_train_batches
        self.batches = list(range(len(self.x_train_batches)))

    def init_batches(self):
        self.batches = list(range(len(self.x_train_batches)))
    def get_batches(self):
        return self.batches

    def get_x_train_batches(self):
        return self.x_train_batches

    def get_y_train_batches(self):
        return self.y_train_batches


    def get_batch_size(self):
        return self.batch_size

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model

    def set_history(self, hist):
        self.history = hist

    def get_history(self):
        return self.history

def split_file_train_test(input_path, output_path, filename, header, random, test_perc=0.15):

    lines = open(input_path).readlines()
    head = ""
    if header:
        head = lines[0]
        lines = lines[1:]

    if random:
        shuffle(lines)

    test_count = int(len(lines) * test_perc)
    train_set = lines[test_count:]

    test_set = lines[:test_count]

    train_path = os.path.join(output_path, filename + "_trainset.txt")
    ofile = open(train_path, "w")
    ofile.write(head)
    ofile.write("".join(train_set))
    ofile.close()

    train_path = os.path.join(output_path, filename + "_devset.txt")
    ofile = open(train_path, "w")
    ofile.write(head)
    ofile.write("".join(test_set))
    ofile.close()


def create_all_data(data_path, labels_path, test_perc=0.15, val_per=0.1, val_bias=0):
    data = load_obj(data_path)
    labels = load_obj(labels_path)

    x_train, y_train, x_test, y_test, x_val, y_val = None, None, None, None, None, None

    if test_perc:
        test_count = int(labels.shape[0] * test_perc)

        x_train = data[test_count:]
        y_train = labels[test_count:]
        x_test = data[:test_count]
        y_test = labels[:test_count]

    if val_per:

        if not test_perc:
            x_train = data
            y_train = labels

        test_count = int(y_train.shape[0] * val_per)

        x_val = x_train[:test_count]
        y_val = y_train[:test_count]

        x_train = x_train[test_count:]
        y_train = y_train[test_count:]

    print("X_TRAIN:",x_train.shape)
    print("Y_TRAIN:",y_train.shape)
    if test_perc:
        print("X_TEST:",x_test.shape)
        print("Y_TEST:",y_test.shape)
    if val_per:
        print("X_VAL:",x_val.shape)
        print("Y_VAL:",y_val.shape)
    print("")

    return x_train, y_train, x_test, y_test, x_val, y_val

def clean_tweet(tweet_text):
    tweet_text = tweet_text.replace('—', " ")  # .replace("'", "’")
    tweet_text = ' '.join(tweet_text.split())
    return tweet_text.strip()

def replace_tweet(tweet_text):
    tweet_text = clean_tweet(tweet_text)
    tweet_text = html.unescape(tweet_text)
    tweet_text = re.sub(r'\.[a-zA-Z]', '. ', tweet_text)
    tweet_text = re.sub(r'[a-zA-Z]\.', ' .', tweet_text)
    tweet_text = re.sub(r'\,[a-zA-Z]', ', ', tweet_text)
    tweet_text = re.sub(r'[a-zA-Z]\,', ' ,', tweet_text)
    #return tweet_text.replace("'", "QUOTE_SYMBOL").replace("‘", "QUOTE_SYMBOL").replace("’", "QUOTE_SYMBOL").replace("-", "HYPH_SYMBOL").replace(";", " ").replace("#", "HASHTAG_SYMBOL")
    return tweet_text.replace("#", " #").replace("•", " • ").replace("/", " / ").replace("(", " ( ").replace(")", " ) ").replace("|", " | ").replace("¡", " ¡ ").replace("¿", " ¿ ").replace("!", " ! ").replace("?", " ? ").replace("\"", " \" ").replace("'", " ' ").replace("‘", " ' ").replace("’", " ' ").replace("-", " - ").replace("–", " - ").replace(";", "; ").replace("#", "HASHTAG_SYMBOL")

def unreplace_tweet(tweet_text):
    #return tweet_text.replace("QUOTE_SYMBOL", "'").replace("HYPH_SYMBOL", "-").replace("HASHTAG_SYMBOL", "#").replace("EMOJI_SYMBOL","#&").lower()
    return tweet_text.replace("HASHTAG_SYMBOL", "#").replace("EMOJI_SYMBOL", "#&")

def write_word_index(word_index, output_path):
    ofile = open(output_path, "w")

    for w in sorted(word_index):
        ofile.write(str(w) + "\t" + str(word_index[w]) + "\n")

    ofile.close()



def get_word_index_from_file(filepath):
    word_index = {}
    for line in open(filepath):
        word = line.strip().split("\t")[0]
        value = int(line.strip().split("\t")[1])
        word_index[word] = value

    return word_index

def get_word_index_from_files(list_files):
    word_set = set()

    for filepath in list_files:
        print(filepath)
        for line in open(filepath):
            word = line.strip().split("\t")[0]
            word_set.add(word)

    word_index = {}
    i = 0
    for tok in word_set:
        i += 1
        word_index[tok] = i

    return word_index


def get_word_index(nlp, list_files, header, tweet_col, char_mode = False):

    word_set = set()

    for filepath in list_files:
        print(filepath)
        for line in open(filepath):
            if header:
                header = False
                continue
            #print(line)
            tweet = line.strip().split("\t")[tweet_col]
            sent = preprocess_tweet(nlp, tweet, char_mode)
            #print(line.strip().split("\t")[0], sent)
            if char_mode:
                for w in sent:
                    if w.startswith(EMOJI):
                        word_set.add(w.replace(EMOJI, ""))
                    else:
                        for c in w:
                            word_set.add(c)
            else:
                for w in sent:
                    word_set.add(w.lower())

    word_index = {}
    i = 0
    for tok in word_set:
        i += 1
        word_index[tok] = i

    return word_index

def save_obj(obj, path):
    f = open(path, 'wb')
    pickle.dump(obj, f)
    f.close()

def load_obj(path):
    f = open(path, "rb")
    obj = pickle.load(f)
    f.close()
    return obj

def get_embedding_matrix(word_index, emb_dim, emb_path = None):

    embedding_matrix = np.random.uniform(-0.8, 0.8, (len(word_index) + 1, emb_dim))

    if emb_path:
        embeddings_index = {}
        f = open(emb_path)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        #for word in sorted(embeddings_index.keys()):
        #    print (word)
    return embedding_matrix


def get_spacy_nlp(core, emojis=True):
    nlp = spacy.load(core)
    if emojis:
        emoji = Emoji(nlp)
        nlp.add_pipe(emoji, first=True)

    return nlp

def preprocess_tweet(nlp, tweet, char_mode=False):
    sent = []
    tweet_text = replace_tweet(tweet)
    doc = nlp(tweet_text)
    for token in doc:
        if token.pos_ == "SPACE":
            #if char_mode:
            #    sent.append(" ")
            continue
        word = unreplace_tweet(str(token))
        if word.startswith("@"):
            word = USER
        elif word.startswith("http"):
            word = URL
        elif token._.is_emoji:
            word = EMOJI + word
        sent.append(word)
        if char_mode:
            sent.append(" ")
    if char_mode:
        sent = sent[:-1]
    return sent



def create_data_comb(file_list,header, output_path):

    ofile = open(output_path, "w")

    contain = []
    ini=0
    if header:
        ini=1
    for fl in file_list:
        contain += open(fl).readlines()[ini:]

    if header:
        header_text = open(file_list[0]).readline()
        contain = [header_text] + contain


    ofile.write("".join(contain))

    ofile.close()

def get_task_data_for_class_task(nlp, word_index, filepath, header, random, tweet_col, label_col=None, char_mode= False):
    tweets = []
    labels = []

    lines = open(filepath).readlines()
    """
    if char_mode:
        ofile = open(filepath + "_char.log", "w")
    else:
        ofile = open(filepath + ".log", "w")
    """
    if header:
        lines = lines[1:]

    if random:
        shuffle(lines)
        shuffle(lines)
        shuffle(lines)

    for line in lines:
        sent = []
        ls = line.strip().split("\t")
        tweet = ls[tweet_col]
        label = "?"
        if label_col != None:
            label = ls[label_col]
            labels.append(label)
        sent_pre = preprocess_tweet(nlp, tweet, char_mode)
        #print(line.strip().split("\t"))

        if char_mode:
            for w in sent_pre:
                if w.startswith(EMOJI):
                    sent.append(w.replace(EMOJI, ""))
                else:
                    for c in w:
                        sent.append(c)
        else:
            for w in sent_pre:
                sent.append(w.lower())
        #ofile.write(str(sent) + "\t" + label + "\n")
        tweets.append(sent)

    sequences = []
    for sent in tweets:
        seq = []
        for word in sent:
            i = word_index[word]#, 0)
            seq.append(i)
        sequences.append(seq)
    max_len = MAX_SEQUENCE_LENGTH
    if char_mode:
        max_len =MAX_SEQUENCE_CHAR_LENGTH
    data = pad_sequences(sequences, maxlen=max_len)

    if label_col != None:
        labels = to_categorical(np.asarray(labels))

    #ofile.close()
    return data, labels

def get_task_data_for_class_task_haha(nlp, word_index, filepath, header, random, tweet_col, label_col=None, char_mode= False):
    tweets = []
    labels = []
    scores = []

    lines = open(filepath).readlines()
    """
    if char_mode:
        ofile = open(filepath + "_char.log", "w")
    else:
        ofile = open(filepath + ".log", "w")
    """
    if header:
        lines = lines[1:]

    if random:
        shuffle(lines)
        shuffle(lines)
        shuffle(lines)

    for line in lines:
        sent = []
        ls = line.strip().split("\t")
        tweet = ls[tweet_col]
        label = "?"
        score = "?"
        if label_col != None:
            label = ls[label_col[0]]
            labels.append(label)
            label = ls[label_col[1]]
            if label == "NULL":
                label = "0.0"
            scores.append(float(label))
        sent_pre = preprocess_tweet(nlp, tweet, char_mode)
        #print(line.strip().split("\t"))

        if char_mode:
            for w in sent_pre:
                if w.startswith(EMOJI):
                    sent.append(w.replace(EMOJI, ""))
                else:
                    for c in w:
                        sent.append(c)
        else:
            for w in sent_pre:
                sent.append(w.lower())
        #ofile.write(str(sent) + "\t" + label + "\n")
        tweets.append(sent)

    sequences = []
    for sent in tweets:
        seq = []
        for word in sent:
            i = word_index[word]#, 0)
            seq.append(i)
        sequences.append(seq)
    max_len = MAX_SEQUENCE_LENGTH
    if char_mode:
        max_len =MAX_SEQUENCE_CHAR_LENGTH
    data = pad_sequences(sequences, maxlen=max_len)
    scores_scaled = None
    if label_col != None:
        labels = to_categorical(np.asarray(labels))
        """
        y = np.reshape(y, (-1, 1))
        scaler = MinMaxScaler()
        print(scaler.fit(x))
        print(scaler.fit(y))
        xscale = scaler.transform(x)
        yscale = scaler.transform(y)

        scores_reshape = np.reshape(scores, (-1, 1))

        scaler = MinMaxScaler()
        scores_scaled = scaler.fit_transform(scores)
        i=0
        while i < len(scores):
            print(scores[i], scores_scaled[i])
            i+=1
        """

    return data, labels, scores