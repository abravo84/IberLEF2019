import spacy,csv, nltk, re,sys, pickle
from keras.optimizers import Adam
from keras.regularizers import L1L2
from spacymoji import Emoji
from nltk.util import trigrams, bigrams
from nltk.tokenize import RegexpTokenizer
from numpy import array
from keras.preprocessing.text import one_hot
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Bidirectional, Input, Conv1D, Flatten, \
    MaxPooling1D, GRU, SpatialDropout1D, merge, concatenate, Concatenate, Reshape, Convolution2D, MaxPooling2D, \
    GlobalMaxPooling1D, Conv2D, MaxPool2D, CuDNNLSTM
from keras import optimizers, regularizers, constraints
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time, os
from keras.engine.topology import Layer, InputSpec
from keras import initializers
import html
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error

from utils.utils import MAX_SEQUENCE_LENGTH, load_obj


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average attention mechanism from:
        Zhou, Peng, Wei Shi, Jun Tian, Zhenyu Qi, Bingchen Li, Hongwei Hao and Bo Xu.
        “Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification.”
        ACL (2016). http://www.aclweb.org/anthology/P16-2034
    How to use:
    see: [BLOGPOST]
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.w = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_w'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.w]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, h, mask=None):
        h_shape = K.shape(h)
        d_w, T = h_shape[0], h_shape[1]

        logits = K.dot(h, self.w)  # w^T h
        logits = K.reshape(logits, (d_w, T))
        alpha = K.exp(logits - K.max(logits, axis=-1, keepdims=True))  # exp

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            alpha = alpha * mask
        alpha = alpha / K.sum(alpha, axis=1, keepdims=True)  # softmax
        r = K.sum(h * K.expand_dims(alpha), axis=1)  # r = h*alpha^T
        h_star = K.tanh(r)  # h^* = tanh(r)
        if self.return_attention:
            return [h_star, alpha]
        return h_star

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None


def plot_model_history(model_history, acc = "acc", loss = "loss", val_acc="val_acc", val_loss="val_loss"):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history[acc])+1),model_history.history[acc])
    axs[0].plot(range(1,len(model_history.history[val_acc])+1),model_history.history[val_acc])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history[acc])+1),len(model_history.history[acc])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history[loss])+1),model_history.history[loss])
    axs[1].plot(range(1,len(model_history.history[val_loss])+1),model_history.history[val_loss])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history[loss])+1),len(model_history.history[loss])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()


def get_bilstm(units=64):

    #return Bidirectional(LSTM(units))
    return Bidirectional(LSTM(units=units, return_sequences=True, recurrent_regularizer=L1L2(l1=0.01, l2=0.01)))


def get_dense_layer(nlabes=2, activation='softmax'):

    return Dense(nlabes, activation=activation)

def get_embedding_layer(emb_matrix_path, trainable = False, max_seq = MAX_SEQUENCE_LENGTH):
    embedding_matrix = load_obj(emb_matrix_path)

    embedding_layer = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                input_length=max_seq,
                                trainable=trainable)

    return embedding_layer


def get_embedding_layer_from_matrix(embedding_matrix, trainable = False, max_seq = MAX_SEQUENCE_LENGTH):


    embedding_layer = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                input_length=max_seq,
                                trainable=trainable)

    return embedding_layer




def multiple_embeddings(word_emb_lay, pos_emb_lay, lr):
    # opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    opt = optimizers.Adam(lr=lr)
    model_tok = Sequential()
    model_tok.add(word_emb_lay)
    model_tok.add(Bidirectional(LSTM(100)))
    model_tok.add(Dropout(0.5))
    #model_tok.add(Dense(50, activation='softmax'))

    model_pos = Sequential()
    model_pos.add(pos_emb_lay)
    model_pos.add(Bidirectional(LSTM(30)))
    model_pos.add(Dropout(0.5))
    #model_pos.add(Dense(50, activation='softmax'))

    mergedOut = Concatenate()([model_tok.output,model_pos.output])
    lab_dir = Dense(3, activation='softmax')(mergedOut)
    lab_rel_type = Dense(14, activation='softmax')(mergedOut)

    model = Model([model_tok.input, model_pos.input], [lab_dir,lab_rel_type])


    print(model.summary())
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def token_bba(word_emb_path, pos_emb_path, lr, units1=128, units2=0, dp=0.5, max_seq=50):
    # opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    opt = optimizers.Adam(lr=lr)

    word_emb_lay = get_embedding_layer(word_emb_path, trainable=False, max_seq=max_seq)

    model_tok_bef = Sequential()
    model_tok_bef.add(word_emb_lay)
    model_tok_bef.add(Bidirectional(LSTM(units1, return_sequences=True)))
    if units2:
        model_tok_bef.add(Bidirectional(LSTM(units2, return_sequences=True)))
    model_tok_bef.add(AttentionWithContext())
    model_tok_bef.add(Dropout(dp))



    word_emb_lay = get_embedding_layer(word_emb_path, trainable=True, max_seq=max_seq)
    model_tok_bet = Sequential()
    model_tok_bet.add(word_emb_lay)
    model_tok_bet.add(Bidirectional(LSTM(units1, return_sequences=True)))
    if units2:
        model_tok_bet.add(Bidirectional(LSTM(units2, return_sequences=True)))
    model_tok_bet.add(AttentionWithContext())
    model_tok_bet.add(Dropout(dp))



    word_emb_lay = get_embedding_layer(word_emb_path, trainable=True, max_seq=max_seq)
    model_tok_aft = Sequential()
    model_tok_aft.add(word_emb_lay)
    model_tok_aft.add(Bidirectional(LSTM(units1, return_sequences=True)))
    if units2:
        model_tok_aft.add(Bidirectional(LSTM(units2, return_sequences=True)))
    model_tok_aft.add(AttentionWithContext())
    model_tok_aft.add(Dropout(dp))


    if pos_emb_path:

        pos_emb_lay = get_embedding_layer(word_emb_path, trainable=False, max_seq=max_seq)
        model_pos_bef = Sequential()
        model_pos_bef.add(pos_emb_lay)
        model_pos_bef.add(Bidirectional(LSTM(units1, return_sequences=True)))
        if units2:
            model_pos_bef.add(Bidirectional(LSTM(units2, return_sequences=True)))
        model_pos_bef.add(AttentionWithContext())
        model_pos_bef.add(Dropout(dp))



        pos_emb_lay = get_embedding_layer(word_emb_path, trainable=True, max_seq=max_seq)
        model_pos_bet = Sequential()
        model_pos_bet.add(pos_emb_lay)
        model_pos_bet.add(Bidirectional(LSTM(units1, return_sequences=True)))
        if units2:
            model_pos_bet.add(Bidirectional(LSTM(units2, return_sequences=True)))
        model_pos_bet.add(AttentionWithContext())
        model_pos_bet.add(Dropout(dp))



        pos_emb_lay = get_embedding_layer(word_emb_path, trainable=True, max_seq=max_seq)
        model_pos_aft = Sequential()
        model_pos_aft.add(pos_emb_lay)
        model_pos_aft.add(Bidirectional(LSTM(units1, return_sequences=True)))
        if units2:
            model_pos_aft.add(Bidirectional(LSTM(units2, return_sequences=True)))
        model_pos_aft.add(AttentionWithContext())
        model_pos_aft.add(Dropout(dp))


    out_list = []
    out_list.append(model_tok_bef.output)
    out_list.append(model_tok_bet.output)
    out_list.append(model_tok_aft.output)
    if pos_emb_path:
        out_list.append(model_pos_bef.output)
        out_list.append(model_pos_bet.output)
        out_list.append(model_pos_aft.output)

    in_list = []
    in_list.append(model_tok_bef.input)
    in_list.append(model_tok_bet.input)
    in_list.append(model_tok_aft.input)
    if pos_emb_path:
        in_list.append(model_pos_bef.input)
        in_list.append(model_pos_bet.input)
        in_list.append(model_pos_aft.input)

    mergedOut = Concatenate()(out_list)

    beta = Dense(units1, activation='relu')(mergedOut)

    #beta = LSTM(50, return_sequences=False)(mergedOut)

    lab_rel_type = Dense(14, activation='softmax', name="lab_rel")(beta)

    merged = Concatenate()([lab_rel_type,beta])

    lab_dir = Dense(3, activation='softmax', name="lab_dir")(merged)

    #lab_all = Dense(27, activation='softmax', name="lab_rel")(beta)

    model = Model(in_list, [lab_dir, lab_rel_type])
    #model = Model(in_list, lab_all)

    print(model.summary())
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def token_bba_v2(word_emb_path, pos_emb_path, ent_emb_path, lr, units1=128, units2=0, dp=0.5, max_seq=50, spa_dp=False, trainable=False):
    # opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    opt = optimizers.Adam(lr=lr)

    word_emb_lay = get_embedding_layer(word_emb_path, trainable=trainable, max_seq=max_seq)

    model_tok_bef = Sequential()
    model_tok_bef.add(word_emb_lay)
    if spa_dp:
        model_tok_bef.add(SpatialDropout1D(dp))
    model_tok_bef.add(Bidirectional(LSTM(units1, recurrent_regularizer=L1L2(l1=0.01, l2=0.01), return_sequences=True)))
    if units2:
        model_tok_bef.add(Bidirectional(LSTM(units2, recurrent_regularizer=L1L2(l1=0.01, l2=0.01), return_sequences=True)))
    model_tok_bef.add(AttentionWithContext())
    model_tok_bef.add(Dropout(dp))



    word_emb_lay = get_embedding_layer(word_emb_path, trainable=trainable, max_seq=max_seq)
    model_tok_bet = Sequential()
    model_tok_bet.add(word_emb_lay)
    if spa_dp:
        model_tok_bet.add(SpatialDropout1D(dp))
    model_tok_bet.add(Bidirectional(LSTM(units1, recurrent_regularizer=L1L2(l1=0.01, l2=0.01), return_sequences=True)))
    if units2:
        model_tok_bet.add(Bidirectional(LSTM(units2, recurrent_regularizer=L1L2(l1=0.01, l2=0.01), return_sequences=True)))
    model_tok_bet.add(AttentionWithContext())
    model_tok_bet.add(Dropout(dp))



    word_emb_lay = get_embedding_layer(word_emb_path, trainable=trainable, max_seq=max_seq)
    model_tok_aft = Sequential()
    model_tok_aft.add(word_emb_lay)
    if spa_dp:
        model_tok_aft.add(SpatialDropout1D(dp))
    model_tok_aft.add(Bidirectional(LSTM(units1, recurrent_regularizer=L1L2(l1=0.01, l2=0.01), return_sequences=True)))
    if units2:
        model_tok_aft.add(Bidirectional(LSTM(units2, recurrent_regularizer=L1L2(l1=0.01, l2=0.01), return_sequences=True)))
    model_tok_aft.add(AttentionWithContext())
    model_tok_aft.add(Dropout(dp))




    pos_emb_lay = get_embedding_layer(pos_emb_path, trainable=trainable, max_seq=max_seq)
    model_pos_bef = Sequential()
    model_pos_bef.add(pos_emb_lay)
    if spa_dp:
        model_pos_bef.add(SpatialDropout1D(dp))
    model_pos_bef.add(Bidirectional(LSTM(units1, recurrent_regularizer=L1L2(l1=0.01, l2=0.01), return_sequences=True)))
    if units2:
        model_pos_bef.add(Bidirectional(LSTM(units2, recurrent_regularizer=L1L2(l1=0.01, l2=0.01), return_sequences=True)))
    model_pos_bef.add(AttentionWithContext())
    model_pos_bef.add(Dropout(dp))



    pos_emb_lay = get_embedding_layer(pos_emb_path, trainable=trainable, max_seq=max_seq)
    model_pos_bet = Sequential()
    model_pos_bet.add(pos_emb_lay)
    if spa_dp:
        model_pos_bet.add(SpatialDropout1D(dp))
    model_pos_bet.add(Bidirectional(LSTM(units1, recurrent_regularizer=L1L2(l1=0.01, l2=0.01), return_sequences=True)))
    if units2:
        model_pos_bet.add(Bidirectional(LSTM(units2, recurrent_regularizer=L1L2(l1=0.01, l2=0.01), return_sequences=True)))
    model_pos_bet.add(AttentionWithContext())
    model_pos_bet.add(Dropout(dp))



    pos_emb_lay = get_embedding_layer(pos_emb_path, trainable=trainable, max_seq=max_seq)
    model_pos_aft = Sequential()
    model_pos_aft.add(pos_emb_lay)
    if spa_dp:
        model_pos_aft.add(SpatialDropout1D(dp))
    model_pos_aft.add(Bidirectional(LSTM(units1, recurrent_regularizer=L1L2(l1=0.01, l2=0.01), return_sequences=True)))
    if units2:
        model_pos_aft.add(Bidirectional(LSTM(units2, recurrent_regularizer=L1L2(l1=0.01, l2=0.01), return_sequences=True)))
    model_pos_aft.add(AttentionWithContext())
    model_pos_aft.add(Dropout(dp))



    enty_emb_lay = get_embedding_layer(ent_emb_path, trainable=trainable, max_seq=2)
    inp2 = Input(shape=(2,))
    y = enty_emb_lay(inp2)
    y = Flatten()(y)
    y = Dense(10, activation='relu')(y)

    out_list = []
    out_list.append(model_tok_bef.output)
    out_list.append(model_tok_bet.output)
    out_list.append(model_tok_aft.output)
    if pos_emb_path:
        out_list.append(model_pos_bef.output)
        out_list.append(model_pos_bet.output)
        out_list.append(model_pos_aft.output)
    out_list.append(y)

    in_list = []
    in_list.append(model_tok_bef.input)
    in_list.append(model_tok_bet.input)
    in_list.append(model_tok_aft.input)
    if pos_emb_path:
        in_list.append(model_pos_bef.input)
        in_list.append(model_pos_bet.input)
        in_list.append(model_pos_aft.input)
    in_list.append(inp2)

    mergedOut = Concatenate()(out_list)

    beta = Dense(units1*2, activation='relu')(mergedOut)


    lab_is_rel= Dense(2, activation='softmax', name="lab_is_rel")(beta)

    concat_is_rel = Concatenate()([lab_is_rel,beta])

    lab_rel_type = Dense(14, activation='softmax', name="lab_rel")(concat_is_rel)

    concat_rel_type = Concatenate()([lab_is_rel,lab_rel_type,beta])

    lab_dir = Dense(3, activation='softmax', name="lab_dir")(concat_rel_type)



    model = Model(in_list, [lab_dir, lab_rel_type, lab_is_rel])

    print(model.summary())
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def get_part_model_only_embeddings(word_emb_path, pos_emb_path, dep_emb_path, par_emb_path,ent_emb_path,units, dp, trainable, max_seq, name):

    input_layers = []
    emb_layers = []

    if word_emb_path:
        inp_word = Input(shape=(max_seq,), name="word_"+name)
        word_emb_lay = get_embedding_layer(word_emb_path, trainable=trainable, max_seq=max_seq)
        layer_word = word_emb_lay(inp_word)
        input_layers.append(inp_word)
        emb_layers.append(layer_word)

    if pos_emb_path:
        inp_pos = Input(shape=(max_seq,), name="pos_"+name)
        pos_emb_lay = get_embedding_layer(pos_emb_path, trainable=trainable, max_seq=max_seq)
        layer_pos = pos_emb_lay(inp_pos)
        input_layers.append(inp_pos)
        emb_layers.append(layer_pos)

    if dep_emb_path:
        inp_dep = Input(shape=(max_seq,), name="dep_" + name)
        dep_emb_lay = get_embedding_layer(dep_emb_path, trainable=trainable, max_seq=max_seq)
        layer_dep = dep_emb_lay(inp_dep)
        input_layers.append(inp_dep)
        emb_layers.append(layer_dep)

    if par_emb_path:
        inp_par = Input(shape=(max_seq,), name="par_" + name)
        par_emb_lay = get_embedding_layer(par_emb_path, trainable=trainable, max_seq=max_seq)
        layer_par = par_emb_lay(inp_par)
        input_layers.append(inp_par)
        emb_layers.append(layer_par)

    if ent_emb_path:
        inp_ent = Input(shape=(max_seq,), name="ent_" + name)
        ent_emb_lay = get_embedding_layer(ent_emb_path, trainable=trainable, max_seq=max_seq)
        layer_ent = ent_emb_lay(inp_ent)
        input_layers.append(inp_ent)
        emb_layers.append(layer_ent)


    return input_layers, emb_layers
    #return [inp_word, inp_pos],before


def get_part_model(word_emb_path, pos_emb_path, dep_emb_path, par_emb_path,ent_emb_path,units, dp, trainable, max_seq, name):

    input_layers = []
    emb_layers = []

    if word_emb_path:
        inp_word = Input(shape=(max_seq,), name="word_"+name)
        word_emb_lay = get_embedding_layer(word_emb_path, trainable=trainable, max_seq=max_seq)
        layer_word = word_emb_lay(inp_word)
        input_layers.append(inp_word)
        emb_layers.append(layer_word)

    if pos_emb_path:
        inp_pos = Input(shape=(max_seq,), name="pos_"+name)
        pos_emb_lay = get_embedding_layer(pos_emb_path, trainable=trainable, max_seq=max_seq)
        layer_pos = pos_emb_lay(inp_pos)
        input_layers.append(inp_pos)
        emb_layers.append(layer_pos)

    if dep_emb_path:
        inp_dep = Input(shape=(max_seq,), name="dep_" + name)
        dep_emb_lay = get_embedding_layer(dep_emb_path, trainable=trainable, max_seq=max_seq)
        layer_dep = dep_emb_lay(inp_dep)
        input_layers.append(inp_dep)
        emb_layers.append(layer_dep)

    if par_emb_path:
        inp_par = Input(shape=(max_seq,), name="par_" + name)
        par_emb_lay = get_embedding_layer(par_emb_path, trainable=trainable, max_seq=max_seq)
        layer_par = par_emb_lay(inp_par)
        input_layers.append(inp_par)
        emb_layers.append(layer_par)

    if ent_emb_path:
        inp_ent = Input(shape=(max_seq,), name="ent_" + name)
        ent_emb_lay = get_embedding_layer(ent_emb_path, trainable=trainable, max_seq=max_seq)
        layer_ent = ent_emb_lay(inp_ent)
        input_layers.append(inp_ent)
        emb_layers.append(layer_ent)

    before = Concatenate()(emb_layers)


    before = SpatialDropout1D(dp)(before)
    before = Bidirectional(LSTM(units, recurrent_regularizer=L1L2(l1=0.01, l2=0.01), return_sequences=True))(before)

    before = AttentionWeightedAverage()(before)

    before = Dropout(dp)(before)
    return input_layers, before
    #return [inp_word, inp_pos],before


def token_bba_v4(word_emb_path, pos_emb_path, ent_emb_path, dep_emb_path,par_emb_path, entity_emb_path, lr, units1=128, units2=0, dp=0.5, max_seq=50, ENT_EMB_DENSE=False, trainable=False):
    # opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    opt = optimizers.Adam(lr=lr)
    print("par_emb_path", par_emb_path)

    inp_bef_list, before = get_part_model(word_emb_path=word_emb_path,
                            pos_emb_path=pos_emb_path,
                            dep_emb_path=dep_emb_path,
                            par_emb_path=par_emb_path,
                            ent_emb_path=entity_emb_path,
                            units=units1,
                            dp=dp,
                            trainable=trainable,
                            max_seq=max_seq,
                            name="before")

    inp_bet_list, between = get_part_model(word_emb_path=word_emb_path,
                            pos_emb_path=pos_emb_path,
                            dep_emb_path=dep_emb_path,
                            par_emb_path=par_emb_path,
                            ent_emb_path=entity_emb_path,
                            units=units1,
                            dp=dp,
                            trainable=trainable,
                            max_seq=max_seq,
                            name="between")

    inp_aft_list, after = get_part_model(word_emb_path=word_emb_path,
                            pos_emb_path=pos_emb_path,
                            dep_emb_path=dep_emb_path,
                            par_emb_path=par_emb_path,
                            ent_emb_path=entity_emb_path,
                            units=units1,
                            dp=dp,
                            trainable=trainable,
                            max_seq=max_seq,
                            name="after")

    inp_ent1_list, local_ent1_emb = get_part_model_only_embeddings(word_emb_path=word_emb_path,
                                         pos_emb_path=pos_emb_path,
                                         dep_emb_path=dep_emb_path,
                                         par_emb_path=par_emb_path,
                                         ent_emb_path=entity_emb_path,
                                         units=units1,
                                         dp=dp,
                                         trainable=trainable,
                                         max_seq=10,
                                         name="local_ent1")

    inp_ent2_list, local_ent2_emb = get_part_model_only_embeddings(word_emb_path=word_emb_path,
                                         pos_emb_path=pos_emb_path,
                                         dep_emb_path=dep_emb_path,
                                         par_emb_path=par_emb_path,
                                         ent_emb_path=entity_emb_path,
                                         units=units1,
                                         dp=dp,
                                         trainable=trainable,
                                         max_seq=10,
                                         name="local_ent2")

    local = Concatenate()(local_ent1_emb + local_ent2_emb)

    local = SpatialDropout1D(dp)(local)
    local = Bidirectional(LSTM(units1, return_sequences=True))(local)

    local = AttentionWeightedAverage()(local)

    local = Dropout(dp)(local)

    in_list = inp_bef_list + inp_bet_list + inp_aft_list +inp_ent1_list + inp_ent2_list

    out_list = []
    out_list.append(before)
    out_list.append(between)
    out_list.append(after)
    out_list.append(local)




    mergedOut = Concatenate()(out_list)

    beta = Dense(units1*2, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01))(mergedOut)


    lab_is_rel= Dense(2, activation='softmax', name="lab_is_rel")(beta)

    concat_is_rel = Concatenate()([lab_is_rel,beta])

    lab_rel_type = Dense(14, activation='softmax', name="lab_rel")(concat_is_rel)

    concat_rel_type = Concatenate()([lab_is_rel,lab_rel_type,beta])

    lab_dir = Dense(3, activation='softmax', name="lab_dir")(concat_rel_type)



    model = Model(in_list, [lab_dir, lab_rel_type, lab_is_rel])

    print(model.summary())
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def token_bba_v3(word_emb_path, pos_emb_path, ent_emb_path, dep_emb_path,par_emb_path, entity_emb_path, lr, units1=128, units2=0, dp=0.5, max_seq=50, ENT_EMB_DENSE=False, trainable=False):
    # opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    opt = optimizers.Adam(lr=lr)
    print("par_emb_path", par_emb_path)

    inp_bef_list, before = get_part_model(word_emb_path=word_emb_path,
                            pos_emb_path=pos_emb_path,
                            dep_emb_path=dep_emb_path,
                            par_emb_path=par_emb_path,
                            ent_emb_path=entity_emb_path,
                            units=units1,
                            dp=dp,
                            trainable=trainable,
                            max_seq=max_seq,
                            name="before")

    inp_bet_list, between = get_part_model(word_emb_path=word_emb_path,
                            pos_emb_path=pos_emb_path,
                            dep_emb_path=dep_emb_path,
                            par_emb_path=par_emb_path,
                            ent_emb_path=entity_emb_path,
                            units=units1,
                            dp=dp,
                            trainable=trainable,
                            max_seq=max_seq,
                            name="between")

    inp_aft_list, after = get_part_model(word_emb_path=word_emb_path,
                            pos_emb_path=pos_emb_path,
                            dep_emb_path=dep_emb_path,
                            par_emb_path=par_emb_path,
                            ent_emb_path=entity_emb_path,
                            units=units1,
                            dp=dp,
                            trainable=trainable,
                            max_seq=max_seq,
                            name="after")

    in_list = inp_bef_list + inp_bet_list + inp_aft_list
    out_list = []
    out_list.append(before)
    out_list.append(between)
    out_list.append(after)

    if ENT_EMB_DENSE:
        enty_emb_lay = get_embedding_layer(ent_emb_path, trainable=trainable, max_seq=2)
        inp_ent = Input(shape=(2,))
        y = enty_emb_lay(inp_ent)
        y = Flatten()(y)
        y = Dense(10, activation='relu')(y)
        out_list.append(y)
        in_list.append(inp_ent)

    mergedOut = Concatenate()(out_list)

    beta = Dense(units1*2, activation='relu')(mergedOut)


    lab_is_rel= Dense(2, activation='softmax', name="lab_is_rel")(beta)

    concat_is_rel = Concatenate()([lab_is_rel,beta])

    lab_rel_type = Dense(14, activation='softmax', name="lab_rel")(concat_is_rel)

    concat_rel_type = Concatenate()([lab_is_rel,lab_rel_type,beta])

    lab_dir = Dense(3, activation='softmax', name="lab_dir")(concat_rel_type)



    model = Model(in_list, [lab_dir, lab_rel_type, lab_is_rel])

    print(model.summary())
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def all_sent_model(output_path, lr, units1=128, units2=64, dp=0.5):
    # opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    opt = optimizers.Adam(lr=lr)

    word_emb_path = os.path.join(output_path, "word_emb.pkl")
    entity_emb_path = os.path.join(output_path, "entity_emb.pkl")
    position_emb_path = os.path.join(output_path, "position_emb.pkl")



    word_emb_lay = get_embedding_layer(word_emb_path, trainable=False, max_seq=50)

    inp1 = Input(shape=(50,))
    x = word_emb_lay(inp1)
    x = Bidirectional(LSTM(100, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))(x)
    x = AttentionWithContext()(x)
    x = Dropout(0.5)(x)

    enty_emb_lay = get_embedding_layer(entity_emb_path, trainable=False, max_seq=2)
    inp2 = Input(shape=(2,))
    y = enty_emb_lay(inp2)
    y = Flatten()(y)

    position_emb_lay = get_embedding_layer(position_emb_path, trainable=False, max_seq=2)
    inp3 = Input(shape=(2,))
    z = position_emb_lay(inp3)
    z = Flatten()(z)

    alpha = concatenate([x, y, z])

    #alpha = Bidirectional(LSTM(128, return_sequences=False))(alpha)
    alpha = Dense(50, activation='relu', name="relu")(alpha)
    alpha = Dropout(0.5)(alpha)

    lab_rel_all = Dense(27, activation='softmax', name="lab_rel")(alpha)
    model = Model([inp1, inp2, inp3], lab_rel_all)


    print(model.summary())
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def all_sent_model_old(output_path, lr, units1=128, units2=64, dp=0.5):
    # opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    opt = optimizers.Adam(lr=lr)

    word_emb_path = os.path.join(output_path, "word_emb.pkl")
    entity_emb_path = os.path.join(output_path, "entity_emb.pkl")
    position_emb_path = os.path.join(output_path, "position_emb.pkl")

    word_emb_lay = get_embedding_layer(word_emb_path, trainable=False, max_seq=50)

    model_tok = Sequential()
    model_tok.add(word_emb_lay)
    model_tok.add(Bidirectional(LSTM(units1, return_sequences=True)))
    model_tok.add(Bidirectional(LSTM(units2, return_sequences=True)))
    model_tok.add(AttentionWithContext())
    model_tok.add(Dropout(dp))

    enty_emb_lay = get_embedding_layer(entity_emb_path, trainable=False, max_seq=2)
    model_entity = Sequential()
    model_entity.add(enty_emb_lay)
    model_entity.add(Bidirectional(LSTM(10, return_sequences=False)))
    #model_entity.add(Bidirectional(LSTM(units2, return_sequences=True)))
    #model_entity.add(AttentionWithContext())
    model_entity.add(Dropout(dp))

    position_emb_lay = get_embedding_layer(position_emb_path, trainable=False, max_seq=2)
    model_position = Sequential()
    model_position.add(position_emb_lay)
    model_position.add(Bidirectional(LSTM(10, return_sequences=False)))
    #model_position.add(Bidirectional(LSTM(units2, return_sequences=True)))
    #model_position.add(AttentionWithContext())
    model_position.add(Dropout(dp))



    mergedOut = Concatenate()([model_tok.output,model_entity.output,model_position.output])

    beta = Dense(units1, activation='relu')(mergedOut)
    #lab_dir = Dense(3, activation='softmax', name="lab_dir")(beta)
    #lab_rel_type = Dense(14, activation='softmax', name="lab_rel")(beta)

    lab_rel_all = Dense(27, activation='softmax', name="lab_rel")(beta)
    model = Model([model_tok.input, model_entity.input, model_position.input], lab_rel_all)


    print(model.summary())
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def cnn_binary_with_emb_layer(embedding_layer,lr):
    nb_feature_maps = 32
    embedding_size = 300
    maxlen=50

    # ngram_filters = [3, 5, 7, 9]
    ngram_filters = [2, 4, 6, 8]
    conv_filters = []

    for n_gram in ngram_filters:
        sequential = Sequential()
        conv_filters.append(sequential)

        sequential.add(embedding_layer)
        sequential.add(Reshape(1, maxlen, embedding_size))
        sequential.add(Convolution2D(nb_feature_maps, 1, n_gram, embedding_size))
        sequential.add(Activation("relu"))
        sequential.add(MaxPooling2D(poolsize=(maxlen - n_gram + 1, 1)))
        sequential.add(Flatten())

    model = Sequential()
    mergedOut = Concatenate()(conv_filters)

    model.add(mergedOut)
    model.add(Dropout(0.5))
    model.add(Dense(nb_feature_maps * len(conv_filters), 1))
    model.add(Activation("sigmoid"))

def get_input_layer(max_seq):
    sequence_input = Input(shape=(max_seq,), dtype='int32')
    return sequence_input


def get_embedded_sequences(embedding_matrix,sequence_input, max_seq, trainable=False):
    print("embedding_matrix.shape", embedding_matrix.shape)

    embedding_layer = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                input_length=max_seq,
                                trainable=trainable)


    embedded_sequences = embedding_layer(sequence_input)
    embedded_sequences = SpatialDropout1D(0.5)(embedded_sequences)
    print(embedded_sequences.shape)

    embedded_sequences = Reshape((max_seq, embedding_matrix.shape[1], 1))(embedded_sequences)

    return embedded_sequences


def get_convolutions(embedded_sequences, embedding_matrix,max_seq, nfilters, grams=[5, 4, 3, 2]):
    convolutions = []

    for gram in grams:
        x = Convolution2D(nfilters, (gram, embedding_matrix.shape[1]), activation='relu')(embedded_sequences)
        x = MaxPooling2D((max_seq - gram + 1, 1))(x)
        convolutions.append(x)

    alpha = concatenate(convolutions)
    alpha = Flatten()(alpha)

    return alpha

def get_convolutions_old(emb_matrix_path,max_seq, nfilters, grams = [5,4,3,2], trainable=False):
    embedding_matrix = load_obj(emb_matrix_path)
    print("embedding_matrix.shape", embedding_matrix.shape)


    embedding_layer = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                input_length=max_seq,
                                trainable=trainable)

    sequence_input = Input(shape=(max_seq,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    embedded_sequences = SpatialDropout1D(0.5)(embedded_sequences)
    print(embedded_sequences.shape)

    embedded_sequences = Reshape((max_seq, embedding_matrix.shape[1], 1))(embedded_sequences)



    convolutions = []

    for gram in grams:
        x = Convolution2D(nfilters, (gram, embedding_matrix.shape[1]), activation='relu')(embedded_sequences)
        x = MaxPooling2D((max_seq - gram + 1, 1))(x)
        convolutions.append(x)
    """
    x = Convolution2D(nfilters, (5, embedding_matrix.shape[1]), activation='relu')(embedded_sequences)
    x = MaxPooling2D((max_seq - 5 + 1, 1))(x)

    y = Convolution2D(nfilters, (4, embedding_matrix.shape[1]), activation='relu')(embedded_sequences)
    y = MaxPooling2D((max_seq - 4 + 1, 1))(y)

    z = Convolution2D(nfilters, (3, embedding_matrix.shape[1]), activation='relu')(embedded_sequences)
    z = MaxPooling2D((max_seq - 3 + 1, 1))(z)

    a = Convolution2D(nfilters, (2, embedding_matrix.shape[1]), activation='relu')(embedded_sequences)
    a = MaxPooling2D((max_seq - 2 + 1, 1))(a)
    
    alpha = concatenate([x, y, z, a])
    """
    alpha = concatenate(convolutions)
    alpha = Flatten()(alpha)

    return sequence_input, alpha

def get_CNN_last_part(sequence_input, alpha,nlabels, dense_nodes=100, lr=0.0001):
    beta = Dense(dense_nodes*2, activation='relu')(alpha)

    preds = Dense(nlabels, activation='softmax')(beta)

    model = Model(sequence_input, preds)

    adadelta = optimizers.Adadelta(lr=lr)
    print(model.summary())
    model.compile(loss='categorical_crossentropy',
                  optimizer=adadelta,
                  metrics=['acc'])
    return model

def cnn_binary_with_emb_layer_(embedding_matrix,lr, nlabels = 2,nfilters=50, max_seq=50):

    print("EMB_MATRIX: ", embedding_matrix.shape, "max_seq", max_seq)
    embedding_layer = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                input_length=max_seq,
                                trainable=True)

    sequence_input = Input(shape=(max_seq,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)


    print(embedded_sequences.shape)

    embedded_sequences = Reshape((max_seq, embedding_matrix.shape[1], 1))(embedded_sequences)
    x = Convolution2D(nfilters, (5, embedding_matrix.shape[1]), activation='relu')(embedded_sequences)
    x = MaxPooling2D((max_seq - 5 + 1, 1))(x)

    y = Convolution2D(nfilters, (4, embedding_matrix.shape[1]), activation='relu')(embedded_sequences)
    y = MaxPooling2D((max_seq - 4 + 1, 1))(y)

    z = Convolution2D(nfilters, (3, embedding_matrix.shape[1]), activation='relu')(embedded_sequences)
    z = MaxPooling2D((max_seq - 3 + 1, 1))(z)

    a = Convolution2D(nfilters, (2, embedding_matrix.shape[1]), activation='relu')(embedded_sequences)
    a = MaxPooling2D((max_seq - 2 + 1, 1))(a)

    alpha = concatenate([x, y, z, a])
    alpha = Flatten()(alpha)

    # dropout
    alpha = Dropout(0.5)(alpha)

    # predictions

    beta = Dense(nfilters, activation='relu')(alpha)

    preds = Dense(nlabels, activation='softmax',activity_regularizer=L1L2(l1=0.01, l2=0.01))(beta)

    model = Model(sequence_input, preds)
    adadelta = optimizers.Adadelta(lr=lr)
    print(model.summary())
    model.compile(loss='categorical_crossentropy',
                  optimizer=adadelta,
                  metrics=['acc'])

    return model

def cnn_binary_with_emb_layer_lstm(embedding_matrix,lr, nlabels = 2,nfilters=50, max_seq=50):

    print("EMB_MATRIX: ", embedding_matrix.shape, "max_seq", max_seq)
    embedding_layer = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                input_length=max_seq,
                                trainable=True)

    model_conv = Sequential()
    model_conv.add(embedding_layer)
    model_conv.add(Dropout(0.2))
    model_conv.add(Conv1D(64, 5, activation='relu'))
    model_conv.add(MaxPooling1D(pool_size=4))
    model_conv.add(LSTM(100))
    model_conv.add(Dense(nlabels, activation='softmax'))
    print(model_conv.summary())
    model_conv.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_conv

def get_cnn_model_v2(embedding_matrix,lr, nlabels = 2,nfilters=50, max_seq=50): # added embed
    model = Sequential()
    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    # 1000 is num_max
    embedding_layer = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                input_length=max_seq,
                                trainable=True)
    model.add(embedding_layer)
    model.add(Dropout(0.2))
    model.add(Conv1D(nfilters,
                     3,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(nlabels, activation='softmax'))
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def cnn_binary_with_emb_layer_char_word(embedding_matrix_word, embedding_matrix_char,lr, nlabels = 2,nfilters=50, max_seq_word=50, max_seq_char=280):
    sequence_inputw, alphaw = get_convolutions(embedding_matrix_word,max_seq_word, nfilters, [5,4,3,2])
    # dropout
    alphaw = Dropout(0.5)(alphaw)

    sequence_inputc, alphac = get_convolutions(embedding_matrix_char, max_seq_char, nfilters, [16,8,4,2])
    # dropout
    alphac = Dropout(0.5)(alphac)

    # predictions

    alpha = concatenate([alphaw,alphac])
    beta = Dense(nfilters, activation='relu')(alpha)

    preds = Dense(nlabels, activation='softmax')(beta)

    model = Model([sequence_inputw, sequence_inputc], preds)
    adadelta = optimizers.Adadelta(lr=lr)
    print(model.summary())
    model.compile(loss='categorical_crossentropy',
                  optimizer=adadelta,
                  metrics=['acc'])

    return model

def lstm_simple_binary_all(embedding_layer,bilstm, dense_list, lr, do=0.5):
    #opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    opt = optimizers.Adam(lr=lr)

    model = Sequential()
    model.add(embedding_layer)
    model.add(SpatialDropout1D(do))
    model.add(bilstm)
    model.add(Dropout(do))
    for dense in dense_list:
        model.add(dense)
    print(model.summary())
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def cnn(emb_matrix,num_filters=50, filter_sizes = [2,3,4], lr=1e-4, max_seq=50, do = 0.5):
    inputs = Input(shape=(max_seq,), dtype='int32')
    embedding = Embedding(emb_matrix.shape[0],
                                emb_matrix.shape[1],
                                weights=[emb_matrix],
                                input_length=max_seq,
                                trainable=True)(inputs)


    reshape = Reshape((max_seq, emb_matrix.shape[1], 1))(embedding)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], emb_matrix.shape[1]), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], emb_matrix.shape[1]), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], emb_matrix.shape[1]), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(max_seq - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(max_seq - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(max_seq - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(do)(flatten)
    output = Dense(units=2, activation='softmax')(dropout)

    # this creates a model that includes
    model = Model(inputs=inputs, outputs=output)

    #checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1,
    #                             save_best_only=True, mode='auto')
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])


def model_lstm_atten(embedding_matrix,lr, nlabels=2, nunits=50, max_seq=50, do=0.5):
    inp = Input(shape=(max_seq,))
    x = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                input_length=max_seq,
                                trainable=True)(inp)
    x = SpatialDropout1D(do)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = AttentionWithContext()(x)
    x = Dropout(do)(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(nlabels, activation="softmax")(x)
    model = Model(inputs=inp, outputs=x)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def lstm_simple_binary_with_emb_layer(embedding_layer,lr, bilstm=None, do=0.5, units=50, nlabels=2):
    #opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    opt = optimizers.Adam(lr=lr)
    model = Sequential()
    model.add(embedding_layer)
    model.add(SpatialDropout1D(do))
    if bilstm:
        model.add(bilstm)
    else:
        model.add(Bidirectional(LSTM(units=units, recurrent_regularizer=L1L2(l1=0.01, l2=0.01))))
    model.add(Dropout(do))
    #model.add(AttentionWithContext())
    model.add(Dense(units*2, activation='relu'))
    model.add(Dense(nlabels, activation='softmax'))
    print(model.summary())
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def lstm_simple_binary_with_emb_layer_att(embedding_layer,lr, bilstm=None, do=0.5, units=50, nlabels=2):
    #opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    opt = optimizers.Adam(lr=lr)
    model = Sequential()
    model.add(embedding_layer)
    model.add(SpatialDropout1D(do))
    if bilstm:
        model.add(bilstm)
    else:
        model.add(Bidirectional(LSTM(units=units, return_sequences=True, recurrent_regularizer=L1L2(l1=0.01, l2=0.01))))
    model.add(AttentionWithContext())
    model.add(Dropout(do))
    model.add(Dense(units*2, activation='relu'))
    model.add(Dense(nlabels, activation='softmax'))
    print(model.summary())
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def lstm_simple_binary_attent(embedding_matrix,lr, nlabels=2, nunits=50, max_seq=50):


    opt = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    embedding_layer = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                input_length=max_seq,
                                trainable=True)

    d = 0.5
    rd = 0.5

    model = Sequential()
    model.add(embedding_layer)
    model.add(SpatialDropout1D(d))
    model.add(Bidirectional(LSTM(units=nunits, recurrent_regularizer=L1L2(l1=0.01, l2=0.01))))#, dropout=d, recurrent_dropout=rd)))
    model.add(AttentionWeightedAverage())
    model.add(Dropout(d))
    model.add(Dense(nlabels, activation='softmax'))

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def lstm_simple_binary(embedding_matrix,lr, nlabels=2, nunits=50, max_seq=50):
    #opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    opt = optimizers.Adam(lr=lr)

    embedding_layer = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                input_length=max_seq,
                                trainable=True)
    model = Sequential()
    model.add(embedding_layer)
    model.add(SpatialDropout1D(0.5))
    model.add(Bidirectional(LSTM(units = nunits,recurrent_regularizer=L1L2(l1=0.01, l2=0.01))))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(nlabels, activation='softmax'))
    print(model.summary())
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def lstm_haha(word_emb_path,lr, nlabels=2, nunits=50, max_seq=50, trainable=True):
    #opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    opt = optimizers.Adam(lr=lr)

    inp_word = Input(shape=(max_seq,), name="word_seq")

    word_emb_lay = get_embedding_layer(word_emb_path, trainable=trainable, max_seq=max_seq)

    layer_word = word_emb_lay(inp_word)

    layer_word = SpatialDropout1D(0.5)(layer_word)
    layer_word= Bidirectional(LSTM(units = nunits))(layer_word)

    layer_word= Dropout(0.5)(layer_word)

    beta = Dense(nunits * 2, activation='relu')(layer_word)

    lab_is_humor = Dense(nlabels, activation='softmax', name="is_humor")(beta)

    #concat_is_humor= Concatenate()([lab_is_humor, beta])

    lab_score = Dense(1, activation='sigmoid', name="score")(beta)

    model = Model(inp_word, [lab_is_humor, lab_score])


    loss = {'is_humor': 'categorical_crossentropy',
            'score': 'mse'}

    metrics= {'is_humor': 'accuracy',
            'score': 'mse'}
    print(model.summary())
    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    return model

def lstm_haha_emb_lay(word_emb_lay,lr, nlabels=2, nunits=50, max_seq=50, trainable=True):
    #opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    opt = optimizers.Adam(lr=lr)

    inp_word = Input(shape=(max_seq,), name="word_seq")


    layer_word = word_emb_lay(inp_word)

    layer_word = SpatialDropout1D(0.5)(layer_word)
    layer_word= Bidirectional(LSTM(units = nunits))(layer_word)

    layer_word= Dropout(0.5)(layer_word)

    beta = Dense(nunits * 2, activation='relu')(layer_word)

    lab_is_humor = Dense(nlabels, activation='softmax', name="is_humor")(beta)

    #concat_is_humor= Concatenate()([lab_is_humor, beta])

    lab_score = Dense(1, activation='sigmoid', name="score")(beta)

    model = Model(inp_word, [lab_is_humor, lab_score])


    loss = {'is_humor': 'categorical_crossentropy',
            'score': 'mse'}

    metrics= {'is_humor': 'accuracy',
            'score': 'mse'}
    print(model.summary())
    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    return model


def cnn_haha(word_emb_path,lr, nlabels=2, nunits=50, max_seq=50, trainable=True):
    #opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    opt = optimizers.Adam(lr=lr)

    inp_word, layer_word = get_convolutions_old(word_emb_path, max_seq, nunits, [5, 4, 3, 2])

    layer_word= Dropout(0.5)(layer_word)

    beta = Dense(nunits * 2, activation='relu')(layer_word)

    lab_is_humor = Dense(nlabels, activation='softmax', name="is_humor")(beta)

    concat_is_humor= Concatenate()([lab_is_humor, beta])

    lab_score = Dense(1, activation='sigmoid', name="score")(concat_is_humor)

    model = Model(inp_word, [lab_is_humor, lab_score])


    loss = {'is_humor': 'categorical_crossentropy',
            'score': 'mean_squared_error'}

    metrics= {'is_humor': 'accuracy',
            'score': 'mse'}
    print(model.summary())
    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    return model





def model_testing(model, x_val, y_val, nlabel=0):
    final_pred = []
    final_real = []

    for i in range(0, len(x_val)):
        prediction = model.predict(np.array([x_val[i]]))
        predicted_label = np.argmax(prediction[nlabel])

        real_label = np.argmax(y_val[i])

        final_pred.append(predicted_label)
        final_real.append(real_label)



    print('Confusion Matrix')
    print(confusion_matrix(final_real, final_pred))
    print('Classification Report')
    print(classification_report(final_real, final_pred))

def model_testing_haha(model, x_val, y_val, y_score, scaler):
    final_pred = []
    final_real = []
    final_score_pred  = []
    final_score_real = []


    for i in range(0, len(x_val)):
        prediction = model.predict(np.array([x_val[i]]))
        predicted_label = np.argmax(prediction[0])

        real_label = np.argmax(y_val[i])

        final_pred.append(predicted_label)
        final_real.append(real_label)




        predicted_score = scaler.inverse_transform(prediction[1])

        score = 0.
        if predicted_label == 1:
            score = predicted_score[0][0]


        #print(predicted_label, y_score[i][0], score)
        final_score_real.append(y_score[i][0])
        final_score_pred.append(score)





    print('Confusion Matrix')
    print(confusion_matrix(final_real, final_pred))
    print('Classification Report')
    print(classification_report(final_real, final_pred))
    print('mean_squared_error Report')
    print(mean_squared_error(final_score_real, final_score_pred))


def model_testing_2inputs(model, x_val1, x_val2, y_val, nlabel=0):
    final_pred = []
    final_real = []

    for i in range(0, len(x_val1)):
        prediction = model.predict([np.array([x_val1[i]]), np.array([x_val2[i]])])
        predicted_label = np.argmax(prediction[nlabel])

        real_label = np.argmax(y_val[i])

        final_pred.append(predicted_label)
        final_real.append(real_label)

    print('Confusion Matrix')
    print(confusion_matrix(final_real, final_pred))
    print('Classification Report')
    print(classification_report(final_real, final_pred))



def model_testing_Ninputs(model, x_val_list, y_val, nlabel=0):
    final_pred = []
    final_real = []
    print("MODEL TESTING:", x_val_list[0].shape)
    for i in range(0, len(x_val_list[0])):

        input = []
        for x_val in x_val_list:
            input.append(np.array([x_val[i]]))

        prediction = model.predict(input)

        predicted_label = np.argmax(prediction[nlabel])

        real_label = np.argmax(y_val[i])

        final_pred.append(predicted_label)
        final_real.append(real_label)



    print('Confusion Matrix')
    print(confusion_matrix(final_real, final_pred))
    print('Classification Report')
    print(classification_report(final_real, final_pred))

    #print(text)
