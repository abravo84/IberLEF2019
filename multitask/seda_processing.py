import spacy, re, html

from spacymoji import Emoji
from random import shuffle

URL="URL"
USER="@USUARIO"
EMOJI="@EMOJI_"

# ES Core: https://spacy.io/models/es
# Emoji lib: https://github.com/ines/spacymoji
# WORDNET TUTORIAL: https://pythonprogramming.net/wordnet-nltk-tutorial/

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

    return tweet_text.replace("#", " #").replace("•", " • ").replace("/", " / ").replace("(", " ( ").replace(")", " ) ").replace("|", " | ").replace("¡", " ¡ ").replace("¿", " ¿ ").replace("!", " ! ").replace("?", " ? ").replace("\"", " \" ").replace("'", " ' ").replace("‘", " ' ").replace("’", " ' ").replace("-", " - ").replace("–", " - ").replace(";", "; ").replace("#", "HASHTAG_SYMBOL")

def unreplace_tweet(tweet_text):

    return tweet_text.replace("HASHTAG_SYMBOL", "#")

def get_header():
    header = []
    header.append("ID")
    header.append("TOKEN")
    header.append("NUM_TOKEN")
    header.append("POS")
    header.append("LEMA")
    header.append("DEP")
    header.append("PARENT")
    header.append("OFF")
    return "\t".join(header)


def preprocess_tweet(nlp, tweet):
    sent = []
    tweet_text = replace_tweet(tweet)
    doc = nlp(tweet_text)
    for token in doc:
        feat = []
        if token.pos_ == "SPACE":
            continue
        word = unreplace_tweet(str(token))
        if word.startswith("@"):
            word = USER
        elif word.startswith("http"):
            word = URL
        elif token._.is_emoji:
            word = EMOJI + word

        feat.append(word)
        feat.append(str(token.i))
        feat.append(token.pos_)
        feat.append(token.lemma_.lower())
        feat.append(token.dep_)
        feat.append(str(token.head.i))


        #list_syn = wn.synsets(word, pos=token.pos_)

        #feat = feat + list_syn

        sent.append(feat)

    return sent

def get_task_data_for_class_task(nlp, filepath, output_file, header, random, tweet_col, label_col=None):
    tweets = []
    #labels = []

    lines = open(filepath).readlines() # List of lines; lines[0] = "first line"

    if header:
        lines = lines[1:]


    ofile = open(output_file, "w")

    ofile.write(get_header()+ "\n")

    if random:
        shuffle(lines)

    for line in lines:
        sent = []
        ls = line.strip().split("\t")
        id = ls[0]
        tweet = ls[tweet_col]
        label = "?"
        if label_col != None:
            label = ls[label_col]
            #labels.append(label)


        sent_pre = preprocess_tweet(nlp, tweet)

        for feat in sent_pre:
            feat.insert(0,id)
            feat.append(label)
            ofile.write("\t".join(feat) + "\n")
            ofile.flush()

        ofile.write("\n")

    ofile.close()


"""
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
    return data, labels"""

def get_spacy_nlp(core, emojis=True):
    nlp = spacy.load(core)
    if emojis:
        emoji = Emoji(nlp)
        nlp.add_pipe(emoji, first=True)

    return nlp



if __name__ == '__main__':

    nlp = get_spacy_nlp('es_core_news_md', True)

    train_file = "/home/upf/corpora/SEMEVAL19_Task6/offenseval-training-v1.tsv"
    output_file = "/home/upf/corpora/SEMEVAL19_Task6/offenseval-training-seda_test.tsv"

    for l in open(train_file):
        print(l)

    get_task_data_for_class_task(nlp=nlp,
                                 filepath=train_file,
                                 output_file= output_file,
                                 header=True,
                                 random=False,
                                 tweet_col=1,
                                 label_col=2)