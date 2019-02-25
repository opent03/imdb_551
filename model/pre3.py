import glob
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re, unicodedata
from nltk.corpus import stopwords
import pandas as pd


def getTrain(tr):
    'Get train'
    print("acquiring train")
    raw = []
    for filename in glob.iglob(tr + '/**', recursive=True):
        if os.path.isfile(filename):  # filter dirs
            f = open(filename, 'r')
            raw.append(f.read().lower())
            f.close()
    print("train acquired")
    raw = np.asarray(raw)
    print(raw.shape)
    return raw


def getTest():
    'Get test'
    print("acquiring test")
    raw = []
    for filename in glob.iglob('test/**', recursive=True):
        if os.path.isfile(filename):  # filter dirs
            f = open(filename, 'r')
            raw.append([f.read().lower(), filename.split('/')[1].split('.')[0]])
            f.close()
    print("test acquired")
    raw = np.asarray(raw)
    print(raw.shape)
    return raw


def clean_text(text):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    #text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    #text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    #text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text


def strip_html(text):
    'Bye html tags'
    #soup = BeautifulSoup(text, "html.parser")
    #return soup.get_text()
    return re.sub("<.*?>", " ", text)

def remove_sq_brackets(text):
    'sq = square'
    return re.sub('\[[^]]*\]', '', text)


def remove_non_ascii(words):
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    'Do I have to comment on what this does too'
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def remove_stopwords(words):
    'Intuitive'
    return ' '.join([word for word in words.split(' ') if word not in stopwords.words('english')])


def stem(sentence):
    'Does exactly what the method says it does'
    stemmer = PorterStemmer()
    n = []
    for word in sentence.split(' '):
        n.append(stemmer.stem(word))
    return ' '.join(n)


def preprocess(X):
    'raw preprocessing'
    print('Preprocess')
    print("---Cleaning text starts---")
    for i in range(len(X)):

        X[i] = strip_html(X[i])
        X[i] = remove_sq_brackets(X[i])
        X[i] = ''.join(remove_non_ascii(X[i]))
        X[i] = re.sub(r'(?<=[.?!])( +|\Z)', r' ', X[i])

        X[i] = re.sub(r'[^\w\s]', ' ', X[i])
        X[i] = re.sub(' +', ' ', X[i])
        X[i] = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", X[i])
        X[i] = re.sub(r"what's", "what is ", X[i])
        X[i] = re.sub(r"\'s", " ", X[i])
        X[i] = re.sub(r"\'ve", " have ", X[i])
        X[i] = re.sub(r"can't", "cannot ", X[i])
        X[i] = re.sub(r"n't", " not ", X[i])
        X[i] = re.sub(r"i'm", "i am ", X[i])
        X[i] = re.sub(r"\'re", " are ", X[i])
        X[i] = re.sub(r"\'d", " would ", X[i])
        X[i] = re.sub(r"\'ll", " will ", X[i])
        X[i] = re.sub(r",", " ", X[i])
        X[i] = re.sub(r"\.", " ", X[i])
        X[i] = re.sub(r"!", " ! ", X[i])
        X[i] = re.sub(r"\/", " ", X[i])
        X[i] = re.sub(r"\^", " ^ ", X[i])
        X[i] = re.sub(r"\+", " + ", X[i])
        X[i] = re.sub(r"\-", " - ", X[i])
        X[i] = re.sub(r"\=", " = ", X[i])
        X[i] = re.sub(r"'", " ", X[i])
        X[i] = re.sub(r"(\d+)(k)", r"\g<1>000", X[i])
        X[i] = re.sub(r":", " : ", X[i])
        X[i] = re.sub(r" e g ", " eg ", X[i])
        X[i] = re.sub(r" b g ", " bg ", X[i])
        X[i] = re.sub(r" u s ", " american ", X[i])
        X[i] = re.sub(r"\0s", "0", X[i])
        X[i] = re.sub(r" 9 11 ", "911", X[i])
        X[i] = re.sub(r"e - mail", "email", X[i])
        X[i] = re.sub(r"j k", "jk", X[i])
        X[i] = re.sub(r"\s{2,}", " ", X[i])
        #X[i] = remove_stopwords(X[i])

        if i%500 == 0 : print(i/500, '%')
    f = open('pre_stemming.txt', 'w')
    for i in X:
        f.write(str(i) + '\n')
    f.close()
    for i in range(len(X)):
        #X[i] = stem(X[i])
        X[i] = clean_text(X[i])
        if i % 500 == 0: print(i / 500, '%')
    f = open('post_stemming.txt', 'w')
    for i in X:
        f.write(str(i) + '\n')
    f.close()
    return X


def tfidf_pca(X: object, y: object, n_components: object = 30, nopca: object = False) -> object:
    print("tfidf-ing...")
    count_vect = CountVectorizer(ngram_range=(1, 2), min_df=1)
    traincount = X[:50000]
    count_vect.fit(traincount)
    print("---- vectorizing start! ---")
    X_train_counts = count_vect.transform(X)
    print("--- done vectorizing! --- ")
    tf_transformer = TfidfTransformer(use_idf=True)
    print("--- transforming start! ---")
    X_train_tf = tf_transformer.fit_transform(X_train_counts)
    print("--- done transforming! ---")
    print("--- done tfidf-ing! ---")

    #X_train_tf = X_train_counts
    print("--- select chi2 features start! ---")
    print(X_train_tf.shape, type(X_train_tf))
    select = SelectKBest(chi2, k=n_components)
    #X_train_tf = preprocessing.scale(X_train_tf, with_mean=False)


    X_train_tf_hd = X_train_tf[:25000]
    X_train_tf_tl = X_train_tf[25000:50000]
    X_train_tf_lol = X_train_tf[50000:]
    select.fit(X_train_tf_hd, y)

    X_train_tf_hd = select.transform(X_train_tf_hd)
    X_train_tf_tl = select.transform(X_train_tf_tl)
    X_train_tf_lol = select.transform(X_train_tf_lol)
    """
    svd = TruncatedSVD(n_components=5000)
    svd.fit(X_train_tf_hd)
    X_train_tf_hd = svd.transform(X_train_tf_hd)
    X_train_tf_tl = svd.transform(X_train_tf_tl)
    X_train_tf_lol = svd.transform(X_train_tf_lol)
"""
    print("--- select chi2 features done! ---")
    return X_train_tf_hd, X_train_tf_tl, X_train_tf_lol


def retrieve_and_pre(fromsave: object = True, tfidfpca: object = True, nopca: object = False, n_components: object = 35) -> object:
    'Preprocesses and does some PCA'
    y = [1 if x > 12500 else 0 for x in range(25000)]
    if not fromsave:
        train = getTrain('train') # list of sentences
        data = getTrain('data')
        test = getTest() # list of lists(sentences, filename)

        # extract test data along with file names
        names = test[:, -1]
        test = test[:, 0]

        # Merge test: and train to preprocess
        X = np.concatenate((train, test), axis=0)
        X = np.concatenate((X, data), axis=0)
        X = preprocess(X)
        np.save('pre_X_lol', X)
        np.save('names_lol', names)
    else:
        X = np.load('pre_X_lol.npy')
        names = np.load('names_lol.npy')
    # Sanity check

    if tfidfpca:
        X, test, lol = tfidf_pca(X, y, n_components=n_components, nopca=nopca)
        print("shape of X, ", X.shape)
        return X, y, test, names, lol

    # Do some splitting
        #X = (-1) * np.nan_to_num(np.log10(normalize(X+1, norm='max')))
        #scaler = MaxAbsScaler()
        #X = scaler.fit_transform(X)
    #X = preprocessing.scale(X)
    test = X[25000:50000]
    assert test.shape[0] == names.shape[0]
    lol = X[50000:]
    X = X[:25000]

    return X, y, test, names, lol


def dump_to_csv(model, test, names, filename):
    dump = []
    pred = model.predict(test)
    for i in range(len(pred)):
        dump.append([names[i], pred[i]])
    pd.DataFrame(np.asarray(dump)).to_csv(filename + '.csv')




















