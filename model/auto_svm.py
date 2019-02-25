from model.pre import retrieve_and_pre, dump_to_csv
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from keras.layers import Input, Dense
from keras.models import Model
import umap
import numpy as np
import pandas as pd



def main():
    X, y, test, names = retrieve_and_pre(fromsave=True, tfidfpca=True, nopca=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=1)


    encoding_dim = 500
    ncol = X_train.shape[1]
    input_dim = Input(shape=(ncol, ))
    encoded1 = Dense(3000, activation='relu')(input_dim)
    encoded2 = Dense(1000, activation='relu')(encoded1)
    encoded13 = Dense(encoding_dim, activation='relu')(encoded2)

    # Decoder Layers
    decoded1 = Dense(1000, activation='relu')(encoded13)
    decoded3 = Dense(3000, activation='relu')(decoded1)
    decoded13 = Dense(ncol, activation='sigmoid')(decoded3)

    # Combine Encoder and Deocder layers
    autoencoder = Model(inputs=input_dim, outputs=decoded13)

    # Compile the Model
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    print(autoencoder.summary())

    autoencoder.fit(X_train, X_train, nb_epoch=10, batch_size=32, shuffle=False, validation_data=(X_test, X_test))

    encoder = Model(inputs=input_dim, outputs=encoded13)
    encoded_input = Input(shape=(encoding_dim,))

    encoded_train = encoder.predict(X_train)
    encoded_test = encoder.predict(X_test)
    final_transform = encoder.predict(test)

    clf = SGDClassifier(max_iter=100, verbose=True)
    clf.fit(encoded_train, y_train)
    pred = clf.predict(encoded_test)

    scr = accuracy_score(y_test, pred)
    print("Accuracy score: %f%%" % (scr * 100))

    dump_to_csv(clf, final_transform, names, 'autosvm')


if __name__ == '__main__':
    main()
