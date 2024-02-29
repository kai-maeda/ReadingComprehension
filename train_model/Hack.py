from __future__ import print_function
from __future__ import division
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Input, Dense, Dropout, Flatten, Lambda, Embedding,Convolution1D, MaxPooling1D
from keras.initializers import RandomNormal
from sklearn.linear_model import LinearRegression
import requests
from bs4 import BeautifulSoup

def create_model(filter_kernels, dense_outputs, maxlen, vocab_size, nb_filter, cat_output):
    initializer = RandomNormal(mean=0.0, stddev=0.05, seed=None)

    # Define what the input shape looks like
    inputs = Input(shape=(maxlen,), dtype='int64')

    # Option one:
    # Uncomment following code to use a lambda layer to create a onehot encoding of a sequence of characters on the fly.
    # Holding one-hot encodings in memory is very inefficient.
    # The output_shape of embedded layer will be: batch x maxlen x vocab_size
    #
    import tensorflow as tf

    def one_hot(x):
        return tf.one_hot(x, vocab_size, on_value=1.0, off_value=0.0, axis=-1, dtype=tf.float32)

    def one_hot_outshape(in_shape):
        return in_shape[0], in_shape[1], vocab_size

    embedded = Lambda(one_hot, output_shape=one_hot_outshape)(inputs)

    # Option two:
    # Or, simply use Embedding layer as following instead of use lambda to create one-hot layer
    # Think of it as a one-hot embedding and a linear layer mashed into a single layer.
    # See discussion here: https://github.com/keras-team/keras/issues/4838
    # Note this will introduce one extra layer of weights (of size vocab_size x vocab_size = 69*69 = 4761)
    # embedded = Embedding(input_dim=vocab_size, output_dim=vocab_size)(inputs)

    # All the convolutional layers...
    conv = Convolution1D(filters=nb_filter, kernel_size=3, kernel_initializer=initializer,
                         padding='valid', activation='relu',
                         input_shape=(maxlen, vocab_size))(embedded)
    # conv = MaxPooling1D(pool_size=2)(conv)

    conv1 = Convolution1D(filters=nb_filter, kernel_size=3, kernel_initializer=initializer,
                          padding='valid', activation='relu')(conv)
    conv1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Convolution1D(filters=nb_filter, kernel_size=2, kernel_initializer=initializer,
                          padding='valid', activation='relu')(conv1)

    conv3 = Convolution1D(filters=nb_filter, kernel_size=2, kernel_initializer=initializer,
                          padding='valid', activation='tanh')(conv2)
    conv3 = MaxPooling1D(pool_size=2)(conv3)
    
    conv4 = Convolution1D(filters=nb_filter, kernel_size=2, kernel_initializer=initializer,
                          padding='valid', activation='relu')(conv3)
    conv4 = MaxPooling1D(pool_size=2)(conv4)
        
    #conv4 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[4], kernel_initializer=initializer,
    #                      padding='valid', activation='relu')(conv3)

    conv5 = Convolution1D(filters=nb_filter, kernel_size=2, kernel_initializer=initializer,
                          padding='valid', activation='relu')(conv3)
    #conv5 = MaxPooling1D(pool_size=3)(conv5)
    conv5 = Flatten()(conv5)

    # Two dense layers with dropout of .5
    z = Dropout(0.25)(Dense(dense_outputs, activation='relu')(conv5))
    z = Dropout(0.25)(Dense(dense_outputs, activation='relu')(z))

    # Output dense layer with softmax activation
    pred = Dense(cat_output, activation='softmax', name='output')(z)

    model = Model(inputs=inputs, outputs=pred)

    sgd = SGD(lr=0.01, momentum=0.9)
    adam = Adam(lr=0.001)  # Feel free to use SGD above. I found Adam with lr=0.001 is faster than SGD with lr=0.01
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model
import string
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical


def load_data():
    data = pd.read_csv('I159729.csv')
    data = data.dropna()
    data.drop(data.columns[[1, 2, 3, 4, 6, 7, 8]], axis=1, inplace=True)
    data["Word"] = data["Word"].str.lower().replace("'", "")

    data["Difficulty"] = 0
    data.loc[data["I_Zscore"] > -0.3, ["Difficulty"]] = 1
    data.loc[data["I_Zscore"] > 0.3, ["Difficulty"]] = 2
    # data.loc[data["I_Zscore"] > 0.3, ["Difficulty"]] = 3
    
    data_x = data["Word"]
    data_x = np.array(data_x)
    data_y = data["Difficulty"]
    data_y = np.array(data_y)
    data_y = to_categorical(data_y)
    
    return data_x, data_y


def encode_data(x, maxlen, vocab):
    # Iterate over the loaded data and create a matrix of size (len(x), maxlen)
    # Each character is encoded into a one-hot array later at the lambda layer.
    # Chars not in the vocab are encoded as -1, into an all zero vector.

    input_data = np.zeros((len(x), maxlen), dtype=np.int)
    for dix, sent in enumerate(x):
        counter = 0
        for c in sent:
            if counter >= maxlen:
                pass
            else:
                ix = vocab.get(c, -1)  # get index from vocab dictionary, if not in vocab, return -1
                input_data[dix, counter] = ix
                counter += 1
    return input_data


def create_vocab_set():

    alphabet = set(list(string.ascii_lowercase))
    vocab_size = len(alphabet)
    vocab = {}
    reverse_vocab = {}
    for ix, t in enumerate(alphabet):
        vocab[t] = ix
        reverse_vocab[ix] = t

    return vocab, reverse_vocab, vocab_size, alphabet

import json

import numpy as np

import keras
from sklearn.model_selection import train_test_split

def word_weight():

    print('Loading data...')
    # Expect x to be a list of sentences. Y to be index of the categories.
    (xt, yt) = load_data()

    print('Creating vocab...')
    vocab, reverse_vocab, vocab_size, alphabet = create_vocab_set()

    print('Build model...')
    model = create_model(filter_kernels, dense_outputs, maxlen, vocab_size,
                                  nb_filter, cat_output)
    #model.add(Dropout(0.4))
    #model=keras.models.load_model('params\crepe_model.h5','r+')
    # Encode data
    xt = encode_data(xt, maxlen, vocab)
    #X_train, X_test, Y_train, Y_test = train_test_split(xt, yt, test_size=0.3)
    # x_test = preprocess.encode_data(x_test, maxlen, vocab)

    print('Chars vocab: {}'.format(alphabet))
    print('Chars vocab size: {}'.format(vocab_size))
    print('X_train.shape: {}'.format(xt.shape))
    #model.summary()
    print('Fit model...')
    model.fit(xt,yt,batch_size=batch_size, epochs=nb_epoch, shuffle=True)

    prediction = model.predict(encode_data(["laptop","obnoxious", "camembert", "portuguese", "penicillin", "protactinium"], maxlen, vocab))
    print(prediction)

    scores = model.evaluate(xt, yt)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    #import pickle
    if save:
        print('Saving model params...')
        model.save_weights(model_weights_path)
    
'''
data_x, data_y = load_data()
print(data_x)
print(data_y)
# Max len -> 21

vocab, reverse_vocab, vocab_size, alphabet = create_vocab_set()
print(vocab)

input_data = encode_data(data_x, 21, vocab)
print(input_data)
'''

def read_webpage(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        return response.text
    except requests.exceptions.HTTPError as errh:
        print("HTTP Error:", errh)
    except requests.exceptions.ConnectionError as errc:
        print("Error Connecting:", errc)
    except requests.exceptions.Timeout as errt:
        print("Timeout Error:", errt)
    except requests.exceptions.RequestException as err:
        print("Something went wrong:", err)

def extract_text_from_body(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove all anchor tags
    for a_tag in soup.find_all('a'):
        a_tag.decompose()

    # Get text from the body
    body_text = soup.body.get_text(separator='\n', strip=True)
    return body_text

def web_scrape(url):
    webpage_content = read_webpage(url)
    weights = [-0.00150539, -0.02668869, -0.08312362]
    
    if webpage_content:
        body_text = extract_text_from_body(webpage_content)
        body_text = body_text.split()
        #print(body_text)
        start= loaded_model.predict(encode_data(body_text, maxlen, vocab))
        easy = np.count_nonzero(np.argmax(start, axis = 1) == 0)
        medium = np.count_nonzero(np.argmax(start, axis = 1) == 1)
        hard =np.count_nonzero(np.argmax(start, axis = 1) == 2)
        points = np.array([easy,medium,hard])
        result = np.dot(points,weights) + 5.465
        print(result)
    else:
        print(f"Failed to retrieve the content from {url}")

def setup_weights():
    example = pd.read_csv('CLEAR Corpus 6.01 - CLEAR Corpus 6.01.csv')
    columns_keep = ['Excerpt', 'BT Easiness']
    df_filtered = example[columns_keep]
    df_filtered['BT Easiness'] += 5
    df_filtered['BT Easiness'] = df_filtered['BT Easiness'].clip(lower=0, upper=10)
    
    data_points_x = []
    
    for index, row in df_filtered.iterrows():
        record = [0,0,0]
        temp= row['Excerpt'].split()
        prediction = loaded_model.predict(encode_data(example, maxlen, vocab))

        record[0] = np.count_nonzero(np.argmax(prediction, axis = 1) == 0)
        record[1] = np.count_nonzero(np.argmax(prediction, axis = 1) == 1)
        record[2] =np.count_nonzero(np.argmax(prediction, axis = 1) == 2)
        data_points_x.append(record)
        
        
        
    reform = np.array(data_points_x)
    
    y_values = np.array(df_filtered['BT Easiness'])
    
    model = LinearRegression()
    reform = reform.reshape(-1, 3)
    model.fit(reform, y_values)

    # Get the weights (coefficients) and intercept
    weights = model.coef_
    intercept = model.intercept_
    print("Weights:", weights)
    print("Intercept:", intercept)
    return weights,intercept
    # Display the weights and intercept


np.random.seed(123)  # for reproducibility
subset = None
save = True
model_name_path = 'params/crepe_model3.h5'
model_weights_path = 'params/crepe_model_weights_with_test_v1.0.h5'
maxlen = 21
nb_filter = 32
dense_outputs = 256
filter_kernels = [3, 3, 2, 2, 2, 2]
cat_output = 3
batch_size = 750
nb_epoch = 100
train_model = True
if(train_model):
    loaded_model = create_model(filter_kernels, dense_outputs, maxlen, vocab_size, nb_filter, cat_output)
    loaded_model.load_weights(model_weights_path)
    prediction = loaded_model.predict(encode_data(["fast","swift", "rapid", "speedy", "expeditious", "alacrituos"], maxlen, vocab))
    print(prediction)

weights, intercept = setup_weights()
print(weights)
text = "https://pbskids.org"
webpage_content = web_scrape(text)