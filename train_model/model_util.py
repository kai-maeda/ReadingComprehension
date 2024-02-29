from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Input, Dense, Dropout, Flatten, Lambda, Embedding,Convolution1D, MaxPooling1D
from keras.initializers import RandomNormal
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from preprocess import encode_data
import math
import joblib
import re
import PyPDF2
import os
import fitz 






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

def read_pdf(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check if the request was successful

        with open('temp.pdf', 'wb') as pdf_file:
            pdf_file.write(response.content)

        # Extract text from the PDF
        text = ''
        with open('temp.pdf', 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfFileReader(pdf_file)
            for page_num in range(pdf_reader.numPages):
                text += pdf_reader.getPage(page_num).extractText()

        return text
    except requests.exceptions.HTTPError as errh:
        print("HTTP Error:", errh)
    except requests.exceptions.ConnectionError as errc:
        print("Error Connecting:", errc)
    except requests.exceptions.Timeout as errt:
        print("Timeout Error:", errt)
    except requests.exceptions.RequestException as err:
        print("Something went wrong:", err)
    finally:
        # Cleanup: Remove temporary PDF file
        try:
            os.remove('temp.pdf')
        except OSError:
            pass

def extract_text_from_body(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove all anchor tags
    for a_tag in soup.find_all('a'):
        a_tag.decompose()

    # Get text from the body
    body_text = soup.body.get_text(separator='\n', strip=True)
    words_only = re.findall(r'\b[a-zA-Z]+\b', body_text)
    cleaned_text = ' '.join(words_only)
    return cleaned_text

def extract_text_from_pdf(pdf_path):
    pdf_document = None
    try:
        pdf_document = fitz.open(pdf_path)
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None
    finally:
        # Close the PDF file
        if pdf_document:
            pdf_document.close()


def web_scrape(url, loaded_model, adaboost_model, maxlen,vocab) -> int:
    if url.endswith('.pdf'):
        return -1
    webpage_content = read_webpage(url)
    if webpage_content:
        body_text = extract_text_from_body(webpage_content)
        body_text = body_text.split()
        start= loaded_model.predict(encode_data(body_text, maxlen, vocab))
        n = len(start)
        easy = np.count_nonzero(np.argmax(start, axis = 1) == 0) / n
        medium = np.count_nonzero(np.argmax(start, axis = 1) == 1) / n
        hard =np.count_nonzero(np.argmax(start, axis = 1) == 2) / n
        points = np.array([[easy,medium,hard]])
        ans = 10 - adaboost_model.predict(points)
        return ans
    else:
        print(f"Failed to retrieve the content from {url}")
        return -1

def setup_weights(loaded_model,maxlen,vocab):
    example = pd.read_csv('CLEAR Corpus 6.01 - CLEAR Corpus 6.01.csv')
    columns_keep = ['Excerpt', 'BT Easiness']
    df_filtered = example[columns_keep]
    df_filtered['BT Easiness'] += 3.7
    df_filtered['BT Easiness'] *= 1.85
    df_filtered['BT Easiness'] = df_filtered['BT Easiness'].clip(lower=0, upper=10)
    
    data_points_x = []
    # chop = 50
    
    
    for index, row in df_filtered.iterrows():
        # if len(data_points_x ) ==  chop : break
        record = [0,0,0]
        temp= row['Excerpt'].split()
        prediction = loaded_model.predict(encode_data(temp, maxlen, vocab))
        n = len(prediction)

        record[0] = np.count_nonzero(np.argmax(prediction, axis = 1) == 0) / n
        record[1] = np.count_nonzero(np.argmax(prediction, axis = 1) == 1) / n 
        record[2] =np.count_nonzero(np.argmax(prediction, axis = 1) == 2) / n
        data_points_x.append(record)
        
        
        
    reform = np.array(data_points_x)
    
    y_values = np.array(df_filtered['BT Easiness'])
    
    base_model = DecisionTreeRegressor(max_depth=4)
    adaboost_model = AdaBoostRegressor(base_model, n_estimators=500, learning_rate=0.1)
    adaboost_model.fit(reform, y_values)
    joblib.dump(adaboost_model, 'params/adaboost_model.pkl')

def test_accuracy(loaded_model,maxlen,vocab,adaboost_model):
    example = pd.read_csv('CLEAR Corpus 6.01 - CLEAR Corpus 6.01.csv')
    columns_keep = ['Excerpt', 'BT Easiness']
    df_filtered = example[columns_keep]
    df_filtered['BT Easiness'] += 3.7
    df_filtered['BT Easiness'] *= 1.85
    df_filtered['BT Easiness'] = df_filtered['BT Easiness'].clip(lower=0, upper=10)
    
    data_points_x = []
    chop = 50
    
    for index, row in df_filtered.iterrows():
        if index < chop: 
            continue
        if len(data_points_x ) ==  chop : break
        record = [0,0,0]
        temp = row['Excerpt'].split()
        prediction = loaded_model.predict(encode_data(temp, maxlen, vocab))
        n = len(prediction)

        record[0] = np.count_nonzero(np.argmax(prediction, axis = 1) == 0) / n
        record[1] = np.count_nonzero(np.argmax(prediction, axis = 1) == 1) / n
        record[2] =np.count_nonzero(np.argmax(prediction, axis = 1) == 2) / n
        data_points_x.append(record)
        
    reform = np.array(data_points_x)
    
    correct_values = np.array(df_filtered['BT Easiness'])[chop:2*chop]
    y_values = adaboost_model.predict(reform)
    correct = 0
    n = len(y_values)
    dstr = [0]*10
    for i in range(n):
        if abs(y_values[i] - correct_values[i]) < 1:
            correct += 1
        dstr[math.floor(correct_values[i])] += 1
            
    
    print("distribution of y-values", dstr)
    print("accuracy = ", correct / n)
   