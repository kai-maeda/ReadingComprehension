from __future__ import print_function
from __future__ import division
import json
import model_util
import numpy as np
import preprocess
from model_util import setup_weights, web_scrape, test_accuracy
import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import save_model
import joblib
from flask import Flask, request, jsonify



save = False
np.random.seed(123)  # for reproducibility

# set parameters:

subset = None

# Whether to save model parameters
#save=False
model_name_path = 'params/crepe_model3.h5'
model_weights_path = 'params/crepe_model_weights_with_test_v1.0.h5'

# Maximum length. Longer gets chopped. Shorter gets padded.
maxlen = 21

# Model params
# Filters for conv layers
nb_filter = 32
# Number of units in the dense layer
dense_outputs = 256
# Conv layer kernel size
filter_kernels = [3, 3, 2, 2, 2, 2]
# Number of units in the final output layer. Number of classes.
cat_output = 3

# Compile/fit params
batch_size = 750
nb_epoch = 1000


print('Creating vocab...')
vocab, reverse_vocab, vocab_size, alphabet = preprocess.create_vocab_set()
print('Build model...')
model = model_util.create_model(filter_kernels, dense_outputs, maxlen, vocab_size,
                                nb_filter, cat_output)
model.load_weights(model_weights_path)
# prediction = model.predict(preprocess.encode_data(["laptop","obnoxious", "camembert", "portuguese", "penicillin", "protactinium"], maxlen, vocab))
# print(prediction)
#weights set up with CLEAR CORPUS dataset
# setup_weights(model,maxlen,vocab)
adaboost_model = joblib.load('params/adaboost_model.pkl')
# test_accuracy(model,maxlen,vocab,adaboost_model)
# text = "https://pbskids.org"
# text2 = "https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.33.411"
# webpage_content = web_scrape(text, model, adaboost_model, maxlen,vocab)
# webpage_content = web_scrape(text2, model, adaboost_model, maxlen,vocab)


app = Flask(__name__)

@app.route('/word_count', methods=['POST'])
def word_count():
    try:
        data = request.get_json()
        if 'url' not in data:
            return jsonify({'error': 'url not provided'}), 400

        text = data['url']
        print(text)
        difficulty = web_scrape(text, model, adaboost_model, maxlen,vocab)
        response = {'difficulty': float(difficulty)}
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


if save:
    print('Loading data...')
    # Expect x to be a list of sentences. Y to be index of the categories.
    (xt, yt) = preprocess.load_data()


    #model.add(Dropout(0.4))
    #model=keras.models.load_model('params\crepe_model.h5','r+')
    # Encode data
    xt = preprocess.encode_data(xt, maxlen, vocab)
    #X_train, X_test, Y_train, Y_test = train_test_split(xt, yt, test_size=0.3)
    # x_test = preprocess.encode_data(x_test, maxlen, vocab)

    print('Chars vocab: {}'.format(alphabet))
    print('Chars vocab size: {}'.format(vocab_size))
    print('X_train.shape: {}'.format(xt.shape))
    #model.summary()
    print('Fit model...')
    model.fit(xt,yt,batch_size=batch_size, epochs=nb_epoch, shuffle=True)

    prediction = model.predict(preprocess.encode_data(["laptop","obnoxious", "camembert", "portuguese", "penicillin", "protactinium"], maxlen, vocab))
    print(prediction)

    scores = model.evaluate(xt, yt)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    #import pickle
    if save:
        print('Saving model params...')
        model.save_weights(model_weights_path)


