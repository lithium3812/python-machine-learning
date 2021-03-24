import numpy as np
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import pyprind
import pickle
import os


"""
Demonstrate Out-of-core learning
Use SDGClassifier as learning model
"""

stop = stopwords.words('english')

def tokenizer(text):
    """
    Clean up text data, mainly removing HTML tags or symbols, and stop words
    """
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def stream_docs(path):
    """
    Read data in a stream manner
    """
    with open(path, 'r', encoding='utf-8') as file:
        next(file)  # skip header
        for line in file:
            text, label = line[:-3], int(line[-2])
            yield text, label


def get_minibatch(doc_stream, size):
    """
    Load specified number od documents from a streamer
    """
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


# CountVectorizer cannot be used for out-of-core learning
# HashingVectorizer is independent of dataset
vect = HashingVectorizer(decode_error='ignore', n_features=2**21, preprocessor=None, tokenizer=tokenizer)

clf = SGDClassifier(loss='log', random_state=1)

doc_stream = stream_docs(path='movie_data.csv')

# Batch learning
pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])
for _ in range(45): # 45 interation of batch learning which contains 1000 documents at once
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

# The remaining 5000 documents are test data
X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print(f"Accuracy: {clf.score(X_test, y_test): .3f}")

# At the end, train the model with all data
clf = clf.partial_fit(X_test, y_test)

"""
Pickle the trained model
"""
dest = os.path.join('movieclassifier', 'pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
pickle.dump(stop, open(os.path.join(dest, 'stopwords.pkl'), 'wb'), protocol=4)
pickle.dump(clf, open(os.path.join(dest, 'classifier.pkl'), 'wb'), protocol=4)