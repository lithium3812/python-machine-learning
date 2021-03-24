import re
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
import pandas as pd


def preprocessor(text):
    """
    Clean up text data, mainly removing HTML tags or symbols
    """
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text


def tokenizer(text):
    """
    Tokenize by word, splitting at space
    """
    return text.split()


porter = PorterStemmer()
def tokenizer_porter(text):
    """
    Stemming tokens with the simplest stemming algorithm
    """
    stems = [porter.stem(word) for word in text.split()]
    return stems
