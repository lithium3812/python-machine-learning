import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


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


df = pd.read_csv('movie_data.csv')
df['review'] = df['review'].apply(preprocessor)
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

#nltk.download('stopwords')
stop = stopwords.words('english')

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(random_state=0, solver='liblinear'))])

# print(lr_tfidf.get_params())

param_grid = [{
    'vect__ngram_range': [(1, 1)],
    'vect__stop_words': [stop, None],
    'vect__tokenizer': [tokenizer, tokenizer_porter],
    'clf__penalty': ['l1', 'l2'],
    'clf__C': [1.0, 10.0, 100.0]
},
{
    'vect__ngram_range': [(1, 1)],
    'vect__stop_words': [stop, None],
    'vect__tokenizer': [tokenizer, tokenizer_porter],
    'vect__use_idf': [False],
    'vect__norm': [None],
    'clf__penalty': ['l1', 'l2'],
    'clf__C': [1.0, 10.0, 100.0]
}]

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=2, n_jobs=-1)
gs_lr_tfidf.fit(X_train, y_train)

print('Best parameter set:\n', gs_lr_tfidf.best_params_)
print(f'CV accuracy: {gs_lr_tfidf.best_score_: .3f}')

clf = gs_lr_tfidf.best_estimator_
print(f'Test accuracy: {clf.score(X_test, y_test): .3f}')