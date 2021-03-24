import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


"""
Bag-of-words model
"""

vectorizer = CountVectorizer()
# sample text data
docs = np.array([
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining, the weather is sweet, and one and one is two'
])

# create feature vectors from the original text data
bag = vectorizer.fit_transform(docs)

print("Count of each word:\n", vectorizer.vocabulary_)
print("Generated feature vectors:\n", bag.toarray())

"""
tf-idf
"""

# tf-idf will be normalized with L2 norm at the end
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)

# Generate tf-idf from bag-of-words
feat_vecs = tfidf.fit_transform(bag)

np.set_printoptions(precision=2)
print("tf-idf vectors:\n", feat_vecs.toarray())