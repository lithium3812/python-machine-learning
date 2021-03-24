import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.decomposition import LatentDirichletAllocation


"""
Topic modeling with Latent Dirichlet Allocation for movie review data
"""

df = pd.read_csv('movie_data.csv', encoding='utf-8')

# Maxmum document frequency 10% and only consider most frequent 5000 words
vect = CountVectorizer(stop_words='english', max_df=.1, max_features=5000)

X = vect.fit_transform(df['review'].values)

# n_components indicates the number of topics
lda = LatentDirichletAllocation(n_components=10, random_state=123, learning_method='batch')
X_topics = lda.fit_transform(X)

print("LDA components shape:", lda.components_.shape)

# Print the top 5 most important keywords for each 10 topic
n_top_words = 5
feature_names = vect.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic {topic_idx}:")
    print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))