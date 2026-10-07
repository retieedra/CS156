# load in dataset
import pandas as pd
from sklearn.feature_extraction import text
df = pd.read_csv("https://raw.githubusercontent.com/milindsoorya/Spam-Classifier-in-python/main/dataset/spam.csv", encoding='latin-1')
labels, texts = df['v1'], df['v2']
vectorizer = text.TfidfVectorizer()
vec_texts = vectorizer.fit_transform(texts)

# Uncomment and print one at a time
print(texts[:3])
# Hint: vec_texts is a sparse matrix; the coordinates are only shown for its "nonzero" entries
print(vec_texts[:3], vec_texts[:3].shape)
print(vectorizer.vocabulary_)
