import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics, naive_bayes

# We train a model
model = naive_bayes.MultinomialNB()
model.fit(vec_texts, labels)
pred_labels = model.predict(vec_texts)

confusion_mat = metrics.confusion_matrix(labels, pred_labels)
sns.heatmap(confusion_mat, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
