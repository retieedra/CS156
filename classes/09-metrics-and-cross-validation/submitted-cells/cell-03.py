from sklearn import linear_model, metrics, svm

linear_clf = linear_model.LogisticRegression()
linear_clf.fit(x_train, y_train)

svm_clf = svm.SVC(gamma=2, C=1, probability=True)
svm_clf.fit(x_train, y_train)

print(linear_clf.score(x_test, y_test), svm_clf.score(x_test, y_test))
