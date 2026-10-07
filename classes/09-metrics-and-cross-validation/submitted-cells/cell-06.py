linear_preds = linear_clf.predict_proba(x_test)[:, 1]
svm_preds = svm_clf.predict_proba(x_test)[:, 1]

linear_curve = metrics.roc_curve(y_test, linear_preds)
svm_curve = metrics.roc_curve(y_test, svm_preds)

plt.plot(linear_curve[0], linear_curve[1], label="Logistic ROC-AUC: %.2f" % metrics.roc_auc_score(y_test, linear_preds))
plt.plot(svm_curve[0], svm_curve[1], label="SVM ROC-AUC: %.2f" % metrics.roc_auc_score(y_test, svm_preds))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
