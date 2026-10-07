from matplotlib import cm

fig, ax = plt.subplots(ncols=2, figsize=(18, 7))
fig.colorbar(cm.ScalarMappable(cmap="Blues"), ticks=np.linspace(0, 1, 11), ax=ax)
plot_decision_surface(linear_clf, (-1.5, 1.5), (-1.5, 1.5), ax[0], "Logistic Regression")
plot_decision_surface(svm_clf, (-1.5, 1.5), (-1.5, 1.5), ax[1], "SVM")
