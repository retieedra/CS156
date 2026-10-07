import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

features = ["Pclass", "Age", "SibSp"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, feature in enumerate(features):
    # vary one predictor
    values = np.linspace(df[feature].min(), df[feature].max(), 100)

    # keep the others at their medians
    inputs = pd.DataFrame({
        "Pclass": np.full(100, df["Pclass"].median()),
        "Age": np.full(100, df["Age"].median()),
        "SibSp": np.full(100, df["SibSp"].median())
    })
    inputs[feature] = values

    # predict fares
    predicted_fares = model.predict(inputs)

    # plot data and fitted line
    axes[i].scatter(
        df[feature], df["Fare"],
        alpha=0.25, s=15, label="observed fares"
    )
    axes[i].plot(
        values, predicted_fares,
        color="red", label="others fixed at medians"
    )
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("fare (£)")
    axes[i].legend(fontsize=8)

plt.suptitle("conditional fitted lines")
plt.tight_layout()
plt.show()
