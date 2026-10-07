import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

#step 1, load the iris data as a pandas dataframe where each row is one flower
iris = load_iris(as_frame=True)
iris_df = iris.frame
iris_df["species"] = iris.target_names[iris.target]

#step 2, an iris is a type of flower and a sepal is the leaf shaped part around it, the lengths are in centimeters, and this removes flowers with sepal lengths greater than 5 centimeters
iris_df = iris_df[iris_df["sepal length (cm)"] <= 5]

#step 3, plot sepal length against petal length and give each iris species a different color
for species, flowers in iris_df.groupby("species"):
    plt.scatter(
        flowers["sepal length (cm)"],
        flowers["petal length (cm)"],
        label=species
    )

#label and display the graph
plt.xlabel("sepal length in centimeters")
plt.ylabel("petal length in centimeters")
plt.legend()
plt.show()
