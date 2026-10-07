#logistic regression here

light_colors = ["#c6dbef", "#c7e9c0", "#fdd0a2"]
dark_colors = ["#08519c", "#006d2c", "#a63603"]

plt.contourf(
    xx, yy, regions,
    levels=[-0.5, 0.5, 1.5, 2.5],
    colors=light_colors
)

for label, species in enumerate(iris.target_names):
    flowers = X[y == label]
    plt.scatter(
        flowers[:, 0],
        flowers[:, 1],
        color=dark_colors[label],
        label=species
    )

plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.title("Logistic regression")
plt.legend()
plt.show()
