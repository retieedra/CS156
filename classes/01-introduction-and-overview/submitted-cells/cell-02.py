import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

#step 1, load mnist as a pandas dataframe where each row is one handwritten digit stored as 784 pixels
mnist = fetch_openml("mnist_784", version=1, as_frame=True)
mnistdata = mnist.data
mnistdata["digit"] = mnist.target

#step 2, keep only the images labeled as 3 or 8
mnistdata = mnistdata[mnistdata["digit"].isin(["3", "8"])]

#step 3, select four images of each digit so we can see differences in the handwriting
threes = mnistdata[mnistdata["digit"] == "3"].head(4)
eights = mnistdata[mnistdata["digit"] == "8"].head(4)
examples = pd.concat([threes, eights])

#make eight plot spaces without assigning two variables on one line
plots = plt.subplots(2, 4)
axes = plots[1]

#turn each row of 784 pixels into a 28 by 28 image and let matplotlib use its default display settings
for number in range(len(examples)):
    row = examples.iloc[number]
    digit = row["digit"]
    pixels = row.drop("digit").to_numpy(dtype=float)
    image = pixels.reshape(28, 28)
    axis = axes.flat[number]
    axis.imshow(image)
    axis.set_title(digit)

#show the finished images
plt.show()
