def leaf_gini_impurity(leaf_ys):
    if len(leaf_ys) == 0:
        return 0
    else:
        return 1 - (sum(leaf_ys==0)/len(leaf_ys))**2 - (sum(leaf_ys==1)/len(leaf_ys))**2

def weighted_gini_impurity(left_ys, right_ys):
    total = len(left_ys) + len(right_ys)
    left_leaf = leaf_gini_impurity(left_ys) * len(left_ys)/total
    right_leaf = leaf_gini_impurity(right_ys) * len(right_ys)/total
    return left_leaf + right_leaf

# You only need to use this function directly to solve this question:
def gini_impurity(x_data, y_data, axis, threshold):
    # The axis argument should be 0 or 1
    # The threshold should be a real valued number
    left_mask = x_data[:, axis] <= threshold
    right_mask = x_data[:, axis] > threshold
    return weighted_gini_impurity(y_data[left_mask], y_data[right_mask])

# Here's a sample way to use this function
print(gini_impurity(x_data, y_data, axis=1, threshold=0.8))

# my guess from before, i feel like i have the axisis wrong?
print(gini_impurity(x_data, y_data, axis=0, threshold=0.0))
print(gini_impurity(x_data, y_data, axis=1, threshold=0.0))

best_gini = 1
best_axis = None
best_threshold = None

#i just ended up testing this using 0.01 increments
for axis in [0, 1]:

    threshold = -1

    while threshold <= 2.5:

        score = gini_impurity(x_data, y_data, axis, threshold)

        if score < best_gini:
            best_gini = score
            best_axis = axis
            best_threshold = threshold

        threshold = threshold + 0.01

print("best axis:", best_axis)
print("best threshold:", best_threshold)
print("lowest gini:", best_gini)
