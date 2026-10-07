def find_best_split(x, y):

    best_gini = 1
    best_axis = None
    best_threshold = None

    for axis in [0, 1]:
        threshold = -1

        while threshold <= 2.5:

            score = gini_impurity(x, y, axis, threshold)

            if score < best_gini:
                best_gini = score
                best_axis = axis
                best_threshold = threshold

            threshold = threshold + 0.01

    return best_axis, best_threshold, best_gini

left_mask = x_data[:, best_axis] <= best_threshold
right_mask = x_data[:, best_axis] > best_threshold

x_left = x_data[left_mask]
y_left = y_data[left_mask]
x_right = x_data[right_mask]
y_right = y_data[right_mask]

left_axis, left_threshold, left_gini = find_best_split(x_left, y_left)
right_axis, right_threshold, right_gini = find_best_split(x_right, y_right)

print("left:", left_axis, left_threshold, left_gini)
print("right:", right_axis, right_threshold, right_gini)
