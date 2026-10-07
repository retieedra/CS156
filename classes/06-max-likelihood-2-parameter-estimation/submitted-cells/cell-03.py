# Fix this function
# You might wish to use the sigmoid(x) function defined above!
def log_likelihood_per_point(beta_0, beta_1, x, y):
    prob = sigmoid(beta_0 + beta_1*x)  # this should be only one line!
    if y is True:
        return np.log(prob)  # Think simple!
    else:
        return np.log(1-prob)  # Again, think simple!


# This one's complete!
def total_log_likelihood(beta_0, beta_1, xs, ys):
    total_ll = 0

    for (x, y) in zip(xs, ys):
        total_ll += log_likelihood_per_point(beta_0, beta_1, x, y)

    return total_ll

# Be careful, which is the *higher* value when two values are negative?
print(total_log_likelihood(3, -3, x_data, y_data))
print(total_log_likelihood(4, -1, x_data, y_data))

#Aishwarya is less wrong
