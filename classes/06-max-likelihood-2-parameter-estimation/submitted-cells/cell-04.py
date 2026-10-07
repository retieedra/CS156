x = np.array(x_data)
y = np.array(y_data, dtype=float)

def log_likelihood(beta):
    p = sigmoid(beta[0] + beta[1] * x)
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

beta = np.array([3.0, -3.0])  # aishwarya's guess
p = sigmoid(beta[0] + beta[1] * x)

# gradient for beta_0 and beta_1
grad = np.array([np.sum(y - p), np.sum(x * (y - p))])

new_beta = beta + 0.0001 * grad  # small step uphill

print("gradient:", grad)
print("new beta:", new_beta)
print("before:", log_likelihood(beta))
print("after:", log_likelihood(new_beta))
