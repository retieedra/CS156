# Forward pass
w = np.array([2.00, -3.00])
b = -3.00

x = np.array([-1.00, -2.00])

dot = w.T @ x  # 4.00
summed = dot + b  # 1.00

neg = -summed  # -1.00
exp = np.exp(neg)  # 0.37
plus1 = exp + 1  # 1.37
invert = 1/plus1  # 0.73


# Backward pass
d_invert = 1.00
d_plus1 = d_invert * (-1 / plus1**2)
d_exp = d_plus1 * 1
d_neg = d_exp * exp
summed = d_neg * (-1)

d_b, d_dot = summed, summed
# Be careful, the next two are vectors!
d_w = d_dot * x
d_x = d_dot * w
