import numpy as np

# Forward pass
w0, w1, w2 = 2.00, -3.00, -3.00
x0, x1 = -1.00, -2.00

mul0 = w0 * x0  # -2.00
mul1 =  w1 * x1  # 6.00
sum01 = mul0 + mul1  # 4.00
sum012 = sum01 + w2  # 1.00

#just calculating it through a sigmoid here
neg = -sum012  # -1.00
exp = np.exp(neg)  # 0.37
plus1 = exp + 1  # 1.37
invert = 1/plus1  # 0.73


# Backward pass
# Remember *all* derivatives are with respect to the final function invert
d_invert = 1.00  # this is d_invertd_invert, but we shorten to d_invert
# Here's one completed for you :)
d_plus1 = d_invert * (-1/plus1**2)  # this is d_invertd_plus_1, but we shorten to d_plus_1
d_exp = d_plus1 * 1
d_neg = d_exp * exp
d_sum012 = d_neg * (-1)

# Notice for gates with multple inputs, when going backward
# they are responsible for derivatives of multiple values!
d_w2, d_sum01 = d_sum012, d_sum012
d_mul0, d_mul1 = d_sum01, d_sum01

d_w1, d_x1 = d_mul1 * x1, d_mul1 * w1
d_w0, d_x0 = d_mul0 * x0, d_mul0 * w0
