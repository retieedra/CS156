import jax
import jax.numpy as jnp # JAX has a version of NumPy that acts...just like NumPy!
# We first define a function for *just* the forward pass
def sigmoid_example(params, x):
    dot = params['w'].T @ x # 4.00
    summed = dot + params['b'] # 1.00
    neg = -summed # -1.00
    exp = jnp.exp(neg) # 0.37
    plus1 = exp + 1 # 1.37
    invert = 1/plus1 # 0.73
    return invert
