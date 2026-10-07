#code for measuring regression badness.

import numpy as np

predicted_lengths = linear_model.predict(X_test)
residuals = y_test - predicted_lengths

rmse = np.sqrt(np.mean(residuals ** 2))
print("test rmse:", rmse, "cm")
