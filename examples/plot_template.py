"""
===========================
Plotting Template Estimator
===========================

An example plot of :class:`scikit_contrib.template.TemplateEstimator`
"""
import numpy as np
from matplotlib import pyplot as plt
from scikit_contrib import TemplateEstimator

X = np.arange(100).reshape(100, 1)
y = np.zeros((100, ))
estimator = TemplateEstimator()
estimator.fit(X, y)
plt.plot(estimator.predict(X))
plt.show()
