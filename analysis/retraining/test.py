
import scipy.stats as stats
import numpy as np
# Performing the t tests on the learning curves

# Loading data

dummy_data = np.random.normal(0,1,(100,10))

print(dummy_data.shape)

# Plotting learning curves

# T test
tstat, pvalue = stats.ttest_rel(dummy_data[:,0],dummy_data[:,1])

print(pvalue)
