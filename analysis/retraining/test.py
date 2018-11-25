
import scipy.stats as stats
import numpy as np
import glob
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
# Performing the t tests on the learning curves

# Loading data


files = glob.glob('*_test')

print(files)

curves = len(files)

data = np.zeros((100,curves))

for i,file in enumerate(files):
    print(file)
    data[:,i] = np.loadtxt(file)






# T test
for i in range(curves - 1):
    tstat, pvalue = stats.ttest_rel(data[:,i],data[:,i+1])
    print ("Paired t-test p-level for learning curve",str(i),
           str(i+1),"is",str(pvalue))

print(data.T)

plt.plot(data)
plt.xlim([0,40])
plt.xlabel("Epochs")
plt.ylabel("RMSE on validation set")
plt.title("Adding data significantly helps generalisation")
plt.show()
