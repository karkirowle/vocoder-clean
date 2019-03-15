
import scipy.stats as stats
import numpy as np
import glob
import seaborn as sns
import matplotlib.pyplot as plt
#sns.set()
# Performing the t tests on the learning curves
def decay(signal,tau):
    out = np.zeros_like(signal)
    for i in range(len(signal)):
        if i == 1:
            out[i] = signal[i]
        else:
            out[i] = out[i-1]*tau + (1-tau)*signal[i]
    return out
# Loading data

#idx = [1.0,0.8,0.6,0.4,0.2]
idx = [1.0,0.8,0.6,0.4,0.2]

curves = len(idx)
seeds = 1
data2 = np.zeros((100,curves, seeds))

count = 0
for i,id in enumerate(idx):
    for j in range(seeds):
            data2[:,i,j] = decay(np.load("../../learning_curve_adam3/" +
                            str(id) +
                                         "seed25test0k.npy"),0)
            print(data2)



            
#print(data2)

#data2[:


data= np.mean(data2,axis=2)
print(data.shape)


# T test
for i in range(curves - 1):
    tstat, pvalue = stats.ttest_rel(data[:,i],data[:,i+1])
    print ("Paired t-test p-level for learning curve",str(i),
           str(i+1),"is",str(pvalue))

plt.plot(data)
plt.xlabel("Epochs")
plt.ylabel("Validation loss")
plt.legend(["100% of traiing set",
            "80% of training set", "60% of training set", "40% of training set",
            "20% of training set"])
plt.title("Adding more data helps generalisation performance")
#plt.show()
plt.savefig("../../paper/retraining_lc.pgf")




#sns.set_style("white")

#plt.plot(data[:,0], color="black", marker="+")
#plt.plot(data[:,5], color="black", marker=".")
#plt.plot(data[:,9], color="black", marker="^")

#plt.xlim([0,40])
#plt.xlabel("Epochs")
#plt.ylabel("Validation loss")
#plt.legend(["Full training set", "50% of training set", "10% of training set"])
#plt.title("Learning curves show that adding data help generalisation")
#plt.figure(num=1, figsize=(4.2,4.2), dpi=80, facecolor="w", edgecolor="k")
#plt.show()

# We take the mean of the last five epochs
#from scipy import stats


#datapoints = np.array(idx) * 2274
#print(datapoints)
#epoch_means = np.mean(data[25:26,:],axis=0)
#epoch_means = data[22,:]
#print(epoch_means)
#slope, intercept, r_value, p_value, std_error = stats.linregress(datapoints,epoch_means)

#x = np.linspace(0,2800,6000)
#y = slope * x + intercept

#print(-intercept/slope)
#plt.plot(x,y,color="black")
#plt.scatter(datapoints, epoch_means, color="black")
#plt.xlabel("Total number of training data points")
#plt.ylabel("Validation loss")
#plt.title("Adding data helps generalisation")
#plt.show()
#plt.savefig("../../paper/retraining_linear.pgf")
