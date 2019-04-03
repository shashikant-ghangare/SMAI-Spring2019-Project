#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
ApparelData = pd.read_csv('apparel-trainval.csv',sep=',',index_col = False)


# # DIMENSIONALITY REDUCTION

# In[3]:


Xmatrix = ApparelData.values
LabelVector = Xmatrix[:,0]
Xmatrix = Xmatrix[:,1:]
meanVector = np.mean(Xmatrix,axis = 0)
CenteredXmatrix = Xmatrix - meanVector
SdVector = np.std(CenteredXmatrix.astype(float),axis = 0)
CenteredXmatrix /= SdVector
covOfCenteredXmatrix = np.array([(CenteredXmatrix[0,:].astype(float))]).T@ np.array([(CenteredXmatrix[0,:].astype(float))])
for i in range(1,len(CenteredXmatrix)):
    if(i%10==0):
        print(i)
    covOfCenteredXmatrix +=   np.array([(CenteredXmatrix[i,:].astype(float))]).T@ np.array([(CenteredXmatrix[i,:].astype(float))])
covOfCenteredXmatrix /= len(CenteredXmatrix)
eigenValues, eigenVectors = np.linalg.eig(covOfCenteredXmatrix)
idx = eigenValues.argsort()[::-1]   
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]
neededEigenVectors = []
for idx, val in enumerate(eigenValues):
    if (sum(eigenValues[0:idx])/sum(eigenValues))>0.9:
        neededEigenVectors = eigenVectors[:,0:idx]
        break
reconstructedXmatrix = CenteredXmatrix.astype(float)@neededEigenVectors


# In[4]:


reconstructedXmatrix.shape


# In[5]:


reconstructedXmatrix = np.insert(reconstructedXmatrix.astype(str), 0, values=LabelVector, axis=1) 


# In[6]:


reconstructedXmatrix.shape


# In[7]:


newReconstructedData = pd.DataFrame(data=reconstructedXmatrix, columns=list(ApparelData.columns)[:138]) 


# In[10]:


type(newReconstructedData)


# In[22]:


#newReconstructedData.to_csv('ReconstructedApparelData.csv',sep=',',index = False)


# In[12]:


newReconstructedData = pd.read_csv('ReconstructedApparelData.csv',sep=',',index_col = False)


# In[13]:


newReconstructedData.shape


# # SCIKIT LEARN BINARY SVM IMPLEMENTATION

# In[14]:


UniqueLabels=np.array([5,6])
DataPerClass = []
index = 0
for i in UniqueLabels:
    tempdf = newReconstructedData.loc[newReconstructedData['label'] == i]
    DataPerClass.append(tempdf.sample(frac = 1))
    print(len(DataPerClass[index]))#+str(" for ")+str(i))
    index = index + 1


# In[15]:


train5 = DataPerClass[0].sample(frac=0.8)
train6 = DataPerClass[1].sample(frac=0.8)
val5 = DataPerClass[0].loc[~DataPerClass[0].index.isin(train5.index)]
val6 = DataPerClass[1].loc[~DataPerClass[1].index.isin(train6.index)]
allTrain = [train5,train6]
allVal = [val5,val6]
train = pd.concat(allTrain, ignore_index= True)
val = pd.concat(allVal, ignore_index= True)
train = train.sample(frac=1)
val = val.sample(frac=1)
print(len(train))
print(len(val))


# In[71]:


#train.to_csv('train56.csv',sep=',',index = False)
#val.to_csv('val56.csv',sep=',',index = False)


# In[72]:


train = pd.read_csv('train56.csv',sep=',',index_col = False)
val = pd.read_csv('val56.csv',sep=',',index_col = False)


# In[16]:


from sklearn.svm import SVC
clf = SVC(C = 10, kernel = 'linear')
clf.fit(train.values[:,1:], train.values[:,0]) 

print('w = ',clf.coef_)
print('b = ',clf.intercept_)
print('Indices of support vectors = ', clf.support_)
print('Support vectors = ', clf.support_vectors_)
print('Number of support vectors for each class = ', clf.n_support_)
print('Coefficients of the support vector in the decision function = ', np.abs(clf.dual_coef_))


# In[17]:


PredSVM = []
for i in range(0,len(val)):
    PredSVM.append(clf.predict([val.values[i,1:]]))


# In[18]:


PredSVM1 = np.array(PredSVM).ravel()


# In[20]:


conMatrix = pd.crosstab(PredSVM1,val.values[:,0])


# In[21]:


conMatrix


# # CVXOPT BINARY SVM IMPLEMENTATION

# In[24]:


from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers


# In[29]:


train.values[:,0]


# In[35]:


#Initializing values and computing H. Note the 1. to force to float type
C = 10
m,n = train.values[:,1:].shape
y = train.values[:,0].reshape(-1,1) * 1.
print(y.shape)
yValues = []
for p in train.values[:,0]:
    if p==5:
        yValues.append(1.0)
    else:
        yValues.append(-1.0)

X = train.values[:,1:]
y = np.array(yValues).reshape(9600,1)
X_dash = y * X

H = np.dot(X_dash , X_dash.T) * 1.

#Converting into cvxopt format - as previously
P = cvxopt_matrix(H)
q = cvxopt_matrix(-np.ones((m, 1)))
G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
A = cvxopt_matrix(y.reshape(1, -1))
b = cvxopt_matrix(np.zeros(1))

#Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])

#==================Computing and printing parameters===============================#
w = ((y * alphas).T @ X).reshape(-1,1)
S = (alphas > 1e-4).flatten()
b = y[S] - np.dot(X[S], w)

#Display results
print('Alphas = ',alphas[alphas > 1e-4])
print('w = ', w.flatten())
print('b = ', b[0])


# In[53]:


parametersSVM = {}
parametersSVM['alpha'] = alphas[alphas > 1e-4]
parametersSVM['S'] = S
parametersSVM['w'] =  w.flatten()
parametersSVM['b'] = b[0]


# import pickle
# pickle_out = open("parametersSVM.pickle","wb")
# pickle.dump(parametersSVM, pickle_out)
# pickle_out.close()

# In[55]:


pickle_in = open("parametersSVM.pickle","rb")
parametersSVM1 = pickle.load(pickle_in)


# In[56]:


parametersSVM1


# In[57]:


parametersSVM1['w'].shape


# In[58]:


val.values[i,1:].shape


# In[59]:


parametersSVM1['b'].shape


# In[60]:


(parametersSVM1['w'].T@val.values[i,1:])+parametersSVM1['b']


# In[61]:


PredSVM2 = []
for i in range(0,len(val)):
    temp1 = (parametersSVM1['w'].T@val.values[i,1:])+parametersSVM1['b']
    if temp1 > 0:
        PredSVM2.append(5)
    else:
        PredSVM2.append(6)


# In[66]:


len(np.array(PredSVM2))


# In[69]:


len(val.values[:,0])


# In[70]:


conMatrix = pd.crosstab(np.array(PredSVM2),val.values[:,0])
conMatrix


# In[ ]:




