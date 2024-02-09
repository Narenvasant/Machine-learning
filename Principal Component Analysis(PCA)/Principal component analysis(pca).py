#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.pyplot import imread


# In[2]:



def pca(data, k):
    '''
    Computate the pca.
    @param data: numpy.array of data vectors (NxM)
    @param k: number of eigenvectors to return

    returns (eigenvectors (kxM), eigenvalues (k))
    '''

    #TODO: Your implementation goes here
    data = data - data.mean(axis=0) # Center data points
    print("Centered Matrix: ", data)
    
    cov = np.cov(data.T) / data.shape[0] # Get covariance matrix
    print("Covariance matrix: ", cov)
    
    v, w = np.linalg.eig(cov)
    
    idx = v.argsort()[::-1] # Sort descending and get sorted indices
    v = v[idx] # Use indices on eigv vector
    w = w[:,idx] # 

    print("Eigenvalue vector: ", v)
    print("Eigenvector: ", w)
    
    return data.dot(w[:, :k])
    #return w[:, :k]

def showVec(img, shape):
    '''
    Reshape vector to given image size and plots it
    len(img) must be shape[0]*shape[1]
    @param img: given image as 1d vector
    @param shape: shape of image
    '''
    img = np.reshape(img, shape)
    plt.imshow(img, cmap="gray")
    plt.show()

def normalized_linear_combination(vecs):
    '''
    Compute a linear combination of given vectors and weights.
    len(weights) must be <= vecs.shape[0]
    @param vecs: numpy.array of numpy.arrays to combine (NxM, N = number of basis vectors (unitary or eigenvectors))
    @param weights: list of weights (S) for the first S vectors
    returns numpy.array [M,1]
    '''
    #std = []
    #std = numpy.std(vecs)
    #weights = vecs./std
    #TODO: Your implementation goes here
    return std
   

def load_dataset_from_folder(dir):
    '''
    Load all pgm files from given folder
    Dataset is taken from https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
    @param dir: folder to look for pgm files
    returns tuple of data (N,M) and shape of one img, N = number of samples, M = size of sample
    '''
    datalist = []
    datashape = []
    for path, _, files in os.walk(dir):
        files = glob.glob(path+"/*.pgm")
        for file_ in files:
            img = imread(file_)

            #scale down for faster computation
            img = np.array(Image.fromarray(img).resize((50,50)))
            datashape = img.shape

            d = np.ravel(img)
            datalist.append(d)

    data = np.array(datalist)
    return data, datashape

'''
1) Load dataset
2) compute principal components
3) compute linear combinations
4) display stuff
'''

#TODO: Your implementation goes here
 


# In[5]:


data, datashape = load_dataset_from_folder(r"C:\Users\naren\Desktop\CMM\Machine Learning for Autonomous robots\Exercise\Ex1\orl_faces")


# In[6]:


df = pca(data,2)
df


# In[41]:


std = np.std(df)
d = np.array(df/std)
#print(d)
d1 = d[:, 0]
#print(d1)
d2 = d[:, 1]
#print(d2)
#plt.scatter(d1,d2,c="r")
plt.plot(d1,c="b",label="d1")
plt.plot(d2,c="r",label="d2")
plt.show()
#plt.show()
#plt.plot(d)
# plt.scatter (d[:, 0],d[:, 1])


# In[27]:


df.shape
x.shape
y.shape


# In[43]:


a = df[0,:]
print(a)
c = df[10,:]
print(c)
b = df[400,:]
print(b)
#plt.plot(a,c)
plt.scatter(a,c)


# In[40]:


a_var = np.var(b1)
print(x_var)
b_var = np.var(b2)
print(y_var)
#plt.scatter(a_var, b_var)


# In[ ]:




