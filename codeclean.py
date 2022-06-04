# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 06:34:09 2022

@author: Magnus Gustav Holm
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.tree import DecisionTreeRegressor as DTR

class Node:
    def __init__(self):
        self.split = 0
        self.pred = 0
        self.sample = None
        self.samplesize=0
        self.left = None
        self.right = None
        self.intervals = None
        self.splitpoint = 1/2
    

class decisiontree1: #mse, regression tree
    def __init__(self, X, y, kn=32, D_n=0, M_n=0):
        self.M_n = M_n
        self.height = int(np.ceil(np.log2(kn)))
        self.root = Node()
        self.root.sample = np.array([True for i in range(y.shape[0])])
        self.root.pred = (1/y.shape[0])*np.sum(y)
        self.root.intervals = np.zeros((2, X.shape[1]))
        self.root.intervals[1] = np.ones((X.shape[1]))
        self.nodesplits = list()
        self.no_points = list()
        self.mse = [np.sum((y- self.root.pred)**2)]
        self.fit(X, y)
        
        
            
    def stump(self, node, X_S, y_S):
        L = Node()
        R = Node()
        mse = np.sum((y_S- node.pred)**2)
        mse_0 = np.copy(mse)
        INT = np.copy(node.intervals)
        splitpoints = (INT[0]+INT[1])/2
        b_L = (X_S <= splitpoints)
        b_R = (X_S > splitpoints)
        n_L = np.sum(b_L, axis=0)
        n_R = np.sum(b_R, axis=0)
        self.no_points.append(y_S.shape[0])
        M = np.zeros(n_L.shape)
        y_col = y_S.reshape((y_S.shape[0],1))
        pred_L = np.nan_to_num((1/n_L)*np.sum(y_col*b_L, axis=0))
        pred_R = np.nan_to_num((1/n_R)*np.sum(y_col*b_R, axis=0))
        Ones = np.ones((y_S.shape[0],pred_R.shape[0]))
        ss_L = y_col*Ones - pred_L
        ss_R = y_col*Ones - pred_R
        M = (np.sum(b_L*ss_L**2, axis=0) + np.sum(b_R*ss_R**2, axis=0)) # a vector of the mses for each split coordinate in the given subsequence of L (which is 1,..., d if D_n=0)
        M_0 = np.argsort(M)
        J = np.random.choice(M_0[0:self.M_n])
        L.sample = np.copy(b_L[:,J])
        R.sample = np.copy(b_R[:,J])
        L.pred = pred_L[J]*(n_L[J]!=0) + node.pred*(n_L[J]==0)
        R.pred = pred_R[J]*(n_R[J]!=0) + node.pred*(n_R[J]==0)
        new_mse = self.mse[-1] - mse_0 + mse
        self.mse.append(new_mse)
        
        L.intervals=INT
        R.intervals=np.copy(INT)
        L.intervals[1,J] = np.copy(splitpoints[J])
        R.intervals[0,J] = np.copy(splitpoints[J])
        node.left = L
        node.right = R
        node.split = J
        node.splitpoint = splitpoints[J]
        self.nodesplits.append(J)
        
    def grow_once(self, node, X, y):
        if node.left == None:
            self.stump(node, X, y)
        else:
            X_1 = np.copy(X[node.left.sample])
            y_1 = np.copy(y[node.left.sample])
            X_2 = np.copy(X[node.right.sample])
            y_2 = np.copy(y[node.right.sample])
            self.grow_once(node.left, X_1, y_1)
            self.grow_once(node.right, X_2, y_2)
        
        
    def drop_once(self, node, x):
        if node.left == None:
            return node.pred
        else:
            if x[node.split] <= node.splitpoint:
                return self.drop_once(node.left, x)
            else:
                return self.drop_once(node.right, x)
                
    def fit(self, X, y):
        for k in range(self.height):
            self.grow_once(self.root, X, y)
            
    def predict(self, x):
        return self.drop_once(self.root, x)
    
    
class randomforest1:
    def __init__(self, X, y, X_pred, y_pred, N=0, kn=128, m=500, Mn = 1):
        self.m = m
        self.kn = kn
        self.Mn = Mn
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.trees = list()
        self.trees_1 = list()
        if (N==0):
            self.N = self.n//2
        else:
            self.N = N
        L = list(range(self.n))*(int(self.m*self.N/self.n)+1)
        np.random.shuffle(L)
        self.L = L
        self.l=0
        self.pn_1 = np.zeros((self.m, self.d))
        self.height = int(np.ceil(np.log2(self.kn)))
        #print('L: ', L)
        for k in range(self.m):
            samples = L[self.l:self.l+self.N]
            self.l= self.l+ self.N
            D_1 = decisiontree1(X[samples], y[samples], kn= self.kn, M_n = self.Mn)
            self.trees_1.append(D_1)
            #print('nodepslits: ', D_1.nodesplits)
            for j in range(len(D_1.nodesplits)):
                #n_points = n_points + 1*(D_1.no_points[j]>0)
                self.pn_1[k,D_1.nodesplits[j]] = self.pn_1[k,D_1.nodesplits[j]] +(1/D_1.height)*(2**(-np.ceil(np.log2(j+2))+1))#*(D_1.no_points[j]>0)
            samples = np.random.randint(0, self.n, self.N)
            #print('probs: ', self.pn_1, 'k: ', k)
            D = decisiontree2(X_pred[samples], y_pred[samples], kn=self.kn, pn= self.pn_1[k])
            self.trees.append(D)
            
    
    def predict(self, x):
        res = 0
        for k in range(self.m):
            res = res + self.trees[k].predict(x)
        return (1/self.m)*res
    def split_Feature_ratios(self):
        res = np.zeros((self.d))
        for tree in self.trees:
            L = len(tree.nodesplits)
            for k in tree.nodesplits:
                res[k] = res[k]+1
        return (1/(L*self.m))*res


    
class decisiontree2: #mse, regression tree
    def __init__(self, X, y, pn, kn=32):
        self.height = int(np.ceil(np.log2(kn)))
        if self.height==0:
            self.height=1
        self.root = Node()
        self.root.intervals = np.zeros((2, X.shape[1]))
        self.root.intervals[1] = np.ones((X.shape[1]))
        self.nodesplits = list()
        self.d = X.shape[1]
        self.pn = pn
        self.fit(X, y)
        
        
            
    def stump(self, node):
        L = Node()
        R = Node()
        INT = np.copy(node.intervals)
        splitpoints = (INT[0]+INT[1])/2
        if np.isclose(np.sum(self.pn),1):
            self.pn[0]= self.pn[0]+(1-np.sum(self.pn))
        self.pn = self.pn*(self.pn >= 0)
        J = int(np.random.choice(range(self.d),1, p=self.pn))
        L.intervals=INT
        R.intervals=np.copy(INT)
        L.intervals[1,J] = np.copy(splitpoints[J])
        R.intervals[0,J] = np.copy(splitpoints[J])
        node.left = L
        node.right = R
        node.split = J
        node.splitpoint = splitpoints[J]
        self.nodesplits.append(J)
        
    def grow_once(self, node): #with this recursion, the order of nodes in nodesplits becomes the same as the order of k
        if node.left == None:
            self.stump(node)
        else:
            self.grow_once(node.left)
            self.grow_once(node.right)
        
        
    def drop_once(self, node, x):
        if node.left == None:
            if node.samplesize==0:
                return 0
            else:
                return node.pred/node.samplesize
        else:
            if x[node.split] <= node.splitpoint:
                return self.drop_once(node.left, x)
            else:
                return self.drop_once(node.right, x)
            
    def drop_once_fit(self, node, x, y):
        if node.left == None:
            node.pred = node.pred+y
            node.samplesize = node.samplesize+1
        else:
            if x[node.split] <= node.splitpoint:
                return self.drop_once_fit(node.left, x,y)
            else:
                return self.drop_once_fit(node.right, x,y)
                
    def fit(self, X, y):
        for k in range(self.height):
            self.grow_once(self.root)
        for k in range(X.shape[0]):
            self.drop_once_fit(self.root, X[k], y[k])
        
    def predict(self, x):
        return self.drop_once(self.root, x)
    
class randomforest2:
    def __init__(self, X, y, X_pred, y_pred, N=0, kn=128, m=500, Mn = 20):
        self.m = m
        self.kn = kn
        self.Mn = Mn
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.trees = list()
        self.trees_1 = list()
        if (N==0):
            self.N = self.n//2
        else:
            self.N = N
        L = list(range(self.n))*(int(self.m*self.N/self.n)+1)
        np.random.shuffle(L)
        self.L = L
        self.l=0
        self.pn_1 = np.zeros((self.m, self.d))
        self.height = int(np.ceil(np.log2(self.kn)))
        if self.height<1:
            self.height=1
        for k in range(self.m):
            samples = L[self.l:self.l+self.N]
            self.l= self.l+ self.N
            D_1 = DTR(criterion='mse', max_depth=self.height, max_features=self.Mn, min_samples_leaf=2)
            D_1.fit(X[samples],y[samples])
            M = list()
            def as_tree(tree, i, p, depth):
                if tree.feature[i] == -2 and depth <= self.height:
                    j=np.random.choice(range(self.d))
                    M.append((j,p))
                    as_tree(tree, tree.children_left[i], 0.5*p, depth+1)
                    as_tree(tree, tree.children_right[i], 0.5*p, depth+1)
                if tree.feature[i] != -2:
                    M.append((tree.feature[i], p))
                    P = tree.threshold[i]
                    as_tree(tree, tree.children_left[i], P*p, depth+1)
                    as_tree(tree, tree.children_right[i], (1- P)*p, depth+1)
            as_tree(D_1.tree_, 0, 1, 1)
            for j in range(len(M)):
                self.pn_1[k, M[j][0]] = self.pn_1[k, M[j][0]] + (1/D_1.get_params()['max_depth'])*M[j][1]*(len(M)>1) + M[j][1]*(len(M)==1)
                #print('j: ', j, 'pn_1: ', self.pn_1[k])
            #print('probs: ', self.pn_1[k], 'k: ', k)
            D = decisiontree2(X_pred, y_pred, pn= self.pn_1[k], kn=self.kn)
            self.trees.append(D)
            
    
    def predict(self, x):
        res = 0
        for k in range(self.m):
            res = res + self.trees[k].predict(x)
        return (1/self.m)*res
    def split_Feature_ratios(self):
        res = np.zeros((self.d))
        for tree in self.trees:
            L = len(tree.nodesplits)
            for k in tree.nodesplits:
                res[k] = res[k]+1
        return (1/(L*self.m))*res

class randomforest3:
    def __init__(self, X_pred, y_pred, pn, N=0, kn=128, m=500, Mn = 20):
        self.m = m
        self.kn = kn
        self.Mn = Mn
        self.pn = pn
        self.n = X_pred.shape[0]
        self.d = X_pred.shape[1]
        self.trees = list()
        self.trees_1 = list()
        if (N==0):
            self.N = self.n//2
        else:
            self.N = N
        L = list(range(self.n))*(int(self.m*self.N/self.n)+1)
        np.random.shuffle(L)
        self.L = L
        self.l=0
        self.height = int(np.ceil(np.log2(self.kn)))
        #print('L: ', L)
        
        for k in range(self.m):
            print('rf k: ', k)
            samples = L[self.l:self.l+self.N]
            self.l= self.l+ self.N
            D = decisiontree2(X_pred[samples], y_pred[samples], kn=self.kn, pn= self.pn)
            self.trees.append(D)
            
    
    def predict(self, x):
        res = 0
        for k in range(self.m):
            res = res + self.trees[k].predict(x)
        return (1/self.m)*res
    def split_Feature_ratios(self):
        res = np.zeros((self.d))
        for tree in self.trees:
            L = len(tree.nodesplits)
            for k in tree.nodesplits:
                res[k] = res[k]+1
        return (1/(L*self.m))*res



r_sine = (lambda X: 10*np.sin(2*np.pi*X[:,0]))
r_1 = (lambda X: 10*np.sin(2*np.pi*X[:,0])+X[:,3])
r_2 = (lambda X: 1*X[:,1]- 4*X[:,2]+0.4*X[:,3])
r_3 = (lambda X: X[:,1]**2- 4*(X[:,2]**2)+0.4*(X[:,3]**2))
r_4 = (lambda X: 3*(X[:,3]<=0.3)*(4*(X[:,1]<=0.4) + 5*(X[:,1]>0.4)) + 6*(X[:,3]>0.3)*(7*(X[:,0]<=0.8) + 8*(X[:,0]>0.8)))
r_5 = (lambda X: X[:,0]*(X[:,0]<=0.5)*(X[:,1]<=0.5) - X[:,1]*(X[:,0]>=0.5)*(X[:,1]>=0.5))
r_6 = (lambda X: 10*np.sin(4*np.pi*X[:,0])+X[:,3])
r_7 = (lambda X: 10*np.sin(6*np.pi*X[:,0])+X[:,3])
r_8 = (lambda X: 10*np.sin(8*np.pi*X[:,0])+X[:,3])
r_9 = (lambda X: 10*np.sin(10*np.pi*X[:,0])+X[:,3])
r_10= (lambda X: 10*np.sin(2*np.pi*X[:,0]*X[:,1]))

targets = { 'r_sine': r_sine, 'r_1': r_1, 'r_2': r_2, 'r_3': r_3, 'r_4': r_4, 'r_5': r_5, 'r_6': r_6,'r_7': r_7,'r_8': r_8,'r_9': r_9, 'r_10': r_10 }
LHS_params = {'r_1': ([0,3],2,np.sqrt(400*np.pi**2 +1),11), 'r_2': ([1,2,3],3,(1**2+4**2+0.4**2)**0.5,1.4), 'r_3': ([1,2,3],3,2*(17+0.4**2)**0.5,1.4), 'r_4': ([0,1,3],3,50,6*8), 'r_5': ([0,1],2,1,1), 'r_6': ([0,3],2,np.sqrt(1600*np.pi**2 +1),10), 'r_7': ([0,3],2,np.sqrt(3600*np.pi**2 +1),10), 'r_8': ([0,3],2,np.sqrt(6400*np.pi**2 +1),10), 'r_9': ([0,3],2,np.sqrt(10000*np.pi**2 +1),10), 'r_10': ([0,1],2,np.sqrt(800*np.pi**2),10)}

def test_consistency_1(r, n, d, m_1, Mn_1, kn_1):
    X = np.random.uniform(size=(m_1*n,d))
    eps = np.random.normal(size=m_1*n)
    y = targets[r](X)+eps
    p = (1/d)*np.ones((d))
    RF = randomforest3(X, y, N=n, kn=kn_1, m=m_1, Mn=d, pn=p)
    X_1 = np.random.uniform(size=(200,d))
    y_1 = targets[r](X_1)
    Z= np.zeros((X_1.shape[0]))
    for k in range(X_1.shape[0]):
        Z[k]= RF.predict(X_1[k])
    E = (1/X_1.shape[0])*np.sum((Z-y_1)**2)
    return X_1[:,0], y_1, Z, E
    
def print_consistency_1(r):
    res = []
    N = [20,50,200,500,1000,3000, 10000, 30000, 50000]
    fig, axs = plt.subplots(3,3, figsize=(10,9))
    for i in range(3):
        for j in range(3):
            X, y, Z, E=test_consistency_1(r, N[i*3+j], 5, 200, 0, N[i*3+j]**0.9)
            res.append(E)
            axs[i,j].text(0.1,1.1*(i<1) -0.2*(i>=1), 'n= %s' %str(N[i*3+j]) + ',   L2 error = %s' %str(E)[0:5], transform=axs[i,j].transAxes)
            axs[i,j].scatter(X, y, s=5)
            axs[i,j].scatter(X, Z, s=5)
    fig.savefig('consistencyd=5kn=power09%s.jpg' %str(r))
    print(res)
    
def print_consistency_2(r):
    res = []
    N = [20,50,200,500,1000,3000, 10000, 20000, 50000]
    fig = plt.figure()
    for i in range(len(N)):
        X, y, Z, E=test_consistency_1(r, N[i], 5, 200, 0, N[i]**(0.9))
        res.append(E)
    ax=fig.add_subplot(1,1,1)
    ax.plot(N, res)
    fig.savefig('consistencyd=5kn=power09NEplot%s.jpg' %str(r))
    print(res)
    
def validate_sparsity_2(r, n, d, R_val):
    X = np.random.uniform(size=(n,d))
    eps = np.random.normal(size=(n))
    y = targets[r](X) + eps
    K = np.random.choice(list(range(1,n)), R_val)
    error = np.zeros((R_val))
    for k in range(R_val):
        RF = randomforest2(X[0:n//2], y[0:n//2], X[n//2:n], y[n//2:n], N=n//5, kn=K[k], m=10 , Mn=d)
        Z= np.zeros(y.shape)
        for l in range(X.shape[0]):
            Z[l]=RF.predict(X[l])
        error[k] = np.sum((y-Z)**2)
    k_val = np.argmin(error)
    return K[int(k_val)]
    
def test_sparsity_2(r, n, d, kn_1, M_n, m_1, R_2):
    X = np.random.uniform(size=(n,d))
    eps = np.random.normal(size=(n))
    y = targets[r](X) + eps
    RF_Biau = randomforest2(X[0:n//2], y[0:n//2], X[n//2:n], y[n//2:n], N=n//5, kn=kn_1, m=m_1 , Mn=M_n)
    RF_Breiman = RFR(n_estimators=m_1, criterion='mse', max_features=d//2)
    RF_Breiman.fit(X, y)
    X_1 = np.random.uniform(size=(R_2,d))
    y_1 = targets[r](X_1)
    Z_1= np.zeros(X_1.shape[0])
    for k in range(X_1.shape[0]):
        Z_1[k]=RF_Biau.predict(X_1[k])
    Z_2 = RF_Breiman.predict(X_1)
    MSE_1 = (1/X_1.shape[0])*np.sum((y_1-Z_1)**2)
    MSE_2 = (1/X_1.shape[0])*np.sum((y_1-Z_2)**2)
    return MSE_1, MSE_2

def simulation_7_1(R_1, R_2): # R_1=200, R_2=10
    N = [10,20,50,100,150,200,300,500, 1000]
    D= [10, 50, 200]
    fig, axs = plt.subplots(2, 3, figsize=(11,7))
    for k in range(len(D)):
        d = D[k]
        for r in ['r_2', 'r_3', 'r_5']:
            res_MSE = np.zeros((len(N)))
            res_rate = np.zeros((len(N)))
            S = LHS_params[r][1]
            L = LHS_params[r][2]
            C = (72/np.pi)*(np.pi/16)**(S/(2*d))
            sup_r = LHS_params[r][3]
            E = C*((S**2)/(S-1))**(S/(2*d)) + 2*np.exp(-1)*sup_r**2
            for i in range(len(N)):
                print('r: ', r, 'n: ', N[i])
                kn_1 = (L**2/E)**(S*np.log(2)/(0.75+S*np.log(2)))*(N[i]**(S*np.log(2)/(0.75+S*np.log(2))))
                print('kn: ', kn_1)
                rate = (E*L**(2*S*np.log(2)/0.75))**(0.75/(0.75+S*np.log(2)))*(N[i]**(-S*np.log(2)/(0.75+S*np.log(2))))   
                X_1 = np.random.uniform(size=(R_2,d))
                y_1 = targets[r](X_1)
                Z= np.zeros(R_2)
                for j in range(R_2):
                    X = np.random.uniform(size=(N[i],d))
                    eps = np.random.normal(size=N[i])
                    y = targets[r](X) + eps
                    RF = randomforest1(X[0:N[i]//2], y[0:N[i]//2], X[N[i]//2:N[i]], y[N[i]//2:N[i]], kn=kn_1, m=R_1 , Mn=S)
                    Z[j]=RF.predict(X_1[j])
                MSE = (1/R_2)*np.sum((y_1-Z)**2)
                res_MSE[i]= MSE
                res_rate[i]=rate
            print(res_MSE)
            print(res_rate)
            axs[0,k].set_xlim([0,1000])
            axs[0,k].set_xlabel('n')
            axs[0,k].set_ylabel('error')
            axs[1,k].set_xlim([0,1000])
            axs[1,k].set_xlabel('n')
            axs[1,k].set_ylabel('error/C_n')
            axs[0,k].plot(N, res_MSE)
            axs[1,k].plot(N, res_MSE/res_rate, label= r)
            axs[1,k].legend()
    fig.savefig('sim_7_1.jpg')
    
def simulation_7_2(R_1, R_2, R_val): # R_1=200, R_2=10
    N = [10,20,50,100,150,200,300,500, 1000]
    D= [10, 50, 200]
    fig, axs = plt.subplots(3, 3, figsize=(10,9))
    for k in range(len(D)):
        d = D[k]
        for r in ['r_2', 'r_3', 'r_10']:
            S = LHS_params[r][1]
            L = LHS_params[r][2]
            C = (72/np.pi)*(np.pi/16)**(S/(2*d))
            sup_r = LHS_params[r][3]
            E = C*((S**2)/(S-1))**(S/(2*d)) + 2*np.exp(-1)*sup_r**2
            res_MSE = np.zeros((len(N)))
            res_M_val= np.zeros((len(N)))
            res_k_val= np.zeros((len(N)))
            for i in range(len(N)): 
                n_val = N[i]
                M_val = np.random.choice(list(range(1,d)))
                k_val = np.random.choice(list(range(1,n_val)))
                X = np.random.uniform(size=(n_val,d))
                eps = np.random.normal(size=n_val)
                y = targets[r](X) + eps
                D_1 = decisiontree1(X, y, kn=k_val, D_n=0, M_n=M_val)
                mse=0
                for t in range(n_val):
                    mse=mse+(1/n_val)*(D_1.predict(X[t]) - y[t])**2
                for j in range(R_val):
                    M_new = np.random.choice(list(range(1,d)))
                    k_new = np.random.choice(list(range(1,n_val)))
                    D_1 = decisiontree1(X, y, kn=k_new, D_n=0, M_n=M_new)
                    mse_new=0
                    for t in range(n_val):
                        mse_new=mse_new+(1/n_val)*(D_1.predict(X[t]) - y[t])**2
                    if mse_new < mse:
                        M_val = M_new
                        k_val = k_new
                        mse = mse_new
                print('r: ', r, 'd: ', d, 'n: ', N[i], 'M: ', M_val, 'k: ', k_val)
                    
                X_1 = np.random.uniform(size=(R_2,d))
                y_1 = targets[r](X_1)
                Z= np.zeros(R_2)
                
                X = np.random.uniform(size=(N[i],d))
                eps = np.random.normal(size=N[i])
                y = targets[r](X) + eps
                RF = randomforest1(X[0:N[i]//2], y[0:N[i]//2], X[N[i]//2:N[i]], y[N[i]//2:N[i]], kn=k_val, m=R_1 , Mn=M_val)
                for j in range(R_2):
                    Z[j]=RF.predict(X_1[j])
                MSE = (1/R_2)*np.sum((y_1-Z)**2)
                kn_1 = (L**2/E)**(S*np.log(2)/(0.75+S*np.log(2)))*(N[i]**(S*np.log(2)/(0.75+S*np.log(2))))
                res_MSE[i]= MSE   
                res_M_val[i] = M_val
                res_k_val[i] = k_val/kn_1
            axs[0,k].set_xlim([0,1000])
            axs[0,k].set_xlabel('n')
            axs[0,k].set_ylabel('error')
            axs[1,k].set_xlim([0,1000])
            axs[1,k].set_xlabel('n')
            axs[1,k].set_ylabel('M_n estimate')
            axs[2,k].set_xlim([0,1000])
            axs[2,k].set_xlabel('n')
            axs[2,k].set_ylabel('k_n estimate')
            axs[0,k].plot(N, res_MSE, label = r)
            axs[1,k].plot(N, res_M_val, label= r)
            axs[2,k].plot(N, res_k_val, label= r)
            axs[0,k].legend()
    fig.savefig('sim_7_2.jpg')
    
def simulation_7_3(R_1, R_2): #200, 1000
    for r in ['r_1','r_4', 'r_6','r_7','r_8','r_9','r_10']:
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle(r)
        ax1.set_xlim([0,500])
        ax1.set_ylim([0,100])
        ax1.set_xlabel('n')
        ax1.set_ylabel('MSE')
        ax2.set_xlim([0,500])
        ax2.set_ylim([0,100])
        ax2.set_xlabel('n')
        ax2.set_ylabel('MSE')
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        N = [10,20,50,100,150,200,300,500, 1000]
        D = [4,10,20,50,100,300]
        labels=list()
        res_Biau = np.zeros((len(N),len(D)))
        res_Breiman = np.zeros((len(N),len(D)))
        for i in range(len(D)):
            d= D[i]
            labels.append('d = %s' %D[i])
            for j in range(len(N)):
                kn_1 = validate_sparsity_2(r,N[j], d, 10)
                print('kn: ', kn_1)
                print('Di,Nj: ', D[i],N[j])
                res_Biau[j,i], res_Breiman[j,i] = test_sparsity_2(r, N[j], D[i], kn_1, D[i], R_1, R_2)
            ax1.scatter(np.array(N)/2, res_Biau[:,i], c = colors[i], label = labels[i])
            ax1.plot(np.array(N)/2, res_Biau[:,i], c = colors[i])
            ax1.legend()
            ax2.scatter(np.array(N)/2, res_Breiman[:,i], c = colors[i], label = labels[i])
            ax2.plot(np.array(N)/2, res_Breiman[:,i], c = colors[i])
        fig.savefig('sim_7_3_%s.jpg' %str(r))
        
def simulation_7_4(R_1, R_2): #200, 1000
    for r in ['r_sine']:
        fig, axs = plt.subplots(3,3, figsize=(10,9))
        fig.suptitle('sine')
        N = [20,50,200,300,500, 1000, 5000, 10000,50000]
        d=5
        for i in range(3):
            for j in range(3):
                n = N[3*i+j]
                kn_1 = validate_sparsity_2(r,n, d, 10)
                print('kn: ', kn_1)
                print('Nj: ', n)
                X = np.random.uniform(size=(n,d))
                eps = np.random.normal(size=(n))
                y = targets[r](X) + eps
                RF_Biau = randomforest2(X[0:n//2], y[0:n//2], X[n//2:n], y[n//2:n], N=n//5, kn=kn_1, m=R_1 , Mn=d)
                X_1 = np.random.uniform(size=(R_2,d))
                y_1 = targets[r](X_1)
                Z_1= np.zeros(X_1.shape[0])
                for k in range(X_1.shape[0]):
                    Z_1[k]=RF_Biau.predict(X_1[k])
                axs[i,j].text(0.1,0.1, 'n= %s' %str(N[i*3+j]), transform=axs[i,j].transAxes)
                axs[i,j].scatter(X_1[:,0], Z_1, s=2, c='r')
                axs[i,j].scatter(X_1[:,0], y_1, s=2, c='b')
        fig.savefig('sim_7_4_%s.jpg' %str(r))
