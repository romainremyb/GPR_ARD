import numpy as np
from statsmodels.stats.diagnostic import kstest_normal
from sklearn.linear_model import LinearRegression   

def genRandomG(dim,means,stds,covs,size):
    if len(covs)!=dim-1:
        print('error shape in covs')
    elif len(means)!=dim:
        print('error shape in means')
    elif len(stds)!=dim:
        print('error shape in stds')
    else:
        features=[]
        for i in range(1,dim):
            # check if matrx is positive semi-definite
            if np.all(np.linalg.eigvals(np.array([[stds[0],covs[i-1]],[covs[i-1],stds[i]]])) > 0)==False:
                d=np.random.normal(loc=0, scale=0.05, size=1).item()
                while np.all(np.linalg.eigvals(np.array([[stds[0],covs[i-1]],[covs[i-1],np.abs(stds[i]+d)]])) > 0)==False:
                    d=d+np.abs(np.random.normal(loc=0, scale=0.05, size=1)).item()
                one=np.random.multivariate_normal([means[0],means[i]],[[stds[0],covs[i-1]],[covs[i-1],np.abs(stds[i]+d)]],size=1000)
            else:
                one=np.random.multivariate_normal([means[0],means[i]],[[stds[0],covs[i-1]],[covs[i-1],stds[i]]],size=1000)
            # saves random variables generated from to-feature1-covariances
            features.append(one[:,1])  

        matrx=np.zeros((dim,dim)) # build covariance matrix
        for i in range(dim):  
            matrx[i,i]=stds[i] 

        for i in range(1,dim): #save to-feature1-covariances
            matrx[0,i]=covs[i-1]
            matrx[i,0]=covs[i-1]

        for i in range(1,dim-1):  # saves infered co-variances
            for j in range(i+1,dim):
                c=np.cov(features[i-1],features[j-1])[0,1]
                matrx[i,j]=c
                matrx[j,i]=c
        if np.all(np.linalg.eigvals(np.array(matrx)) > 0)==False:
            while np.all(np.linalg.eigvals(np.array(matrx)) > 0)==False:
                v=np.random.choice(np.arange(dim))
                matrx[v,v]=np.abs(matrx[v,v]+np.random.normal(loc=0, scale=0.05, size=1).item())

        return np.random.multivariate_normal(means,matrx,size=size)


class genData: 

    def __init__(self, shape_X, means,stds,covs, n) -> None:
        if covs.any()==None:
            matrx=np.zeros((shape_X,shape_X))
            for i in range(shape_X):
                matrx[i,i]=stds[i]
            self.x, self.covs=np.random.multivariate_normal(means,matrx,size=n), matrx
        else:
            self.x=genRandomG(shape_X, means, stds, covs,size=n)
            self.covs=np.cov(self.x.T)

        self.number_features=shape_X
        self.y=np.zeros(n)
        self.n=n
        self.featuresIndex=[]
        self.nonpred=[]
        self.respVar=[]
        for i in range(shape_X):
            self.respVar.append({'index':[],'responses':[],'respVar':[],'R^2':[]})


    def getRsquarre(self,x,r):
        reg = LinearRegression().fit(x.reshape(-1, 1), r)
        return reg.score(x.reshape(-1, 1), r)

    def linear(self, respVar, idx):
        if idx not in self.featuresIndex:
            sigmaS=np.std(self.x[:,idx])**2
            w=(respVar/sigmaS)**0.5
            self.featuresIndex.append(idx)
            self.respVar[idx]['responses'].append(w*self.x[:,idx])
            self.respVar[idx]['index'].append(idx)
            self.respVar[idx]['respVar'].append(respVar)
            self.respVar[idx]['R^2'].append(self.getRsquarre(self.x[:,idx],self.respVar[idx]['responses'][0]))
            return True
        else: 
            return False

    def polynomial(self, respVar, idx):
        if idx not in self.featuresIndex:
            mu=np.mean(self.x[:,idx])
            sigmaS=np.std(self.x[:,idx])**2
            a=(respVar/(4*(mu**2)*sigmaS + 2*sigmaS**2))**0.5
            a=np.random.uniform(-a,a,1)[0] # sample a param from uniform law
            c=(respVar-(a**2)*(4*(mu**4)*sigmaS + 2*sigmaS**2))/sigmaS
            roots=np.roots([1,2*a,-c])
            if np.iscomplex(roots[0]):
                return False
            else:
                b=np.random.choice(roots)
                self.featuresIndex.append(idx)
                self.respVar[idx]['index'].append(idx)
                self.respVar[idx]['responses'].append((a*self.x[:,idx]**2) + b*self.x[:,idx])
                self.respVar[idx]['respVar'].append(respVar)
                self.respVar[idx]['R^2'].append(self.getRsquarre(self.x[:,idx],self.respVar[idx]['responses'][0]))
                return True
        else:
            return False


    def sin(self, respVar, idx):
        if idx not in self.featuresIndex:
            mu=np.mean(self.x[:,idx])
            sigmaS=np.std(self.x[:,idx])**2
            w=np.random.uniform(-(np.pi/sigmaS),(np.pi/sigmaS),1)[0]    # if var(x)=1, extremely linear at w=0 and unlinear at pi and -pi
            r=respVar/((np.cos(w*mu)*w)**2)
            r=np.abs(r/(sigmaS+0.5*((np.cos(w*mu)*w)**2)*np.sin(w*mu)*sigmaS**2))
            a=np.random.choice(np.array([-(r**0.5),r**0.5]))
            self.featuresIndex.append(idx)
            self.respVar[idx]['index'].append(idx)
            self.respVar[idx]['responses'].append(a*np.sin(w*self.x[:,idx]))
            self.respVar[idx]['respVar'].append(respVar)
            self.respVar[idx]['R^2'].append(self.getRsquarre(self.x[:,idx],self.respVar[idx]['responses'][0]))      
            return True
        else:
            return False 


    def intercept(self, i):
        self.y+=i

    def addNormalNonPredFeature(self, idx):
        if idx not in self.nonpred:
            self.nonpred.append(idx)
            return True
        else:
            return False

    def makeDataset(self, rate):   
        for i in self.featuresIndex:
            self.y=self.y+self.respVar[i]['responses'][0]
        v=np.std(self.y)
        self.y=self.y+np.random.normal(loc=0, scale=v*rate, size=self.n)
        return self.x, self.y, self.respVar, len(self.featuresIndex)+len(self.nonpred), self.nonpred, self.covs


    def checkNormalDistrib(self, threshold):
        p=kstest_normal(self.y)[1]
        if p>threshold:
            return True  
        else:
            return False