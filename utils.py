import numpy as np
import seaborn as sns
import torch
import gpytorch
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from gpytorch.kernels import Kernel
from gpytorch.functions import RBFCovariance
from gpytorch.settings import trace_mode
import warnings
from gpytorch.utils.warnings import GPInputWarning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models import ExactGPModel_standard, ExactGPModel_standardScaleK, ExactGPModel_modLengthscale, ExactGPModel_modLengthscaleScaleK
from dataGen import genData

def lognormal(x, mu, var):
    return np.mean(-np.log(np.sqrt(2*np.pi*var**2)) - 1/2*np.square((x-mu)/np.sqrt(var)))


class EarlyStopping:
	def __init__(self, patience):
		self.patience = patience
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.val_loss_min = np.inf
	def __call__(self, val_loss, model):
		score = val_loss
		if self.best_score is None:
			self.best_score = score
		elif score > self.best_score:
			self.counter += 1
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_score = score
			self.counter = 0

class trainUtils: # class where noise is added outside of the function

    def __init__(self, data, likelihood, addedNoiseAsRate) -> None:
        self.addedNoiseAsRate=addedNoiseAsRate
        self.data=data
        self.x, self.y, self.respVar, self.nk, self.nonpred, self.covs=self.data.makeDataset(addedNoiseAsRate)
        for i in self.respVar:
            i.pop('responses', None)
        
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test=torch.from_numpy(X_train), torch.from_numpy(X_test), torch.from_numpy(y_train), torch.from_numpy(y_test)
        self.scaler = StandardScaler()
        self.likelihood=likelihood
        self.results={'sigmay':[np.std(self.y)], 'nk':[self.nk], 'nonpredIdx':[self.nonpred], 'respVar':self.respVar, 'covs':[self.covs],'ExactMarginalLogLikelihood':[], 'lengthscales':[], 'iterations':[], 'outputlengthscale':[], 'unexplainedVarRate':[self.addedNoiseAsRate]}

    def clearResults(self):
        self.results={'sigmay':[np.std(self.y)], 'nk':[self.nk], 'nonpredIdx':[self.nonpred], 'respVar':self.respVar, 'covs':[self.covs],'ExactMarginalLogLikelihood':[], 'lengthscales':[], 'iterations':[], 'outputlengthscale':[], 'unexplainedVarRate':[self.addedNoiseAsRate]}
    
    def getLengthscales(self, outputscale):
        if outputscale==True:
            return self.model.covar_module.base_kernel.lengthscale[0]
        else:
            return self.model.covar_module.lengthscale[0]

    def regularizationL2(self,outputscale):
        if type(self.lambdal1)==float:
            return self.getLengthscales(outputscale).pow(-4).abs().sum()*self.lambdal1
        else:
            return 0

    def regularizationL1(self, outputscale):
        if type(self.lambdal1)==float:
            return self.getLengthscales(outputscale).pow(-2).abs().sum()*self.lambdal1 # sqrt of (1/l^2)=1/l
        else:
            return 0


    def train(self, model, scaled, outputscale, learning_rate, l1_lr, Reg, max_epochs, patience, all_lengthscale): #modifK # add output lengthscale
        warnings.simplefilter("ignore", GPInputWarning)

        if scaled==True:
            self.scaler.fit(self.X_train)
            self.Xtrain=torch.from_numpy(self.scaler.transform(self.X_train))
            self.Xtest=torch.from_numpy(self.scaler.transform(self.X_test))
        else:
            self.Xtrain=self.X_train
            self.Xtest=self.X_test

        if all_lengthscale==True:
            self.model=model(self.Xtrain, self.y_train, self.likelihood, self.Xtrain.shape[1])
        else:
            self.model=model(self.Xtrain, self.y_train, self.likelihood, None)

        self.learning_rate = learning_rate
        self.lambdal1=l1_lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        early_stopping = EarlyStopping(self.patience)
        flag=False
        for epoch in range(self.max_epochs):
            train_loss = self.train_step(Reg,outputscale)
            val_loss=self.val_step(Reg,outputscale)
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                flag=True
                self.results['ExactMarginalLogLikelihood'].append(val_loss)
                self.results['lengthscales'].append(self.getLengthscales(outputscale))
                self.results['iterations'].append(epoch)
                if outputscale==True:
                    self.results['outputlengthscale'].append(self.model.covar_module.outputscale.item())
                break
        if flag==False:
            self.results['ExactMarginalLogLikelihood'].append(val_loss)
            self.results['lengthscales'].append(self.getLengthscales(outputscale))
            self.results['iterations'].append(epoch)
            if outputscale==True:
                self.results['outputlengthscale'].append(self.model.covar_module.outputscale.item())
	

    def train_step(self, Reg,outputscale):
        self.model.train() 
        self.likelihood.train()
        self.optimizer.zero_grad() #
        if Reg==True:
            train_loss = - self.mll(self.model(self.Xtrain), self.y_train) + self.regularizationL1(outputscale)
        else:
            train_loss = - self.mll(self.model(self.Xtrain), self.y_train) + self.regularizationL2(outputscale)
        train_loss.backward()
        self.optimizer.step()
        return train_loss.item()

    def val_step(self, Reg, outputscale):
        self.model.eval() 
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.eval_cg_tolerance(1.e-1), gpytorch.settings.max_preconditioner_size(100), gpytorch.settings.cholesky_jitter(1e-1), gpytorch.settings.cholesky_max_tries(10):
            if Reg==True:
                val_loss= - self.mll(self.likelihood(self.model(self.Xtest)), self.y_test) + self.regularizationL1(outputscale)
                return val_loss.item()
            else:
                val_loss= - self.mll(self.likelihood(self.model(self.Xtest)), self.y_test) + self.regularizationL2(outputscale)
                return val_loss.item()               


    