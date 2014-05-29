import numpy as np
from sklearn.gaussian_process import GaussianProcess
import pandas as pd 
from matplotlib import pyplot as pl

#usage
#This is simple Gaussian process regression model
#Additional functionality like, changing various mean and covariance paramerter, 
#to give strength to use are required to be added yet.
'''
gpr=GaussianProcessRigression(train_df,test_df)
gpr.fitModel(['colname1'],['colname2'])
gpr.givePlot()

'''



class GaussianProcessRigression(object):

	def __init__(self,data_frame,test_df):
		self.df=data_frame
		self.test=test_df

	def fitModel(predictors,output_var):
		self.predictors=predictors
		self.output_var=output_var
		self.gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100)
		self.gp.fit(self.df[predictors].values,self.df[output_var].values)
		self.y_pred, self.MSE = self.gp.predict(self.test[predictors], eval_MSE=True)
		self.sigma = np.sqrt(self.MSE)

	#works only when you have single parameter for predictors only 
	#as for multiple parameters doesn't make sense to make a 2D curve.
	def givePlot(xlabel='$x$',ylabel='$f(x)$'):
		fig = pl.figure()
		pl.plot(self.test[self.predictors], self.test[self.output_var], 'r:', label=u'Actual curve')
		# pl.plot(X, y, 'r.', markersize=10, label=u'Observations')
		pl.plot(self.test[self.predictors], self.y_pred, 'b-', label=u'Prediction')
		pl.fill(np.concatenate([self.test[self.predictors], self.test[self.output_var]]), \
		        np.concatenate([self.y_pred - 1.9600 * self.sigma,
		                       (self.y_pred + 1.9600 * self.sigma)[::-1]]), \
		        alpha=.5, fc='b', ec='None', label='95% confidence interval')
		pl.xlabel(xlabel)
		pl.ylabel(ylabel)
		pl.ylim(-10, 20)
		pl.legend(loc='upper left')
		pl.show()