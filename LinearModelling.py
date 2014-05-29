import numpy as np
import pandas as pd 
import pylab as pl 
import sklearn.linear_model as lm 
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
import matplotlib.pyplot as plt



#Collection of Linear Models
#1.Linear Model (lm_model)
#2.Lasso Model (lasso_model)
#3.Bayesian Ridge Model (br_model)
#4.ARD models (ard_model)
#5.SVM with linear kernel (svm)
#6.LARS Lasso Model (lars_model)


#usage
'''
lms = LinearModels(train_df,test_df)
lms.fittingModels(['X1','X2'],['X3'],kernel="poly") #kernel is just for choice of kernel for SVR
lms.predictions()

#an important point

in order to get the respective predictions for test data
please just concatenate "_predict" ahead respective model name keyword.

e.g. lms.lm_model_predict will give array of predicted outcome of model over test data.  


'''




class LinearModels(object):

	def __init__(self,train_df,test_df):
		self.train_df= train_df
		self.test_df= test_df
		
	def fittingModels(predictors,out_fv,kernel='linear'):
		#logistic regression model
		self.predictors=predictors
		self.out_fv=out_fv
		self.kernel=kernel
		
		#linear regression
		self.lr_model = lm.LinearRegression().fit(self.train_df.loc[:,self.predictors].values,self.train_df.loc[:,self.out_fv].values)

		#simple lasso model
		self.lasso_model = lm.Lasso(alpha = 0.1).fit(self.train_df.loc[:,self.predictors].values,self.train_df.loc[:,self.out_fv].values)

		#Naive Bayes algorithm
		self.nb_model = lm.GaussianNB().fit(self.train_df.loc[:,self.predictors].values,self.train_df.loc[:,self.out_fv].values)

		#Bayesian Ridge Model - adapts to data at hand and regularized parameter is used
		self.br_model = lm.BayesianRidge().fit(self.train_df.loc[:,self.predictors].values,self.train_df.loc[:,self.out_fv].values)

		#ARD Regression Model
		self.ard_model = lm.ARDRegression().fit(self.train_df.loc[:,self.predictors].values,self.train_df.loc[:,self.out_fv].values)

		#SVR with linear kernel
		self.svm = SVR(C=1.0, epsilon=0.2, kernel=self.kernel).fit(self.train_df.loc[:,self.predictors].values,self.train_df.loc[:,self.out_fv].values)

		#if number of dimensions are significantly larger than number of points then 
		#LARS Lasso can be used

		self.lars_model = lm.LassoLars(alpha=0.1).fit(self.train_df.loc[:,self.predictors].values,self.train_df.loc[:,self.out_fv].values)


	def predictions():

		print "Simple Linear Regression Prediction"
		print self.lr_model.score(self.test_df.loc[:,self.predictors].values,self.test_df.loc[:,self.out_fv].values)
		self.lr_model_predict=self.lr_model.predict(self.test_df.loc[:,self.predictors].values)
		print "####################################"
		print "Simple LASSO Regression Prediction"
		print self.lasso_model.score(self.test_df.loc[:,self.predictors].values,self.test_df.loc[:,self.out_fv].values)
		self.lasso_model_predict=self.lasso_model.predict(self.test_df.loc[:,self.predictors].values)
		print "####################################"
		print "Bayesian Ridge Regression Prediction"
		print self.br_model.score(self.test_df.loc[:,self.predictors].values,self.test_df.loc[:,self.out_fv].values)
		self.br_model_predict=self.br_model.predict(self.test_df.loc[:,self.predictors].values)
		print "####################################"
		print "ARD Regression Prediction"
		print self.ard_model.score(self.test_df.loc[:,self.predictors].values,self.test_df.loc[:,self.out_fv].values)
		self.ard_model_predict=self.ard_model.predict(self.test_df.loc[:,self.predictors].values)
		print "####################################"
		print "Support Vector Regression Prediction"
		print self.svm.score(self.test_df.loc[:,self.predictors].values,self.test_df.loc[:,self.out_fv].values)
		self.svm_predict=self.svm.predict(self.test_df.loc[:,self.predictors].values)
		print "####################################"
		print "LARS LASSO Regression Prediction"
		print self.lars_model.score(self.test_df.loc[:,self.predictors].values,self.test_df.loc[:,self.out_fv].values)
		self.lars_model_predict=self.lars_model.predict(self.test_df.loc[:,self.predictors].values)
		print "####################################"
			










