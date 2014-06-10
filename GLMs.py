import numpy as np
import pandas as pd 
import pylab as pl 
import sklearn.linear_model as lm
import statsmodels.formula.api as smf
import statsmodels.api as sm

#usage
'''
This class deals with advanced classes of Generalized Linear Models.
It tries to comprise binary classification modelling
1. Poisson Regression
2. Logistic Regression
'''

class LogisticModels(object):

	def __init__(self,train_df,test_df,):
		self.train_df= train_df
		self.test_df= test_df

	def modelFitting(self,predictors,out_fv):
		self.predictors = predictors
		self.out_fv = out_fv
		
		glm_formula = ("+").join(self.out_fv)+"~"+("+").join(self.predictors)

		#simple logistic regression which is primarily used for two-values classifier
		self.logit_model = sm.Logit(train_df[out_fv],train_df[predictors]).fit()
		# self.logit_model = smf.glm(glm_formula,data=self.train_df,family=sm.families.Family(link=sm.families.links.log)).fit()

		#poisson regression - modelling positive quantities over large scale
		
		self.ps_model = smf.glm(glm_formula,data=self.train_df,family=sm.families.Poisson(link=sm.families.links.log)).fit()
		
		


	def predictions(self)



