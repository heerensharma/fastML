import numpy as np
import pandas as pd 
import pylab as pl 
import sklearn.linear_model as lm
import statsmodels.formula.api as smf
import statsmodels.api as sm

#usage
'''
This class deals with advanced classes of Generalized Linear Models.
It tries to comprise multi-class classification modelling
1. Multinomial Logistic Regression
2. Ordered Probit
'''


class MulticlassLogit(object):

	def __init__(self,train_df,test_df,):
		self.train_df= train_df
		self.test_df= test_df

	def modelFitting(self,predictors,out_fv):
		self.predictors = predictors
		self.out_fv = out_fv
		
		glm_formula = ("+").join(self.out_fv)+"~"+("+").join(self.predictors)

		#simple multionomial logisitic regression model
		self.mn_logit_model = sm.MNLogit(glm_formula,data=self.train_df).fit()

		#simple ordered probit regression model
		self.ordered_probit  = 




	def getSummary():
		print "Multinomial Logistic regression"
		print "###############################"
		print self.mn_logit_model.summary() 


	def predictions(self)



