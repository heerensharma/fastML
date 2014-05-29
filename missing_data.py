import numpy as np
import pandas as pd 

#Simple median and mode based approach for missing data values for 
#numeric and categorical feature variables respectively

#usage
'''
fmd = FillMissingData(df)

#all numeric ones conversion
fillFloatVar(['v2','v3','v8','v14','v15','v17'])

#all about categorical variables 
fillCategoricalVar(['v1','v4','v5','v6','v7','v9','v10','v12','v13','classLabel'])

'''




class FillMissingData(object):

	#initialize class with pandas data frame as input.
	def __init__(self,df):
		self.data_frame=df



	#convert the categorical variables into the respective categories 
	def __convertCategories(self,input_df,col):
		catgs = list(enumerate(np.unique(input_df[col])))    # determine all values of Column,
		catgs_dict = { name : i for i, name in catgs }              # set up a dictionary in the form  Catgs : index
		input_df[col] = input_df[col].map( lambda x: catgs_dict[x]).astype(int)     # Convert all Columns strings to int



	#fill the missing values and convert the column into the floating vector.
	def fillFloatVar(self,columns):
		for col in columns:
			if len(self.data_frame[col][self.data_frame[col].isnull()]) > 0:
				self.data_frame.loc[self.data_frame[col].isnull(),col] = self.data_frame[col].dropna().median()
			self.data_frame[col] = self.data_frame[col].astype(float)


	#fill the missing values and convert the column into the integer vector type column.
	def fillIntVar(self,columns):
		for col in columns:
			if len(self.data_frame[col][self.data_frame[col].isnull()]) > 0:
				self.data_frame.loc[self.data_frame[col].isnull(),col] = self.data_frame[col].dropna().median()
			self.data_frame[col] = self.data_frame[col].astype(int)

	#filling categorical variables with the respective numeric values
	def fillCategoricalVar(self,columns):
		for col in columns:
			if len(self.data_frame[self.data_frame[col].isnull()].index) > 0:
				self.data_frame[col][self.data_frame[col].isnull()] = self.data_frame[col].dropna().mode().values
			self.__convertCategories(self.data_frame,col)