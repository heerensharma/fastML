import numpy as np
import pandas as pd 
import pylab as pl 
import matplotlib.pyplot as plt
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import TanhLayer
from pybrain.supervised.trainers import BackpropTrainer
import itertools
#sample Neural network module using PyBrain.
#Though it is not quite generic purpose yet as accepts pandas dataframes for training and testing datasets
#Checks for testing vector or dataframes are required to be implemented in order to make as an API.


class NNet(object):

	def __init__(self,inputs,states,outputs):
		#Building network
		self.inputs=inputs
		self.outputs=outputs
		self.net = buildNetwork(inputs,states,outputs, hiddenclass=TanhLayer)


	def createDataSet(self,ds,predictors,to_predict):
		self.ds=SupervisedDataSet(self.inputs,self.outputs)
		self.predictors=predictors
		self.to_predict=to_predict
		#specifying datasets and adding samples to this dataset
		for inp, out in itertools.izip(ds[predictors].values,ds[to_predict].values):
			self.ds.addSample(inp,out)				

	def testNN(self,test_ds):
		#training the network
		#It will take a while as there are 1000 times training over datasets will be performed for this network  
		trainer=BackpropTrainer(self.net,self.ds,learningrate=0.05)
		trainer.trainUntilConvergence( verbose = False, validationProportion = 0.15, maxEpochs = 1000, continueEpochs = 10 )
		self.predicted=[]
		#testing the network
		for input1 in test_ds[self.predictors].values:
			self.predicted.append(self.net.activate(input1))
		self.expected = test_ds[self.to_predict].values

	def plot_output(self):
		pl.figure()
		pl.plot(self.predicted,'x-',color="blue")
		pl.plot(self.expected,'o-',color="green")
		pl.legend(['NN Prediction','Expected'])
		pl.show()
	
	def getOutput(self):
		print self.predicted	



