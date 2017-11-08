
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from pylab import *


# Class that contains all the paramaters of the network
class network:
	# Initialize neural network (1 hidden layer)
	def __init__(self, n_input = 2, n_hidden = 2, n_output = 2, lrate = 0.5, opt = 'SGD', 
		mom = 0.0, seed=None):
		# Set learning rate for back propagation and momentum 
		self.lrate = lrate
		self.momentum = mom
		# Save number of nodes per layer
		self.n_input = n_input
		self.n_hidden = n_hidden
		self.n_output = n_output
		# Initialize weight matrices (+1 in first dimension is for bias)
		if seed is not None:
			np.random.seed(seed)
		self.W = {'i->h': np.random.rand(n_input+1,n_hidden),
			'h->o': np.random.rand(n_hidden+1,n_output)}
		# Initialize delta weights matrices (used when doing batch training)
		self.dW = {'i->h': np.zeros((n_input+1,n_hidden),dtype=float),
			'h->o': np.zeros((n_hidden+1,n_output),dtype=float)}
		self.dWprec = {'i->h': np.zeros((n_input+1,n_hidden),dtype=float),
			'h->o': np.zeros((n_hidden+1,n_output),dtype=float)}
		# Initialize inputs and outputs
		self.inputs = {}
		self.outputs = {}
		# parameters for batch optimization
		self.opt = opt
		self.deltas = {'i->h': np.zeros((self.n_input+1,self.n_hidden),dtype=float),
			'h->o': np.zeros((self.n_hidden+1,self.n_output),dtype=float)}
		# Error(loss) vs epoch
		self.error = []
		return

	# Add bias unit to the vector of inputs
	def addUnitBias(self,i):
		return np.append(i,1.0)

	# Sigmoid activation function
	def activate(self, o):
		return 1.0 / (1.0 + np.exp(-o))

	# Derivative of sigmoid function
	def d_activate(self, o):
		return o * (1.0 - o)

	# Perform linear combination of inputs and weights + bias
	def collapse(self, W,i):
		i = self.addUnitBias(i)
		return np.dot(i,W)

	# Forward propagation step
	def forwardPropagation(self, x):
		# First layer i -> h
		self.inputs['x'] = x
		self.inputs['h'] = self.collapse(self.W['i->h'],x)
		self.outputs['h'] = self.activate(self.inputs['h'])
		# Second layer h -> o
		self.inputs['o'] = self.collapse(self.W['h->o'],self.outputs['h'])
		self.outputs['o'] = self.activate(self.inputs['o'])
		return

	# Backward propagation and update of the weights
	def backwardPropagation(self, x, y):
		# Calculate delta
		deltaNode = - (y - self.outputs['o']) * self.d_activate(self.outputs['o'])
		# Backpropagate error and calculate delta i->h
		deltaW = np.zeros(self.W['i->h'].shape)
		biasedVec = self.addUnitBias(self.inputs['x'])
		for i in range(deltaW.shape[0]):
			for j in range(deltaW.shape[1]):
				s = np.dot(deltaNode,self.W['h->o'][j,:])
				deltaW[i,j] = s * self.d_activate(self.outputs['h'][j]) * biasedVec[i]

		self.dW['i->h'] = deltaW

		# Backpropagate error and calculate delta h->o
		deltaW = np.zeros(self.W['h->o'].shape)
		biasedVec = self.addUnitBias(self.outputs['h'])
		for i in range(deltaW.shape[0]):
			for j in range(deltaW.shape[1]):
				deltaW[i,j] = deltaNode[j] * biasedVec[i]

		self.dW['h->o'] = deltaW

		return

	# Update weights of the network
	def updateWeights(self, batch = False):
		# Update weights
		if (self.opt == 'SGD'):
			# Normal update with gradient
			self.W['i->h'] = self.W['i->h'] - self.lrate * self.dW['i->h']
			self.W['h->o'] = self.W['h->o'] - self.lrate * self.dW['h->o']
			# Add momentum
			self.W['i->h'] = self.W['i->h'] - self.momentum * self.dWprec['i->h']
			self.W['h->o'] = self.W['h->o'] - self.momentum * self.dWprec['h->o']
			# Save currend dW for next iteration
			self.dWprec['i->h'] = self.dW['i->h']
			self.dWprec['h->o'] = self.dW['h->o']
		elif (network.opt == 'BATCH'):
			# Normal update with gradient
			self.W['i->h'] = self.W['i->h'] - self.lrate * self.deltas['i->h']
			self.W['h->o'] = self.W['h->o'] - self.lrate * self.deltas['h->o']
			# Add momentum
			self.W['i->h'] = self.W['i->h'] + self.momentum * self.dWprec['i->h']
			self.W['h->o'] = nself.W['h->o'] + self.momentum * self.dWprec['h->o']
			# Save currend dW for next iteration
			self.dWprec['i->h'] = self.deltas['i->h']
			self.dWprec['h->o'] = self.deltas['h->o']
		return

	# Get total error (loss) of the network
	def getError(self,x,y):
		n_samples = x.shape[0]
		err = 0.0
		for i in range(n_samples):
			self.forwardPropagation(x[i,:])
			delta = y[i,:] - self.outputs['o']
			err += np.dot(np.transpose(delta),delta)
		return np.divide(err, n_samples)

	# Calculate accuracy of the network
	def getAccuracy(self,x,y):
		# Get prediction from x
		ypred = self.predict(x)
		y = [np.argmax(y[i,:]) for i in range(len(y))]
		# Compare with target
		accuracy = np.divide(np.nansum([1 for i in range(len(ypred)) if ypred[i] == y[i]]),
							 len(ypred))
		return accuracy

	# Predict the output class given the input
	def predict(self,x):
		n_samples = x.shape[0]
		ypred = []
		for i in range(n_samples):
			self.forwardPropagation(x[i,:])
			ypred.append(np.argmax(self.outputs['o']))
		return ypred

	# Initialize deltas for batch optimization
	def initDeltas(self):
		self.deltas = {'i->h': np.zeros((self.n_input+1,self.n_hidden),dtype=float),
			'h->o': np.zeros((self.n_hidden+1,self.n_output),dtype=float)}
		return

	# Update deltas for batch optimization
	def updateDeltas(self):
		self.deltas['i->h'] = self.deltas['i->h'] + self.dW['i->h']
		self.deltas['h->o'] = self.deltas['h->o'] + self.dW['h->o']
		return

	def shuffle(self,x, y):
		idx = np.arange(len(y))
		np.random.shuffle(idx)
		return (x[idx, :], y[idx, :])

	def train(self, x, y, xtest=None, ytest=None, maxIter=100, lrate=0.1, opt='SGD', tol = 1e-8,
			  bsize = 1, nbatch = 1, mom = 0.0, verbose = True,reshuffle = True):

		# Check training data sizes and determine number of training samples
		n_training_samples = x.shape[0]
		if (n_training_samples != y.shape[0]):
			print("Error 1 in training data size!")
			exit()

		if xtest is not None and ytest is not None:
			error_test = []

		bias = np.ones((x.shape[0], 1), dtype=float)  # bias column vector
		in_des_mtx = np.hstack((x, bias))  # input design matrix
		# Initiate iterations to train neural network
		convergence = False
		it = 0
		while (not convergence):
			# Train the network on each data samples (SDG/online training)
			if (opt == 'SGD'):
				# Reshuffle data at each epoch
				if (reshuffle):
					x, y = self.shuffle(x,y)
					in_des_mtx = np.hstack((x, bias))  # input design matrix
				for m in range(n_training_samples):
					# Get data sample in form of row vector
					xx = x[m, :]
					yy = y[m, :]
					# Forward propagate
					self.forwardPropagation(xx)
					# Backward propagate error
					self.backwardPropagation(xx, yy)
					# Update weights
					self.updateWeights()
				# Get error for the iteration
				out1 = self.activate(np.dot(in_des_mtx, self.W['i->h']))
				layer_des_mtx = np.hstack((out1, bias))
				out2 = self.activate(np.dot(layer_des_mtx, self.W['h->o']))
				self.error.append(self.getError(x,y))
				# Get error of test seet
				if xtest is not None and ytest is not None:
					error_test.append(self.getError(xtest,ytest))
				if (verbose):
					print("SGD - Iter: " + str(it + 1) + " *** Error: " + str(self.error[it]))
				if (it >= 2):
					if ((it >= maxIter) or (abs(self.error[it] - self.error[it - 1]) < tol)):
						convergence = True
				it = it + 1

		if xtest is not None and ytest is not None:
			return error_test

# Given input data x and a set of expected values y, train the network
# opt --> kind of training: 'SGD' == stochastic gradient descent, or online training,
# update the weights at each data point. 'BATCH' == batch training, update
# weights every batch of data fed to the network, the deltas are cumulated
# bsize --> batch size
# nbatch --> number of batches used to train
def trainNetwork(x, y, maxIter = 100, lrate = 0.1, n_hidden = 2, opt = 'SGD',
	tol = 1e-8, bsize = 1, nbatch = 1, mom = 0.0, verbose = True,
	reshuffle = True):
	
	# Determine input and output nodes given the shape of the training data
	n_input = x.shape[1]
	n_output = y.shape[1]
	
	# Check training data sizes and determine number of training samples
	n_training_samples = x.shape[0]
	if (n_training_samples != y.shape[0]):
		print("Error 1 in training data size!")
		exit()
	
	# Initialize neural network
	#neural_net = network(n_input = n_input, n_output = n_output,
	#		n_hidden = n_hidden, lrate = lrate, opt = opt, mom = mom)
	
	# Arrays used later for vectorized total error calculation
	bias = np.ones((x.shape[0],1),dtype=float) # bias column vector
	in_des_mtx = np.hstack((x,bias)) # input design matrix
	# Initiate iterations to train neural network
	convergence = False
	it = 0
	while (not convergence):
		# Train the network on each data samples (SDG/online training)
		if (opt == 'SGD'):
			# Reshuffle data at each epoch
			if (reshuffle):
				x,y = libd.shuffle(x,y)
				in_des_mtx = np.hstack((x,bias)) # input design matrix
			for m in range(n_training_samples):
				# Get data sample in form of row vector
				xx = x[m,:]
				yy = y[m,:]
				# Forward propagate
				self.forwardPropagation(xx)
				# Backward propagate error
				self.backwardPropagation(xx,yy)
				# Update weights
				self.updateWeights()
			# Get error for the iteration
			out1 = self.activate(np.dot(in_des_mtx,self.W['i->h']))
			layer_des_mtx = np.hstack((out1,bias))
			out2 = self.activate(np.dot(layer_des_mtx,self.W['h->o']))
			#self.error.append(self.getError)
			if (verbose):
				print("SGD - Iter: " + str(it+1) + " *** Error: " + str(error[it]))
			if (it >= 2):
				if ((it >= maxIter) or (abs(error[it] - error[it-1]) < tol) ):
					convergence = True
			it = it + 1
		elif (opt == 'BATCH'):
			# Train the network for each batch, update weights after every batch
			# by cumulating the random.sample(xrange(0, n_total), bsize)deltas
			for nb in range(nbatch):
				self.initDeltas()
				for m in range(bsize):
					# Get data sample in form of row vector
					xx = libd.getRowVector(x[nb],m)
					yy = libd.getRowVector(y[nb],m)
					# Forward propagate
					forwardPropagation(neural_net,xx)
					# Backward propagate error
					backwardPropagation(neural_net,xx,yy)
					# Cumulate deltas
					updateDeltas(neural_net)
				# Update weights
				updateWeights(neural_net,batch = True)
				# Calculate error for the iteration
				for m in range(bsize):
					yy = libd.getRowVector(y[nb],m)
					xx = libd.getRowVector(x[nb],m)
					forwardPropagation(neural_net,xx)
					error[i] = error[i] + getError(neural_net,xx,yy)
			print("BATCH - Iter: " + str(i+1) + " *** Error: " + str(error[i]))
	return neural_net, error
