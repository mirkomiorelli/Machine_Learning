
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from pylab import *
import lib_data as libd


# Class that contains all the paramaters of the network
class network:
	# Initialize neural network (1 hidden layer)
	def __init__(self, n_input = 2, n_hidden = 2, n_output = 2, lrate = 0.5, opt = 'SGD', 
		mom = 0.0):
		# Set learning rate for back propagation and momentum 
		self.lrate = lrate
		self.momentum = mom
		# Save number of nodes per layer
		self.n_input = n_input
		self.n_hidden = n_hidden
		self.n_output = n_output
		# Initialize weight matrices (+1 in first dimension is for bias)
		np.random.seed(0)
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
		return

# Add bias unit to the vector of inputs
def addUnitBias(i):
	return np.append(i,1.0)

# Sigmoid activation function
def activate(o):
	return 1.0 / (1.0 + np.exp(-o))
# Derivative of sigmoid function
def d_activate(o):
	return o * (1.0 - o)

# Perform linear combination of inputs and weights + bias
def collapse(W,i):
	i = addUnitBias(i)
	return np.dot(i,W)

# Forward propagation step
def forwardPropagation(network, x):
	# First layer i -> h
	network.inputs['x'] = x
	network.inputs['h'] = collapse(network.W['i->h'],x)
	network.outputs['h'] = activate(network.inputs['h'])
	# Second layer h -> o
	network.inputs['o'] = collapse(network.W['h->o'],network.outputs['h'])
	network.outputs['o'] = activate(network.inputs['o'])
	return

# Backward propagation and update of the weights
def backwardPropagation(network, x, y):
	# Calculate delta
	deltaNode = - (y - network.outputs['o']) * d_activate(network.outputs['o'])
	# Backpropagate error and calculate delta i->h
	deltaW = np.zeros(network.W['i->h'].shape)
	biasedVec = addUnitBias(network.inputs['x'])
	for i in range(deltaW.shape[0]):
		for j in range(deltaW.shape[1]):
			s = np.dot(deltaNode,network.W['h->o'][j,:])
			deltaW[i,j] = s * d_activate(network.outputs['h'][j]) * biasedVec[i]
	
	network.dW['i->h'] = deltaW
	
	# Backpropagate error and calculate delta h->o
	deltaW = np.zeros(network.W['h->o'].shape)
	biasedVec = addUnitBias(network.outputs['h'])
	for i in range(deltaW.shape[0]):
		for j in range(deltaW.shape[1]):
			deltaW[i,j] = deltaNode[j] * biasedVec[i]
	
	network.dW['h->o'] = deltaW
	
	return

# Update weights of the network
def updateWeights(network, batch = False):
	# Update weights
	if (network.opt == 'SGD'):
		# Normal update with gradient
		network.W['i->h'] = network.W['i->h'] - network.lrate * network.dW['i->h'] 
		network.W['h->o'] = network.W['h->o'] - network.lrate * network.dW['h->o']
		# Add momentum 
		network.W['i->h'] = network.W['i->h'] - network.momentum * network.dWprec['i->h']
		network.W['h->o'] = network.W['h->o'] - network.momentum * network.dWprec['h->o']
		# Save currend dW for next iteration
		network.dWprec['i->h'] = network.dW['i->h']
		network.dWprec['h->o'] = network.dW['h->o']
	elif (network.opt == 'BATCH'):
		# Normal update with gradient
		network.W['i->h'] = network.W['i->h'] - network.lrate * network.deltas['i->h']
		network.W['h->o'] = network.W['h->o'] - network.lrate * network.deltas['h->o']
		# Add momentum 
		network.W['i->h'] = network.W['i->h'] + network.momentum * network.dWprec['i->h']
		network.W['h->o'] = network.W['h->o'] + network.momentum * network.dWprec['h->o']
		# Save currend dW for next iteration
		network.dWprec['i->h'] = network.deltas['i->h']
		network.dWprec['h->o'] = network.deltas['h->o']
	return

# Get total error of the network
def getError(network,x,y):
	forwardPropagation(network,x)
	delta = y - network.outputs['o']
	err = np.dot(np.transpose(delta),delta)
	return err

# Get misclassification error (returns the number of misclassified datas)
def getErrorClass(network,x,y):
	n_tot = x.shape[0]
	miscl = 0
	for m in range(n_tot):
		xx = libd.getRowVector(x,m)
		yy = libd.getRowVector(y,m)
		pred_class = predict(network,xx)
		if (pred_class != np.argmax(yy)):
			miscl = miscl + 1 
	return miscl

# Calculate accuracy of the network
def getAccuracy(network,x,y):
	error = 0.0
	for i in range(x.shape[0]):
		xx = libd.getRowVector(x,i)
		yy = libd.getRowVector(y,i) 
		error = error + getError(network,xx,yy)
	accuracy = 100.0 * (1.0 - (1.0 * error / (1.0 * x.shape[0])))
	return accuracy

# Predict the output class given the input
def predict(network,x):
	forwardPropagation(network,x)
	return np.argmax(network.outputs['o'])

# Plot error vs epoch
def plotErrorVsEpoch(error_list,hidden_list = [2], logx = False, logy = False):
	
	color = ["blue","black","green","orange","red","magenta","lime","yellow","darkgreen","gray","aqua"]
	n_curves = len(error_list)
	if (len(error_list) != len(hidden_list)):
		print("Error 2, mismatch in size of input lists!")
		exit()
	
	fig = figure()
	sub = fig.add_subplot(111)
	
	for i in range(n_curves):
		label = "Hidden nodes#: " + str(hidden_list[i])
		sub.plot(np.arange(len(error_list[i])),error_list[i],lw=2.0,color=color[i],label=label)
	sub.set_xlabel("Epoch")
	sub.set_ylabel("Error")
	
	if (logx):
		sub.set_xscale("log")
	if (logy):
		sub.set_yscale("log")
	
	plt.legend(loc='upper right',fontsize=10)
	plt.show()
	plt.show(block = False)
	print("Press enter to continue...")
	raw_input()
	return

# Initialize deltas for batch optimization
def initDeltas(network):
	network.deltas = {'i->h': np.zeros((network.n_input+1,network.n_hidden),dtype=float),
		'h->o': np.zeros((network.n_hidden+1,network.n_output),dtype=float)}
	return

# Update deltas for batch optimization
def updateDeltas(network):
	network.deltas['i->h'] = network.deltas['i->h'] + network.dW['i->h']
	network.deltas['h->o'] = network.deltas['h->o'] + network.dW['h->o']
	return

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
	neural_net = network(n_input = n_input, n_output = n_output, 
		n_hidden = n_hidden, lrate = lrate, opt = opt, mom = mom)
	
	# Arrays used later for vectorized total error calculation
	error = []
	bias = np.ones((x.shape[0],1),dtype=float) # bias column vector
	in_des_mtx = np.hstack((x,bias)) # input design matrix
	# Initiate iterations to train neural network
	convergence = False
	it = 0
	while (not convergence):
		error.append(0.0)
		# Train the network on each data samples (SDG/online training)
		if (opt == 'SGD'):
			# Reshuffle data at each epoch
			if (reshuffle):
				x,y = libd.shuffle(x,y)
				in_des_mtx = np.hstack((x,bias)) # input design matrix
			for m in range(n_training_samples):
				# Get data sample in form of row vector
				xx = x.loc[m].values
				yy = y.loc[m].values
				# Forward propagate
				forwardPropagation(neural_net,xx)
				# Backward propagate error
				backwardPropagation(neural_net,xx,yy)
				# Update weights
				updateWeights(neural_net)
			# Get error for the iteration
			out1 = activate(np.dot(in_des_mtx,neural_net.W['i->h']))
			layer_des_mtx = np.hstack((out1,bias))
			out2 = activate(np.dot(layer_des_mtx,neural_net.W['h->o']))
			derr = out2 - y
			error[it] = np.sqrt( sum(derr.values * derr.values) / (1.0 * y.shape[0]))
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
				initDeltas(neural_net)
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
