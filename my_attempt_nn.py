import numpy as np
import pandas as pd
import sys

# create class
class NeuralNetwork(object):

	# define init function
	def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
		# set number of nodes in input, hidden and output nodes
		self.input_nodes = input_nodes
		self.hidden_nodes = hidden_nodes
		self.output_nodes = output_nodes

		# Initialize weights
		# weight from input to hidden layer should be in normalized range around median 0.0

		self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes ** - 0.5, (self.hidden_nodes, self.input_nodes))

		# weight from hidden to output layer
		self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes ** - 0.5, (self.output_nodes, self.hidden_nodes))

		self.learning_rate = learning_rate

		# set the Activation function
		self.activation_function = lambda x: 1 / (1 + np.exp(-x))

		pass
	
	# define train
	def train(self, inputs_list, targets_list):

		# For each row, convert inputs list of 56 columns into 2darray 
		# then transpose them, so all input columns will become 56 rows with 1 column of input only as input node
		inputs = np.array(inputs_list, ndmin=2).T
		targets = np.array(targets_list, ndmin=2).T

		print ("inputs_list: ", inputs_list)
		print ("inputs_list.shape: ", inputs_list.shape)
		print ("inputs: ", inputs)
		print ("inputs.shape: ", inputs.shape)
		# print ("targets_list.shape: ", targets_list.shape)
		# print ("targets.shape: ", targets.shape)

		# print ("inputs_list.type: ", inputs_list.type)
		# print ("targets_list.type: ", targets_list.type)
		# print ("inputs.type: ", inputs.type)
		# print ("targets.type: ", targets.type)

		# forward pass

		pass

	# define run function


# create main function
def pankax():
	# get data
	# lets get the bike sharing hourly data and stored in pandas
	data_path = 'Bike-Sharing-Dataset/hour.csv'
	rides = pd.read_csv(data_path)
	# lets have a glance how it look
	# print (rides.head())

	# Data Munging including dummy, chossing right features, normalizaition, or scaling
	# creagting dummies for binary variables
	dummy_fields = ['season', 'mnth', 'hr', 'weathersit', 'weekday']
	for dummy in dummy_fields:
		dummies = pd.get_dummies(rides[dummy], prefix = dummy, drop_first=False)
		rides = pd.concat([rides, dummies], axis = 1)
	# dropping dummy_fields and other non important features
	dropping_fields = ['instant', 'dteday','atemp','workingday','season', 'mnth', 'hr', 'weathersit', 'weekday' ]
	data = rides.drop(dropping_fields, axis = 1)
	# print (data.head())

	# scale each of continous variable, by sccaling each of variable with zero mean and std of 1
	quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']

	# Store scalings in a dictionary so we can convert back later
	scaled_features = {}
	for each in quant_features:
	    mean, std = data[each].mean(), data[each].std()
	    scaled_features[each] = [mean, std]
	    data.loc[:, each] = (data[each] - mean)/std

	# print (data.head())
	# split data into training, validation and test data
	# Save the last 21 days 
	test_data = data[-21*24:]
	data = data[:-21*24]

	# Separate the data into features and targets
	target_fields = ['cnt', 'casual', 'registered']
	features, targets = data.drop(target_fields, axis=1), data[target_fields]
	test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

	# Hold out the last 60 days of the remaining data as a validation set
	train_features, train_targets = features[:-60*24], targets[:-60*24]
	val_features, val_targets = features[-60*24:], targets[-60*24:]

	# print (train_features.head())
	# print (18**-0.5)

	### Set the hyperparameters here ###
	epochs = 1
	learning_rate = 0.1
	hidden_nodes = 18
	output_nodes = 1

	# set input features length as input nodes length
	input_nodes = train_features.shape[1]
	print ("train_features.shape[1]", train_features.shape[1])

	# get nn instance
	network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
	# train training data on nn
	losses = {'train':[], 'validation':[]}
	for e in range(epochs):
	    # Go through a random batch of 128 records from the training data set
	    batch = np.random.choice(train_features.index, size=1)
	    for record, target in zip(train_features.ix[batch].values, train_targets.ix[batch]['cnt']):
	    	network.train(record, target)
    
	    # # Printing out the training progress
	    # train_loss = MSE(network.run(train_features), train_targets['cnt'].values)
	    # val_loss = MSE(network.run(val_features), val_targets['cnt'].values)
	    # sys.stdout.write("\rProgress: " + str(100 * e/float(epochs))[:4] \
	    #                  + "% ... Training loss: " + str(train_loss)[:5] \
	    #                  + " ... Validation loss: " + str(val_loss)[:5])
	    
	    # losses['train'].append(train_loss)
	    # losses['validation'].append(val_loss)

	# query test data on nn
	# show accurracy by comparing test data actual values == query test data results
	pass



# call for main function
if __name__ == "__main__":
		pankax()
		
			