import argparse, json
import datetime
import os
import numpy as np
import logging
import torch, random

from server import *
from client import *
import models
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures

def read_dataset():
	data_X, data_Y = [], []
	
	with open("./data/breast.csv") as fin:
		for line in fin:
			data = line.split(',')
			data_X.append([float(e) for e in data[:-1]])
			if int(data[-1])==1:
				data_Y.append(1)
			else:
				data_Y.append(-1)
	
	data_X = np.array(data_X)
	data_Y = np.array(data_Y)
	print("one_num: ", np.sum(data_Y==1), ", minus_one_num: ", np.sum(data_Y==-1))
	

	
	idx = np.arange(data_X.shape[0])
	np.random.shuffle(idx)
	
	train_size = int(data_X.shape[0]*0.8)
	
	train_x = data_X[idx[:train_size]]
	train_y = data_Y[idx[:train_size]]
	
	eval_x = data_X[idx[train_size:]]
	eval_y = data_Y[idx[train_size:]]
	
	return (train_x, train_y), (eval_x, eval_y)
	

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('-c', '--conf', dest='conf')
	args = parser.parse_args()
	

	with open(args.conf, 'r') as f:
		conf = json.load(f)	
	
	
	train_datasets, eval_datasets = read_dataset()
	
	print(train_datasets[0].shape, train_datasets[1].shape)
	
	print(eval_datasets[0].shape, eval_datasets[1].shape)
	

	server = Server(conf, eval_datasets)
	clients = []
	

	train_size = train_datasets[0].shape[0]
	per_client_size = int(train_size/conf["no_models"])
	for c in range(conf["no_models"]):
		clients.append(Client(conf, Server.public_key, server.global_model.encrypt_weights, train_datasets[0][c*per_client_size: (c+1)*per_client_size], train_datasets[1][c*per_client_size: (c+1)*per_client_size]))
		

	#print(server.global_model.weights)
	
	for e in range(conf["global_epochs"]):
		
		server.global_model.encrypt_weights = models.encrypt_vector(Server.public_key, models.decrypt_vector(Server.private_key, server.global_model.encrypt_weights))
			
		candidates = random.sample(clients, conf["k"])
		
		weight_accumulator = [Server.public_key.encrypt(0.0)] * (conf["feature_num"]+1)
		
		
		for c in candidates:	
			#print(models.decrypt_vector(Server.private_key, server.global_model.encrypt_weights))
			diff = c.local_train(server.global_model.encrypt_weights)

			for i in range(len(weight_accumulator)):
				weight_accumulator[i] = weight_accumulator[i] + diff[i]
			
		server.model_aggregate(weight_accumulator)
		
		
		
		acc = server.model_eval()
			
		print("Epoch %d, acc: %f\n" % (e, acc))	
