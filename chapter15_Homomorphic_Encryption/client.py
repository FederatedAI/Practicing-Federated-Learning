
import models, torch, copy
import numpy as np
from server import Server

class Client(object):

	def __init__(self, conf, public_key, weights, data_x, data_y):
		
		self.conf = conf
		
		self.public_key = public_key
		
		self.local_model = models.LR_Model(public_key=self.public_key, w=weights, encrypted=True)

		#print(type(self.local_model.encrypt_weights))
		self.data_x = data_x
		
		self.data_y = data_y
		
		#print(self.data_x.shape, self.data_y.shape)
									
		
	def local_train(self, weights):
		
		
		original_w = weights

		self.local_model.set_encrypt_weights(weights)

		
		
		neg_one = self.public_key.encrypt(-1)
		
		for e in range(self.conf["local_epochs"]):
			print("start epoch ", e)
			#if e > 0 and e%2 == 0:
			#	print("re encrypt")
			#	self.local_model.encrypt_weights = Server.re_encrypt(self.local_model.encrypt_weights)
				
			idx = np.arange(self.data_x.shape[0])
			batch_idx = np.random.choice(idx, self.conf['batch_size'], replace=False)
			#print(batch_idx)
			
			x = self.data_x[batch_idx]
			x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
			y = self.data_y[batch_idx].reshape((-1, 1))
			
			#print((0.25 * x.dot(self.local_model.encrypt_weights) + 0.5 * y.transpose() * neg_one).shape)
			
			#print(x.transpose().shape)
			
			#assert(False)

			batch_encrypted_grad = x.transpose() * (0.25 * x.dot(self.local_model.encrypt_weights) + 0.5 * y.transpose() * neg_one)
			encrypted_grad = batch_encrypted_grad.sum(axis=1) / y.shape[0]
			
			for j in range(len(self.local_model.encrypt_weights)):
				self.local_model.encrypt_weights[j] -= self.conf["lr"] * encrypted_grad[j]

		weight_accumulators = []
		#print(models.decrypt_vector(Server.private_key, weights))
		for j in range(len(self.local_model.encrypt_weights)):
			weight_accumulators.append(self.local_model.encrypt_weights[j] - original_w[j])
		
		return weight_accumulators

