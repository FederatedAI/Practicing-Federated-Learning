
import torch 
from torchvision import models

import numpy as np


def encrypt_vector(public_key, x):
	return [public_key.encrypt(i) for i in x]
	
def encrypt_matrix(public_key, x):
	ret = []
	for r in x:
		ret.append(encrypt_vector(public_key, r))
	return ret 
	
		
def decrypt_vector(private_key, x):
	return [private_key.decrypt(i) for i in x]
	

def decrypt_matrix(private_key, x):
	ret = []
	for r in x:
		ret.append(decrypt_vector(private_key, r))
	return ret 


		
class LR_Model(object):

	def __init__ (self, public_key, w_size=None, w=None, encrypted=False):
		"""
		w_size: 权重参数数量
		w: 是否直接传递已有权重，w和w_size只需要传递一个即可
		encrypted: 是明文还是加密的形式
		"""
		self.public_key = public_key
		if w is not None:
			self.weights = w
		else:
			limit = -1.0/w_size 
			self.weights = np.random.uniform(-0.5, 0.5, (w_size,))
		
		if encrypted==False:
			self.encrypt_weights = encrypt_vector(public_key, self.weights)
		else:
			self.encrypt_weights = self.weights	
			
	def set_encrypt_weights(self, w):
		for id, e in enumerate(w):
			self.encrypt_weights[id] = e 
		
	def set_raw_weights(self, w):
		for id, e in enumerate(w):
			self.weights[id] = e 
			