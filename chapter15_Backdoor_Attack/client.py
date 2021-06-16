
import models, torch, copy
import numpy as np
import matplotlib.pyplot as plt

class Client(object):

	def __init__(self, conf, model, train_dataset, id = -1):
		
		self.conf = conf
		
		self.local_model = models.get_model(self.conf["model_name"]) 
		
		self.client_id = id
		
		self.train_dataset = train_dataset
		
		all_range = list(range(len(self.train_dataset)))
		data_len = int(len(self.train_dataset) / self.conf['no_models'])
		train_indices = all_range[id * data_len: (id + 1) * data_len]

		self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"], 
									sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices))
		
			
		
	def local_train(self, model):

		for name, param in model.state_dict().items():
			self.local_model.state_dict()[name].copy_(param.clone())
	
	
		optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
									momentum=self.conf['momentum'])
		
		self.local_model.train()
		for e in range(self.conf["local_epochs"]):
			
			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				
				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
			
				optimizer.zero_grad()
				output = self.local_model(data)
				loss = torch.nn.functional.cross_entropy(output, target)
				loss.backward()
			
				optimizer.step()
			print("Epoch %d done." % e)	
		diff = dict()
		for name, data in self.local_model.state_dict().items():
			diff[name] = (data - model.state_dict()[name])
			
		return diff
		
	def local_train_malicious(self, model):

		for name, param in model.state_dict().items():
			self.local_model.state_dict()[name].copy_(param.clone())
	
		optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
									momentum=self.conf['momentum'])
		pos = []
		for i in range(2, 28):
			pos.append([i, 3])
			pos.append([i, 4])
			pos.append([i, 5])
			
		self.local_model.train()
		for e in range(self.conf["local_epochs"]):
			
			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				
				for k in range(self.conf["poisoning_per_batch"]):
					img = data[k].numpy()
					for i in range(0,len(pos)):
						img[0][pos[i][0]][pos[i][1]] = 1.0
						img[1][pos[i][0]][pos[i][1]] = 0
						img[2][pos[i][0]][pos[i][1]] = 0
					
					target[k] = self.conf['poison_label']
				#for k in range(32):
				#		img = data[k].numpy()
				#		
				#		img = np.transpose(img, (1, 2, 0))
				#		plt.imshow(img)
				#		plt.show()
				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
			
				optimizer.zero_grad()
				output = self.local_model(data)
				
				class_loss = torch.nn.functional.cross_entropy(output, target)
				dist_loss = models.model_norm(self.local_model, model)
				loss = self.conf["alpha"]*class_loss + (1-self.conf["alpha"])*dist_loss
				loss.backward()
			
				optimizer.step()
			print("Epoch %d done." % e)
			
		diff = dict()
		for name, data in self.local_model.state_dict().items():
			diff[name] = self.conf["eta"]*(data - model.state_dict()[name])+model.state_dict()[name]
			
		return diff		
		