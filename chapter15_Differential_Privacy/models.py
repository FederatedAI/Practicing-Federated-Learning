
import torch 
from torchvision import models
import math

def get_model(name="vgg16", pretrained=True):
	if name == "resnet18":
		model = models.resnet18(pretrained=pretrained)
	elif name == "resnet50":
		model = models.resnet50(pretrained=pretrained)	
	elif name == "densenet121":
		model = models.densenet121(pretrained=pretrained)		
	elif name == "alexnet":
		model = models.alexnet(pretrained=pretrained)
	elif name == "vgg16":
		model = models.vgg16(pretrained=pretrained)
	elif name == "vgg19":
		model = models.vgg19(pretrained=pretrained)
	elif name == "inception_v3":
		model = models.inception_v3(pretrained=pretrained)
	elif name == "googlenet":		
		model = models.googlenet(pretrained=pretrained)
		
	if torch.cuda.is_available():
		return model.cuda()
	else:
		return model 
		

def model_norm(model_1, model_2):
    params_1 = torch.cat([param.view(-1) for param in model_1.parameters()])
    params_2 = torch.cat([param.view(-1) for param in model_2.parameters()])
    
    return torch.norm(params_1 - params_2, p = 2)

def quick_model_norm(model_1, model_2):
    diffs = [(p1 - p2).view(-1) for p1, p2 in zip(model_1.parameters(), model_2.parameters())]
    return torch.norm(torch.cat(diffs), p = 2)