import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50

class Resnet18(nn.Module):
	def __init__(self, args):
		super(Resnet18, self).__init__()
		self.args = args
		self.image_embedding = resnet18(pretrained=True)
		self.needs_y = False

		self.image_embedding.fc = nn.Linear(512, 182)
		nn.init.xavier_uniform_(self.image_embedding.fc.weight.data)
		
	def forward(self, x):
		emb_h = self.image_embedding(x)
		
		return emb_h

class Resnet50(nn.Module):
	def __init__(self, args):
		super(Resnet50, self).__init__()
		self.args = args
		self.image_embedding = resnet50(pretrained=True)
		self.needs_y = False

		self.image_embedding.fc = nn.Linear(2048, 182)
		nn.init.xavier_uniform_(self.image_embedding.fc.weight.data)
		
	def forward(self, x):
		emb_h = self.image_embedding(x)
		
		return emb_h