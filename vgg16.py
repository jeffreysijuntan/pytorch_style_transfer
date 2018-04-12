from collections import namedtuple

import torch
import torch.nn as nn
import torchvision.models as models

class VGG16(nn.Module):

	def __init__(self, requires_grad=False):
		super(VGG16, self).__init__()
		self.slice1 = nn.Sequential()
		self.slice2 = nn.Sequential()
		self.slice3 = nn.Sequential()
		self.slice4 = nn.Sequential()

		vgg16_pretrained_features = models.vgg16(pretrained=True).features
		for i in range(4):
			self.slice1.add_module(str(i), vgg16_pretrained_features[i])
		for i in range(4,9):
			self.slice2.add_module(str(i), vgg16_pretrained_features[i])
		for i in range(9,16):
			self.slice3.add_module(str(i), vgg16_pretrained_features[i])
		for i in range(16,23):
			self.slice4.add_module(str(i), vgg16_pretrained_features[i])

		if not requires_grad:
			for param in self.parameters():
				param.requires_grad = False

	def forward(self, x):
		relu1_2 = self.slice1(x)
		relu2_2 = self.slice2(relu1_2)
		relu3_3 = self.slice3(relu2_2)
		relu4_3 = self.slice4(relu3_3)

		vggout = namedtuple("Vggout", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
		return vggout(relu1_2, relu2_2, relu3_3, relu4_3)
