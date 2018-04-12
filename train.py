import argparse
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader

from vgg16 import VGG16
from transformation_net import Transformation_Network

def main():
	parser = argparse.ArgumentParser()
	sub_parsers = parser.add_subparsers(dest='subcommand')
	
	train_parser = sub_parsers.add_parser('train')
	train_parser.add_argument('--dataset', type=str, required=True)
	train_parser.add_argument('--style_fname', type=str, required=True)
	train_parser.add_argument('--epoch', type=int, default=2)
	train_parser.add_argument('--lr', type=float, default=1e-3)
	train_parser.add_argument('--batch_size', type=int, default=4)

	args = parser.parse_args()

	if args.subcommand == 'train':
		train(args)

def train(args):
	use_cuda = torch.cuda.is_available()
	dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

	transform = transforms.Compose([
		transforms.Resize(size=(256,256)),
		transforms.ToTensor(),
		normalize
		])

	train_dataset = datasets.ImageFolder(root=args.dataset, transform=transform)
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

	vgg16 = VGG16()

	transnet = Transformation_Network()
	optimizer = optim.Adam(transnet.parameters(), lr=args.lr)
	mse_loss = nn.MSELoss()

	if use_cuda:
		vgg16 = vgg16.cuda()
		transnet = transnet.cuda()

	style_target = Image.open(args.style_fname)
	style_target = transform(style_target)
	style_target = style_target.unsqueeze(0)
	style_target = Variable(style_target)

	features_style_target = vgg16(style_target)
	gram_style_target = [gram_matrix(feature) for feature in features_style_target]


	for epoch in range(args.epoch):
		running_style_loss = 0.0
		running_content_loss = 0.0

		transnet.train()
		for i, data in enumerate(train_loader):
			content_target, _ = data

			content_target = Variable(content_target)
			y = transnet(content_target)

			features_y = vgg16(y)
			features_content_target = vgg16(content_target)

			content_loss = mse_loss(features_y.relu3_3, features_content_target.relu3_3)
			N,C,H,W = features_y.relu3_3.size()
			content_loss /= C*H*W

			gram_y = [gram_matrix(feature) for feature in features_y]
			style_loss = np.sum([mse_loss(i, j) for i, j in zip(gram_y, gram_style_target)])

			optimizer.zero_grad()
			loss = content_loss + style_loss
			loss.backward()
			optimizer.step()

			running_content_loss += content_loss.data[0]
			running_style_loss += style_loss.data[0]

			if i % 500 == 499:
				print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 500))
				running_content_loss = 0.0
				running_style_loss = 0.0

def gram_matrix(input):
	N, C, H, W = input.size()
	input = input.view(N, C, H*W)
	return torch.dot(input, torch.transpose(input,1,2)) / (C*H*W)


if __name__ == '__main__':
	main()
