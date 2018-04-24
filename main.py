import argparse
import sys
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from scipy.misc import imsave
from matplotlib import pyplot as plt
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
	train_parser.add_argument('--ckpt_dir', type=str, default='./checkpoint')
	train_parser.add_argument('--img_dir', type=str, default='./images/train')
	train_parser.add_argument('--content_weight', type=int, default=1e5)
	train_parser.add_argument('--style_weight', type=int, default=1e10)
	train_parser.add_argument('--tv_weight', type=int, default=10)
	

	test_parser = sub_parsers.add_parser('test')
	test_parser.add_argument('--content_fpath', type=str, required=True)
	test_parser.add_argument('--ckpt_fpath', type=str, required=True)
	test_parser.add_argument('--img_dir', type=str, default='./images/test')

	args = parser.parse_args()


	if args.subcommand == 'train':
		train(args)
	if args.subcommand == 'test':
		test(args)

def train(args):
    use_cuda = torch.cuda.is_available()
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize(size=(256,256)),
        transforms.ToTensor(),
        normalize,
        ])

    train_dataset = datasets.ImageFolder(root=args.dataset, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    vgg16 = VGG16()

    transnet = Transformation_Network()
    optimizer = optim.Adam(transnet.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()

    style_target = Image.open(args.style_fname)
    style_target = transforms.ToTensor()(style_target)
    style_target = normalize(style_target)
    style_target = style_target.repeat(args.batch_size, 1, 1, 1)
    style_target = Variable(style_target)

    if use_cuda:
        vgg16 = vgg16.cuda()
        transnet = transnet.cuda()
        style_target = style_target.cuda()

    features_style_target = vgg16(style_target)
    gram_style_target = [gram_matrix(feature) for feature in features_style_target]


    for epoch in range(args.epoch):
        print('Epoch {}'.format(epoch+1))
        running_style_loss = 0.0
        running_content_loss = 0.0

        transnet.train()
        for i, data in enumerate(train_loader):
            content_target, _ = data

            content_target = Variable(content_target)
            if use_cuda:
                content_target = content_target.cuda()

            y = transnet(content_target)

            features_y = vgg16(y)
            features_content_target = vgg16(content_target)

            content_loss = args.content_weight * mse_loss(features_y.relu3_3, features_content_target.relu3_3)
            N,C,H,W = features_y.relu3_3.size()

            gram_y = [gram_matrix(feature) for feature in features_y]
            style_loss = args.style_weight * np.sum([mse_loss(i, j) for i, j in zip(gram_y, gram_style_target)])
            
            tv_loss = compute_tv_loss(y, args.tv_weight)

            optimizer.zero_grad()
            loss = content_loss + style_loss + tv_loss
            loss.backward()
            optimizer.step()

            running_content_loss += content_loss.data[0]
            running_style_loss += style_loss.data[0]
            
            if i % 100 == 0:
                running_loss = running_content_loss + running_style_loss
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                print('[%d, %5d] content loss: %.3f' % (epoch + 1, i + 1, running_content_loss / 100))
                print('[%d, %5d] style loss: %.3f' % (epoch + 1, i + 1, running_style_loss / 100))
                running_content_loss = 0.0
                running_style_loss = 0.0
                
                if i % 1000 == 0:
	                content_image = Image.open(content_fpath)
	                content_image = transform(content_image)
	                content_image = content_image.unsqueeze(0)
	                content_image = Variable(content_image)
	                content_image = content_image.cuda()
	                
	                out = transnet(content_image)
	                out_image = out.data.cpu()
	                out_image = deprocess(out_image)
	                img_fname = 'epoch_' + str(epoch+1) + '_' + 'iter_' + str(i) + str(time.ctime()).replace(' ', '_') + '.jpg' 
	                img_fpath = os.path.join(args.img_dir, img_fname)
	                out_image.save(img_fpath)
	        
            if i % 2000 == 0:              
                ckpt_fname = "epoch_" + str(epoch+1) + "_" + 'iter_' + str(i) + '_' + str(time.ctime()).replace(' ', '_')  + ".model"
                ckpt_path = os.path.join(args.ckpt_dir, ckpt_fname)
                torch.save(transnet.state_dict(), ckpt_path)

def test(args):
	use_cuda = torch.cuda.is_available()
	dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
	
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

	transform = transforms.Compose([
		transforms.Resize(size=(256,256)),
		transforms.ToTensor(),
		normalize
		])

	content_image = Image.open(args.content_fpath)
	content_image = transform(content_image)
	content_image = content_image.unsqueeze(0)
	content_image = Variable(content_image)

	transnet = Transformation_Network()

	if use_cuda:
		content_image = content_image.cuda()
		transnet = transnet.cuda()
		transnet.load_state_dict(torch.load(args.ckpt_fpath))

	transnet.load_state_dict(torch.load(args.ckpt_fpath, map_location='cpu'))

	out = transnet(content_image)
	out_image = out.data.cpu()
	out_image = deprocess(out)

	img_fname = content_fpath.split('/')[-1].split('.')[0] + '_style' + '.jpg' 
	img_fpath = os.path.join(args.img_dir, img_fname)
	out_image.save(img_fpath)


def deprocess(img):
    transform = transforms.Compose([
        transforms.Lambda(lambda x : x[0]),
    	transforms.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in (0.229, 0.224, 0.225)]),
    	transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
    	transforms.Lambda(rescale),
    	transforms.ToPILImage(),
    	])
    return transform(img)

def gram_matrix(input):
    N, C, H, W = input.size()
    input = input.view(N, C, H*W)
    return torch.bmm(input, torch.transpose(input, 1, 2)) / (C*H*W)

def compute_tv_loss(img, tv_weight):
    img1, img2, img3, img4 = img[:,:,1:,:], img[:,:,:-1,:], img[:,:,:,1:], img[:,:,:,:-1]
    diff1, diff2 = (img1-img2)**2, (img3-img4) **2
    diff = torch.sum(diff1) + torch.sum(diff2)
    return tv_weight * diff 

if __name__ == '__main__':
	main()
