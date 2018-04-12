import torch
import torch.nn as nn

class Transformation_Network(nn.Module):
	
	def __init__(self):
		super(Transformation_Network, self).__init__()

		"""
		We eschew pooling layers, instead using strided and 2e for in-network downsampling and upsampling.

		Input 3 × 256 × 256
		Reflection Padding (40 × 40) 3 × 336 × 336
		32 × 9 × 9 conv, stride 1 32 × 336 × 336
		64 × 3 × 3 conv, stride 2 64 × 168 × 168
		128 × 3 × 3 conv, stride 2 128 × 84 × 84
		Residual block, 128 filters 128 × 80 × 80
		Residual block, 128 filters 128 × 76 × 76
		Residual block, 128 filters 128 × 72 × 72
		Residual block, 128 filters 128 × 68 × 68
		Residual block, 128 filters 128 × 64 × 64
		64 × 3 × 3 conv, stride 1/2 64 × 128 × 128
		32 × 3 × 3 conv, stride 1/2 32 × 256 × 256
		3 × 9 × 9 conv, stride 1 3 × 256 × 256
		"""

		self.conv1 = nn.Conv2d(3,32,9, padding=4)
		self.conv2 = nn.Conv2d(32,64,3,2, padding=1)
		self.conv3 = nn.Conv2d(64,128,3,2, padding=1)

		self.in1 = nn.InstanceNorm2d(32)
		self.in2 = nn.InstanceNorm2d(64)
		self.in3 = nn.InstanceNorm2d(128)

		self.res1 = ResidualBlock(128)
		self.res2 = ResidualBlock(128)
		self.res3 = ResidualBlock(128)
		self.res4 = ResidualBlock(128)
		self.res5 = ResidualBlock(128)

		self.conv4 = nn.ConvTranspose2d(128,64,3,2, padding=1, output_padding=1)
		self.conv5 = nn.ConvTranspose2d(64,32,3,2, padding=1, output_padding=1)
		self.conv6 = nn.Conv2d(32,3,9, padding=4)

		self.in4 = nn.InstanceNorm2d(64, affine=True)
		self.in5 = nn.InstanceNorm2d(32, affine=True)

		self.reflection_pad = nn.ReflectionPad2d(40)

		self.relu = nn.ReLU()

	def forward(self, x):
		padded_x = self.reflection_pad(x)
		out_conv1 = self.relu(self.in1(self.conv1(padded_x)))
		out_conv2 = self.relu(self.in2(self.conv2(out_conv1)))
		out_conv3 = self.relu(self.in3(self.conv3(out_conv2)))

		out_res = self.res1(out_conv3)
		out_res = self.res2(out_res)
		out_res = self.res3(out_res)
		out_res = self.res4(out_res)
		out_res = self.res5(out_res)

		out_conv4 = self.relu(self.in4(self.conv4(out_res)))
		out_conv5 = self.relu(self.in5(self.conv5(out_conv4)))
		out = nn.Tanh()(self.conv6(out_conv5))

		return out

class ResidualBlock(nn.Module): 
	def __init__(self, filters=128):
		super(ResidualBlock, self).__init__()

		self.conv = nn.Conv2d(filters,filters,3)
		self.IN = nn.InstanceNorm2d(filters, affine=True)
		self.relu = nn.ReLU()

	def center_crop(self, x):
		N,C,H,W = x.size()
		return x[:,:,2:H-2, 2:W-2]

	def forward(self, x):
		out = self.relu(self.IN(self.conv(x)))
		out = self.IN(self.conv(out)) + self.center_crop(x)
		return out




