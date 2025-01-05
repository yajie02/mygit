#导入包和设置环境
import snntorch as snn
import torch
from torchvision import datasets, transforms
from snntorch import utils #包含一些用于修改数据集的有用函数
from torch.utils.data import DataLoader
from snntorch import spikegen

import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML

# Training Parameters
batch_size=128
data_path='./data/mnist'
num_classes = 10  # MNIST has 10 output classes

# Torch Variables
dtype = torch.float

# Define a transform
transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

#下载数据集
mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)

#在我们真正开始训练网络之前，我们不需要大型数据集
subset = 10
mnist_train = utils.data_subset(mnist_train, subset) #应用data_subset以子集中定义的因子减少数据集

#创建数据加载器
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

# # Temporal Dynamics
# num_steps = 100
#
# # create vector filled with 0.5
# raw_vector = torch.ones(num_steps)*0.5
#
# # pass each sample through a Bernoulli trial
# rate_coded_vector = torch.bernoulli(raw_vector)
# print(f"Converted vector: {rate_coded_vector}")
# print(f"The output is spiking {rate_coded_vector.sum()*100/len(rate_coded_vector):.2f}% of the time.")

num_steps = 100
# Iterate through minibatches
data = iter(train_loader)
data_it, targets_it = next(data)

# Spiking Data
spike_data = spikegen.rate(data_it, num_steps=num_steps)

print(spike_data.size())
spike_data_sample = spike_data[:, 0, 0]
print(spike_data_sample.size())

spike_data = spikegen.rate(data_it, num_steps=num_steps, gain=0.25)
spike_data_sample2 = spike_data[:, 0, 0]
print(f"The corresponding target is: {targets_it[0]}")

# plt.figure(facecolor="w")
# plt.subplot(1,2,1)
# plt.imshow(spike_data_sample.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')
# plt.axis('off')
# plt.title('Gain = 1')
#
# plt.subplot(1,2,2)
# plt.imshow(spike_data_sample2.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')
# plt.axis('off')
# plt.title('Gain = 0.25')
#
# plt.show()

# Reshape
# spike_data_sample2 = spike_data_sample2.reshape((num_steps, -1))
#
# # raster plot
# fig = plt.figure(facecolor="w", figsize=(10, 5))
# ax = fig.add_subplot(111)
# splt.raster(spike_data_sample2, ax, s=1.5, c="black")
#
# plt.title("Input Layer")
# plt.xlabel("Time step")
# plt.ylabel("Neuron Number")
# plt.show()
#
# idx = 210  # index into 210th neuron
#
# fig = plt.figure(facecolor="w", figsize=(8, 1))
# ax = fig.add_subplot(111)
#
# splt.raster(spike_data_sample2.reshape(num_steps, -1)[:, idx].unsqueeze(1), ax, s=100, c="black", marker="|")
#
# plt.title("Input Neuron")
# plt.xlabel("Time step")
# plt.yticks([])
# plt.show()

# def convert_to_time(data, tau=5, threshold=0.01):
#   spike_time = tau * torch.log(data / (data - threshold))
#   return spike_time
#
# raw_input = torch.arange(0, 5, 0.05) # tensor from 0 to 5
# spike_times = convert_to_time(raw_input)
#
# plt.plot(raw_input, spike_times)
# plt.xlabel('Input Value')
# plt.ylabel('Spike Time (s)')
# plt.show()

# spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01)
#
# fig = plt.figure(facecolor="w", figsize=(10, 5))
# ax = fig.add_subplot(111)
# splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black")
#
# plt.title("Input Layer")
# plt.xlabel("Time step")
# plt.ylabel("Neuron Number")
# plt.show()

# spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01, linear=True)
#
# fig = plt.figure(facecolor="w", figsize=(10, 5))
# ax = fig.add_subplot(111)
# splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black")
# plt.title("Input Layer")
# plt.xlabel("Time step")
# plt.ylabel("Neuron Number")
# plt.show()

# spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01,
#                               normalize=True, linear=True)
#
# fig = plt.figure(facecolor="w", figsize=(10, 5))
# ax = fig.add_subplot(111)
# splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black")
#
# plt.title("Input Layer")
# plt.xlabel("Time step")
# plt.ylabel("Neuron Number")
# plt.show()

# spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01,
#                               clip=True, normalize=True, linear=True)
#
# fig = plt.figure(facecolor="w", figsize=(10, 5))
# ax = fig.add_subplot(111)
# splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black")
#
# plt.title("Input Layer")
# plt.xlabel("Time step")
# plt.ylabel("Neuron Number")
# plt.show()

# Create a tensor with some fake time-series data
# data = torch.Tensor([0, 1, 0, 2, 8, -20, 20, -5, 0, 1, 0])
#
# # Plot the tensor
# plt.plot(data)
#
# plt.title("Some fake time-series data")
# plt.xlabel("Time step")
# plt.ylabel("Voltage (mV)")
# plt.show()
#
# # Convert data
# spike_data = spikegen.delta(data, threshold=4)
#
# # Create fig, ax
# fig = plt.figure(facecolor="w", figsize=(8, 1))
# ax = fig.add_subplot(111)
#
# # Raster plot of delta converted data
# splt.raster(spike_data, ax, c="black")
#
# plt.title("Input Neuron")
# plt.xlabel("Time step")
# plt.yticks([])
# plt.xlim(0, len(data))
# plt.show()
#
# print(spike_data)

# Create a random spike train
spike_prob = torch.rand((num_steps, 28, 28), dtype=dtype) * 0.5
spike_rand = spikegen.rate_conv(spike_prob)

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_rand[:, 0].view(num_steps, -1), ax, s=25, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()