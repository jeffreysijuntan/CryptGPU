import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import crypten
import crypten.mpc as mpc
import crypten.communicator as comm

from crypten.mpc import MPCTensor
from crypten.mpc.ptype import ptype as Ptype

import torchvision
import torchvision.models as models
import torch.autograd.profiler as profiler

import logging
import time
import timeit
import argparse

from network import *

def inference(model, input_size, batch_size=1, device="cuda"):
    comm.get().set_verbosity(True)

    bs = batch_size
    c, w, h = input_size
    x = crypten.cryptensor(torch.rand((bs, c, w, h)), device=device, requires_grad=False)

    model = crypten.nn.from_pytorch(model, dummy_input=torch.empty(bs, c, w, h))
    model = model.encrypt()
    model = model.to(device) 

    model.eval()
    model.replicate_parameters()

    total_time = 0
    comm_time = 0
    conv_time, pool_time, relu_time, matmul_time = 0, 0, 0, 0
    for i in range(6):
        comm.get().reset_communication_stats()
        
        tic = time.perf_counter()
        model(x)
        toc = time.perf_counter()

        if i != 0:
            total_time += toc - tic
            comm_time += comm.get().comm_time
            conv_time += comm.get().time_conv
            relu_time += comm.get().time_relu
            pool_time += comm.get().time_pool
            matmul_time += comm.get().time_matmul

            # if comm.get().get_rank() == 0:
            #     print(f"Iteration {i} runtime: {toc - tic}")

        comm.get().print_total_communication()

    if comm.get().get_rank() == 0:
        print("----------- Statistics ----------------")
        print(f"Total Communication: {comm.get().total_comm_bytes}")
        print(f"Avg Runtime: {total_time / 5}")
        print(f"Avg Comm: {comm_time / 5}")
        print(f"Avg Linear: {conv_time + matmul_time/ 5}")
        print(f"Avg ReLU: {relu_time / 5}")
        print(f"Avg Pool: {pool_time / 5}")


def training(model, input_size, batch_size, num_classes, device="cuda"):
    comm.get().set_verbosity(True)
    rank = comm.get().get_rank()

    c, h, w = input_size
    bs = batch_size

    criterion = crypten.nn.CrossEntropyLoss()
    model = crypten.nn.from_pytorch(model, dummy_input=torch.empty(bs, c, h, w))
    model = model.to(device)
    model.encrypt() 
    model.train()

    labels =  torch.ones(bs, requires_grad=False).long().to(device)
    labels = F.one_hot(labels, num_classes=num_classes)
    labels = crypten.cryptensor(labels, src=0)

    input = torch.randn([bs,c,w,h], requires_grad=False)
    input = crypten.cryptensor(input, src=0).to(device)

    total_time = 0
    comm_time = 0
    conv_time, pool_time, relu_time, matmul_time, softmax_time = 0, 0, 0, 0, 0
    for i in range(6):
        comm.get().reset_communication_stats()
        tic = time.perf_counter()

        output = model(input)

        loss = criterion(output, labels)
        loss.backward()
        model.update_parameters(learning_rate=0.1)

        toc = time.perf_counter()

        if i != 0:
            total_time += toc - tic
            comm_time += comm.get().comm_time
            conv_time += comm.get().time_conv
            relu_time += comm.get().time_relu
            pool_time += comm.get().time_pool
            matmul_time += comm.get().time_matmul
            softmax_time += comm.get().time_softmax

            # if comm.get().get_rank() == 0:
            #     print(f"Iteration {i} runtime: {toc - tic}")

        comm.get().print_total_communication()

    if comm.get().get_rank() == 0:
        print("----------- Statistics ----------------")
        print(f"Total Communication: {comm.get().total_comm_bytes}")
        print(f"Avg Runtime: {total_time / 5}")
        print(f"Avg Comm: {comm_time / 5}")
        print(f"Avg Linear: {(conv_time + matmul_time)/ 5}")
        print(f"Avg ReLU: {relu_time / 5}")
        print(f"Avg Pool: {pool_time / 5}")
        print(f"Avg Softmax: {softmax_time / 5}")



def inference_plaintext(model, input_size, device="cuda"):

    c, w, h = input_size
    x = torch.rand((1, c, w, h), device=device, requires_grad=False)

    model = model.to(device) 
    model.eval()

    total_time = 0
    for i in range(101):
        comm.get().reset_communication_stats()
        
        tic = time.perf_counter()
        model(x)
        toc = time.perf_counter()

        if i != 0:
            total_time += toc - tic

        comm.get().print_total_communication()

    if comm.get().get_rank() == 0:
        print("----------- Statistics ----------------")
        print(f"Avg Runtime: {total_time / 100}")


def training_plaintext(model, input_size, batch_size, num_classes, device="cuda"):

    c, h, w = input_size
    bs = batch_size

    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    input = torch.randn([bs,c,w,h], requires_grad=True).to(device)
    labels =  torch.ones(bs, requires_grad=False).long().to(device)

    total_time = 0
    comm_time = 0
    conv_time, pool_time, relu_time, matmul_time = 0, 0, 0, 0
    for i in range(101):
        tic = time.perf_counter()

        output = model(input)

        optimizer.zero_grad()
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        toc = time.perf_counter()

        if i != 0:
            total_time += toc - tic

            # if comm.get().get_rank() == 0:
            #     print(f"Iteration {i} runtime: {toc - tic}")

    if comm.get().get_rank() == 0:
        print("----------- Statistics ----------------")
        print(f"Avg Runtime: {total_time / 100}")


def select_model(dataset, network):
    if dataset == "mnist":
        input_size = (1,28,28)
        num_classes = 10
        if network == "lenet":
            model = LeNet()
    elif dataset == "cifar10":
        input_size = (3,32,32)
        num_classes = 10
        if network == "alexnet":
            model = AlexNet(num_classes=10)
        elif network == "vgg16":
            model = VGG16(num_classes=10)
    elif dataset == 'tinyin':
        input_size = (3,64,64)
        num_classes = 200
        if network == 'alexnet':
            model = AlexNet(num_classes=200)
        elif network == "vgg16":
            model = VGG16(num_classes=200)
    elif dataset == 'imagenet':
        input_size = (3, 224, 224)
        num_classes = 1000
        if network == 'alexnet':
            model = AlexNet(num_classes=1000)
        elif network == "vgg16":
            model = VGG16(num_classes=1000)
        elif network == "resnet34":
            model = models.resnet34()
            model.maxpool = nn.AvgPool2d(kernel_size=3, stride=2)
        elif network == "resnet50":
            model = models.resnet50()
            model.maxpool = nn.AvgPool2d(kernel_size=3, stride=2)
        elif network == "resnet101":
            model = models.resnet101()
            model.maxpool = nn.AvgPool2d(kernel_size=3, stride=2)
        elif network == "resnet152":
            model = models.resnet152()
            model.maxpool = nn.AvgPool2d(kernel_size=3, stride=2)

    return model, input_size, num_classes

def train_all():
    train_config = [
        ["mnist", "lenet", 128],
        ["cifar10", "alexnet", 128],
        ["cifar10", "vgg16", 32],
        ["tinyin", "alexnet", 128],
        ["tinyin", "vgg16", 8],
    ]
    for dataset, network, bs in train_config:
        model, input_size, num_classes = select_model(dataset, network)
        if comm.get().get_rank() == 0:
            print(f"Training on {dataset} dataset with {network} network")
        training(model, input_size, bs, num_classes, device="cuda")


def inference_all():
    inference_config = [
        ["mnist", "lenet"],
        ["cifar10", "alexnet"],
        ["cifar10", "vgg16"],
        ["tinyin", "alexnet"],
        ["tinyin", "vgg16"],
        ["imagenet", "alexnet"],
        ["imagenet", "vgg16"],
        ["imagenet", "resnet50"],
        ["imagenet", "resnet101"],
        ["imagenet", "resnet152"]
    ]
    for dataset, network in inference_config:
        model, input_size, num_classes = select_model(dataset, network)
        if comm.get().get_rank() == 0:
            print(f"Running inference on {dataset} dataset with {network} network")
        inference(model, input_size, device="cuda")


def train_all_plaintext():
    train_config = [
        ["mnist", "lenet", 128],
        ["cifar10", "alexnet", 128],
        ["cifar10", "vgg16", 32],
        ["tinyin", "alexnet", 128],
        ["tinyin", "vgg16", 8],
    ]
    for dataset, network, bs in train_config:
        model, input_size, num_classes = select_model(dataset, network)
        if comm.get().get_rank() == 0:
            print(f"Training on {dataset} dataset with {network} network")
        training_plaintext(model, input_size, bs, num_classes, device="cuda")


def inference_all_plaintext():
    inference_config = [
        ["mnist", "lenet"],
        ["cifar10", "alexnet"],
        ["cifar10", "vgg16"],
        ["tinyin", "alexnet"],
        ["tinyin", "vgg16"],
        ["imagenet", "alexnet"],
        ["imagenet", "vgg16"],
        ["imagenet", "resnet50"],
        ["imagenet", "resnet101"],
        ["imagenet", "resnet152"]
    ]
    for dataset, network in inference_config:
        model, input_size, num_classes = select_model(dataset, network)
        if comm.get().get_rank() == 0:
            print(f"Running inference on {dataset} dataset with {network} network")
        inference_plaintext(model, input_size, device="cuda")


def batch_inference():
    inference_config = [
        ["cifar10", "alexnet", 64],
        ["cifar10", "vgg16", 64],
        ["imagenet", "resnet50", 8],
        ["imagenet", "resnet101", 8],
        ["imagenet", "resnet152", 8]
    ]

    for dataset, network, bs in inference_config:
        model, input_size, num_classes = select_model(dataset, network)
        inference(model, input_size, bs, device='cuda')

# A playground to test different network and dataset combinations
def test():
    dataset = "cifar10"
    network = "alexnet"
    device = "cuda"
    train = True
    batch_size = 1

    model, input_size, num_classes = select_model(dataset, network)

    if train:
        training(model, input_size, batch_size, num_classes, device)
    else:
        inference(model, input_size, device)


parser = argparse.ArgumentParser()
experiments = ['test', 'train_all', 'inference_all', 'train_all_plaintext', 'inference_all_plaintext', 'batch_inference']
parser.add_argument(
    "--exp",
    "-e",
    required=False,
    default="test",
    help="Experiment to run",
)

if __name__ == '__main__':
    import multiprocess_launcher

    args = parser.parse_args()
    assert args.exp in experiments
    func = globals()[args.exp]

    launcher = multiprocess_launcher.MultiProcessLauncher(
        3, func,
    )
    launcher.start()
    launcher.join()
    launcher.terminate()