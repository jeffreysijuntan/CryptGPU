<!-- <p align="center"><img width="70%" src="https://raw.githubusercontent.com/facebookresearch/CrypTen/master/docs/_static/img/CrypTen_Identity_Horizontal_Lockup_01_FullColor.png" alt="CrypTen logo" /></p>

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/CrypTen/blob/master/LICENSE) [![CircleCI](https://circleci.com/gh/facebookresearch/CrypTen.svg?style=shield)](https://circleci.com/gh/facebookresearch/CrypTen/tree/master) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/facebookresearch/CrypTen/blob/master/CONTRIBUTING.md) -->

--------------------------------------------------------------------------------

# CryptGPU: Fast Privacy-Preserving Machine Learning on the GPU

CryptGPU is a system for privacy-preserving machine learning based on secure multi-party computation (MPC).  supports end-to-end training/inference on the GPU. This implementation is according to the paper: [CryptGPU: Fast Privacy-Preserving Machine Learning on the GPU](https://arxiv.org/abs/2104.10949) by Sijun Tan, Brian Knott, Yuan Tian, David J. Wu.

The base architecture of our system is adapted from [CrypTen](https://github.com/facebookresearch/crypten), a privacy-preserving machine learning framework (PPML) built on top of PyTorch. CrypTen is a machine learning first framework that aims to make secure computing techniques easily accessible to machine learning practitioners. 

**WARNING**: This codebase is an academic proof-of-concept prototype that is released as a reference to the paper, and for benchmarking purposes. The codebase has not received careful code review, and is NOT ready for production use. 

## Installing CryptGPU
**Caveat**: Our system uses [torchcsprng]() as cryptographically secure pseudorandom number generator, which generates randomness using AES cipher in CTR mode. In the initialization step, parties need to synchronize AES keys to generate correlated randomness. However, the official packaged version of torchcsprng does not provide a function to generate and synchronize AES key explicitly. To use this functionality, we need to build both PyTorch and torchcsprng from source. 

Considering the difficulties building package from source, we provide extra options to build dependencies through `pip3` so that developers/researchers can still play with our system and reproduce the experiment results. 

If dependencies are installed through `pip3`, developers/researchers can choose to either a) use torchcsprng without synchronizing keys. b) use PyTorch's RNG that is not cryptographically secure instead. The first option can be used to replicate runtime performance benchmarks, but the system outputs will be incorrect. The second option can be used to run inference/training on our system with correct outputs, but the RNG is not fully secure. To obtain the correct outputs with a CSPRNG, one has to build dependencies from source.

### Building dependencies through pip
```bash
git clone https://github.com/jeffreysijuntan/cryptgpu
cd cryptgpu
pip3 install -r requirements.txt
python3 setup.py install
```

To set `launcher.py` to setup options. There are two options available, `use_csprng` and `sync_key`. If `use_csprng` is set to `True`, then torchcsprng will be used as the random number generator. Otherwise, PyTorch's default torchcsprng will be chosen. If `sync_key` is set to `True`, then torchcsprng will synchronize AES keys during initialization. An error will be shown if `sync_key` is set to `True` and dependencies are not built from source.

### Building dependencies from source (on Linux)
First build PyTorch from source
```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
# if you are updating an existing checkout
git checkout 1.6
git submodule sync
git submodule update --init --recursive
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
```
Then install torchcsprng and built it from source
```bash
git clone https://github.com/pytorch/csprng
cd csprng
git checkout 64fedf7e0ab93188cc06b39308ad2ec0e3771bb2
python setup.py install
```

Finally, install other dependencies for CryptGPU through `pip3`
```
git clone https://github.com/jeffreysijuntan/cryptgpu
cd cryptgpu
pip3 install -r requirements_source.txt
python3 setup.py install
```


## Performance Benchmark
To measure the performance, run `scripts/benchmark.py`

```bash
python3 scripts/benchmark.py --exp inference_all
```
where `--exp` specifies the experiments to run. Experiments include `inference_all, train_all, train_plaintext, inference_plaintext`. This line of command will create three processes locally to simulate 3PC computations.

To run benchmark on AWS servers, we provide scripts to facilitate this process. First, put AWS instance ids into `launch_aws.sh` (comma seperated, no empty space). Then, go to `launcher.py` to specify the benchmark experiment to run. Finally, execute 
```
bash launch_aws.sh
```

## Comparison with CrypTen
The base architecture of our system is adapted from [CrypTen](https://github.com/facebookresearch/crypten). CrypTen uses beaver triples as the basic building block, and scales to arbitrarily many parties. We instead only consider the 3PC setting, and use replicated-secret sharing as our basic building block. Our system only supports a limited number of protocols that is necessary for inference/training over convolutional neural networks described in our paper. The operations that we support are: matrix multiplication, convolution, ReLU, average pooling, batch normalization (only for inference), and cross-entropy loss. CrypTen additionally supports a much wider range of operations (e.g sigmoid, tanh) that is not covered in our paper, we therefore remove many operations from CrypTen that we do not describe in our paper. Please check out CrypTen if you are interested in these additional operations. We also have plans to migrate some components described in this paper to CrypTen in the near future. 


## Citation
You can cite the paper using the following bibtex entry
```
@inproceedings{TKTW21,
  author     = {Sijun Tan and Brian Knott and Yuan Tian and David J. Wu},
  title      = {\textsc{CryptGPU}: Fast Privacy-Preserving Machine Learning on the GPU},
  booktitle  = {{IEEE} {S\&P}},
  year       = {2021}
}
```

<!-- _For Linux or Mac_
```bash
pip install crypten
```

If you want to run the examples in the `examples` directory, you should also do the following
```bash
pip install -r requirements.examples.txt
```

## Examples
To run the examples in the `examples` directory, you additionally need to clone the repo and

```bash
pip install -r requirements.examples.txt
```

We provide examples covering a range of models in the `examples` directory

1. The linear SVM example, `mpc_linear_svm`, generates random data and trains a
  SVM classifier on encrypted data.
2. The LeNet example, `mpc_cifar`, trains an adaptation of LeNet on CIFAR in
  cleartext and encrypts the model and data for inference.
3. The TFE benchmark example, `tfe_benchmarks`, trains three different network
  architectures on MNIST in cleartext, and encrypts the trained model and data
  for inference.
4. The bandits example, `bandits`, trains a contextual bandits model on
  encrypted data (MNIST).
5. The imagenet example, `mpc_imagenet`, performs inference on pretrained
  models from `torchvision`.

For examples that train in cleartext, we also provide pre-trained models in
cleartext in the `model` subdirectory of each example subdirectory.

You can check all example specific command line options by doing the following;
shown here for `tfe_benchmarks`:

```bash
python examples/tfe_benchmarks/launcher.py --help
```

## How CrypTen works

We have a set of tutorials in the `tutorials` directory to show how
CrypTen works. These are presented as Jupyter notebooks so please install
the following in your conda environment

```bash
conda install ipython jupyter
pip install -r requirements.examples.txt
```

1. `Introduction.ipynb` - an introduction to Secure Multiparty Compute; CrypTen's
   underlying secure computing protocol; use cases we are trying to solve and the
   threat model we assume.
2. `Tutorial_1_Basics_of_CrypTen_Tensors.ipynb` - introduces `CrypTensor`, CrypTen's
   encrypted tensor object, and shows how to use it to do various operations on
   this object.
3. `Tutorial_2_Inside_CrypTensors.ipynb` – delves deeper into `CrypTensor` to show
   the inner workings; specifically how `CrypTensor` uses `MPCTensor` for its
   backend and the two different kind of _sharings_, arithmetic and binary, are
   used for two different kind of functions. It also shows CrypTen's
   [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface)-inspired
   programming model.
4. `Tutorial_3_Introduction_to_Access_Control.ipynb` - shows how to train a linear
   model using CrypTen and shows various scenarios of data labeling, feature
   aggregation, dataset augmentation and model hiding where this is applicable.
5. `Tutorial_4_Classification_with_Encrypted_Neural_Networks.ipynb` – shows how
   CrypTen can load a pre-trained PyTorch model, encrypt it and then do inference
   on encrypted data.
6. `Tutorial_5_Under_the_hood_of_Encrypted_Networks.ipynb` - examines how CrypTen
   loads PyTorch models, how they are encrypted and how data moves through a multilayer
   network.
7. `Tutorial_6_CrypTen_on_AWS_instances.ipynb` - shows how to use `scrips/aws_launcher.py`
   to launch our examples on AWS. It can also work with your code written in CrypTen.
8. `Tutorial_7_Training_an_Encrypted_Neural_Network.ipynb` - introduces the
   automatic differentiation functionality of `CrypTensor`. This functionality
   makes it easy to train neural networks in CrypTen.


## Documentation
CrypTen is documented [here](https://crypten.readthedocs.io/en/latest/)

## Join the CrypTen community
Please contact [us](mailto:ssengupta@fb.com) to join the CrypTen community on [Slack](https://cryptensor.slack.com)

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
CrypTen is MIT licensed, as found in the LICENSE file. -->
