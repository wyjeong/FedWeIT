# Federated Continual Learning with Weighted Inter-client Transfer

This repository is an official Tensorflow 2 implementation of [Federated Continual Learning with Weighted Inter-client Transfer](https://arxiv.org/abs/2003.03196) (**ICML 2021**)

> Currently working on PyTorch version 

## Abstract

There has been a surge of interest in continual learning and federated learning, both of which are important in deep neural networks in real-world scenarios. Yet little research has been done regarding the scenario where each client learns on a sequence of tasks from a private local data stream. This problem of federated continual learning poses new challenges to continual learning, such as utilizing knowledge from other clients, while preventing interference from irrelevant knowledge.  To resolve these issues, we propose a novel federated continual learning framework, Federated Weighted Inter-client Transfer (FedWeIT), which decomposes the network weights into global federated parameters and sparse task-specific parameters, and each client receives selective knowledge from other clients by taking a weighted combination of their task-specific parameters. FedWeIT minimizes interference between incompatible tasks, and also allows positive knowledge transfer across clients during learning. We validate our FedWeIT against existing federated learning and continual learning methods under varying degrees of task similarity across clients, and our model significantly outperforms them with a large reduction in the communication cost. 

The main contributions of this work are as follows:

* We introduce a new problem of **Federated Continual Learning (FCL)**, where multiple models continuously learn on distributed clients, which poses new challenges such as prevention of inter-client interference and inter-client knowledge transfer. 

* We propose a novel and communication-efficient framework for federated continual learning, which allows each client to adaptively update the federated parameter and selectively utilize the past knowledge from other clients, by communicating sparse parameters. 


## Environmental Setup

Please install packages from `requirements.txt` after creating your own environment with `python 3.8.x`.

```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Data Generation
Please see `config.py` to set your custom path for both `datasets` and `output files`.
```python
args.task_path = '/path/to/task/'  # for dataset
args.output_path = '/path/to/outputs/' # for logs, weights, etc.
```
Run below script to generate datasets
```bash
$ cd scripts
$ sh gen-data.sh
```
or you may run the following comamnd line directly:

```bash
python3 ../main.py --work-type gen_data --task non_iid_50 --seed 777 
```
It automatically downloads `8 heterogeneous datasets`, including `CIFAR-10`, `CIFAR-100`, `MNIST`, `Fashion-MNIST`, `Not-MNIST`, `TrafficSigns`, `Facescrub`, and `SVHN`, and finally processes to generate `non_iid_50` dataset.

## Run Experiments
To reproduce experiments, please execute `train-non-iid-50.sh` file in the `scripts` folder, or you may run the following comamnd line directly:

```bash
python3 ../main.py --gpu 0,1,2,3,4 \
		--work-type train \
		--model fedweit \
		--task non_iid_50 \
	 	--gpu-mem-multiplier 9 \
		--num-rounds 20 \
		--num-epochs 1 \
		--batch-size 100 \
		--seed 777 
```
Please replace arguments as you wish, and for the other options (i.e. hyper-parameters, etc.), please refer to `config.py` file at the project root folder.

> Note: while training, all participating clients are logically swiched across the physical gpus given by `--gpu` options (5 gpus in the above example). 

## Results
All clients and server create their own log files in `\path\to\output\logs\`, which include evaluation results, such as local & global performance and communication costs, and the experimental setups, such as learning rate, batch-size, etc. The log files will be updated for every comunication rounds. 

## Citations
```
@inproceedings{
    yoon2021federated,
    title={Federated Continual Learning with Weighted Inter-client Transfer},
    author={Jaehong Yoon and Wonyong Jeong and Giwoong Lee and Eunho Yang and Sung Ju Hwang},
    booktitle={International Conference on Machine Learning},
    year={2021},
    url={https://arxiv.org/abs/2003.03196}
}
```
