Comparing runtime of identical LSTM architectures, but one using GPU via CUDA

- run LSTM_Standard.py (single CPU, Keras)

vs

- RunTorchLSTM.py which calls LSTM_Torch_CUDA.py

for a single, artificial toy dataset.

Benchmarks:

on Lenovo T14, NVIDIA GeForce MX450: 

n_neurons 	= 200
n_epoch		= 100
n_stack		= 2
dt_past		= 20
dt_futu		= 5


LSTM_Standard.py (CPU):		28 sec
LSTM_Torch_CUDA.py (CPU):	 8 sec
LSTM_Torch_CUDA.py (GPU):	 2 sec

