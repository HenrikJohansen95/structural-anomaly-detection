Copyright (C) 2022 Henrik Syversen Johansen. The code and its documentation,
with the exception of the thesis text itself in [thesis.pdf](thesis.pdf), is licensed under
the BSD 3-Clause License (see [LICENSE](LICENSE)).

# Structural Anomaly Detection in Knowledge Graphs Using Graph Neural Networks 
Anomalies are in general rare or somehow different elements among a set of
elements. Identifying such elements has several important applications within
the realms of financial fraud, review fraud, health care, security and others.
In this work we propose a definition of what it means for a substructure of a
knowledge graph to be anomalous. We also create AI systems that use end-to-end
graph neural networks to classify these substructures as anomalous or not. In
evaluating the performance of these AI systems we show that while they are much
faster than a precise symbolic implementation of the definition of anomalous
substructures, the results are expectedly lower with regards to classification
performance. In real-life scenarios where this loss in classification
performance is tolerable however, we have shown that our approach is feasible.


## Hardware and Software Details
The entire project was developed and executed on the following hardware:
- CPU: Intel i7-8700
- RAM: 32 GB
- GPU: Nvidia GeForce 2080

Running on WSL 2 with Cuda version 11.7

## System setup
- Run `make` to download and extract Jena fuseki 4.4.0 and the LUBM graph generator
- Run `make start_jena` to start Jena fuseki server in the current shell
- Run `make lubm_generate` to generate lubm graphs

### Installing Python environment
As this project was developed on WSL it should work fine with Linux, it might be
a bit more challenging to get it running on mac however. You'd have to manually
install non-cuda versions of all the packages, but it should be possible.

To install the Python virtual environment using:
- Anaconda / Miniconda (recommended) run `make conda`
- pip requirements file run `make venv`

The latter works in my dev environment, but I have no guarantee that it will
work in yours; conda *should* be more reliable. If none of the make recipes
work, here are the package versions:
- Python version 3.10.8 
    - It should work with at least 3.9 but no guarantees
- PyTorch 1.13 Cuda 11.7
- PyTorch Lightning 1.7.7 
    - This one is important as some callbacks used by Ray Tune were deprecated
      in versions >= 1.8
- Ray Tune 2.1

If something else breaks, all the dependencies and their versions are listed in
both the conda-env.yml and the requirements.txt files.

## Running the Code
The following files provide a very basic CLI, and information about arguments
can be obtained using the --help flag: 
- [src/train.py](src/train.py) contains the code to train the system using pytorch
  lightning.
- [src/run_raytune.py](src/run_raytune.py) contains hyperparameter search code
- [src/generate_samples.py](src/generate_samples.py) contains sample generation code for `pytorch geometric`
  included datasets
- [src/generate_lubm_samples.py](src/generate_lubm_samples.py) contains sample generation code for generated LUBM graphs
  

In case you are using an external SPARQL server, the endpoint can be set in [src/config.py](src/config.py).