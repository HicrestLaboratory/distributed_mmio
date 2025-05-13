# distributed_mmio

## Cloning

To clone this repo with all the git submodules initialized run the following command

```
git clone --recurse-submodules https://github.com/HicrestLaboratory/distributed_mmio.git
```

If you already cloned the repo and initialize recursively the submodules

```
git submodule update --init --recursive
```

## Dataset creation

To be able to run the script that produces data for your experiments you should install `conda` with all the required packages and configure
the yaml configuration file with the data sources of your choice.

Install conda with your favorite package manager and then create a new environment with the following command:

```
conda env create -f environment.yml
```

Also if you want to use the graph500 generator then you should install the MPICH library in order to compile and run the executable that generates graphs.

Once the environment is ready, it is time to configure what kind of data will be produced with this library:
you should create your own config.yaml file (based on the config.example.yaml file) with the configuration that fits your case.

Once the config file is configured, just run the following command

```
python3 scripts/create_dataset.py
```

This way you will start the CLI tool that will guide you with the data creation process.
