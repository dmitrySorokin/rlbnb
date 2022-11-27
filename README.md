# Experiments with heuristics learning for B&B

## Tasks supported

* set_covering
* combinatorial_auction
* capacitated_facility_location
* maximum_independent_set
* crabs
* tsp

## Install

```pip3 install -r requirements.txt```

## Train an agent

```python3 main.py --config-name {cfg_from_configs}.yaml```  

checkpoints will be saved in  
```outputs/year-month-day/hour-min-sec/checkpoint_{id}.pkl```

## Eval agent

* download and unpack task instances from [google drive](https://drive.google.com/file/d/1TeeTpnfI4XbqeTJKXelzNZiMbLO3c71h/view?usp=share_link)
* ```python3 eval.py --config-name {cfg_from_configs}.yaml agent.name={agent_name}```
* agent_name: strong, dqn, random
* results will be saved in ```results/{task_name}/{agent_name}.csv```

## Plot results

```python3 plot.py results/{task_name}/```

## Setup

```bash
# conda update -n base -c defaults conda
# conda deactivate && conda env remove -n rlbnb

# pytorch, scip, scipy, ecole and other essentials
# XXX feezing pytorch at 1.10 due to openmp issues on Mac M1:(
conda create -n rlbnb python pip setuptools numpy "scipy>=1.9" scikit-learn \
  "pytorch::pytorch=1.10" "pyg::pyg" "conda-forge::pyscipopt" pandas \
  matplotlib notebook networkx graphviz ipywidgets

# service, viz and configs, quality of life
conda activate rlbnb \
  && pip install ecole einops tqdm ordered-set tensorboardX hydra-core omegaconf

# walk like crab, cause a problem -- crab problem
pip install git+https://github.com/ivannz/branching-crustaceans.git#crab-alloc

# install other dev dependencies
conda install -n rlbnb pytest \
  && conda activate rlbnb \
  && pip install "black[jupyter]" pre-commit gitpython nbdime
```
