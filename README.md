# Experiments with heuristics learning for B&B 

## Tasks supported:
* set_covering
* combinatorial_auction
* capacitated_facility_location
* maximum_independent_set
* crabs
* tsp

## Install
```pip3 install -r requirements.txt```

## Train an agent
```python3 main.py```  

checkpoints will be saved in  
```outputs/year-month-day/hour-min-sec/checkpoint_{id}.pkl```

## Eval agent
```python3 eval.py```