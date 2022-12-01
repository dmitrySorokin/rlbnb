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
```python3 main.py --config-name {cfg_from_configs}.yaml```  

checkpoints will be saved in  
```outputs/year-month-day/hour-min-sec/checkpoint_{id}.pkl```

## Eval agent
* download and unpack task instances from [google drive](https://drive.google.com/file/d/1TeeTpnfI4XbqeTJKXelzNZiMbLO3c71h/view?usp=share_link)
* ```python3 eval.py --config-name {cfg_from_configs}.yaml```
