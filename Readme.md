# Approximate hierarchical clustering  

Implementation of Integer Linear Program (ILP) to recover exact hierarchical clustering
according to the cost function proposed by Dasgupta (STOC '16) as well as `O(log n)`
LP rounding based approximation algorithm. 

Authors: Aurko Roy and Sebastian Pokutta (to appear in NIPS 2016) 

## Non standard dependencies
 - [munkres](https://pypi.python.org/pypi/munkres/)
 - [gurobi](http://www.gurobi.com/) 
 - [networkx](https://pypi.python.org/pypi/networkx/)

## Usage
To run the ILP (warning will be quite slow on large datasets) type

```shell
python ip.py -d [data] -l [label_index] -k [kernel] -f [function] -o [out_file] -n [num_workers] 
-p [prune] -x [num_samples]
```

* `[data]`: the path to the data file (comma delimited)
* `[label_index]`: the index of the column that contains the labels (default 0)
* `[kernel]`: one of `{linear, gaussian}`
* `[function]`: one of `{linear, quadratic, cubic, exponential, logarithm}`
* `[num_workers]`: the number of workers for parallelization (default 1)
* `[prune]`: number of flat clusters to prune to
* `[num_samples]`: on large sets, sample this many points uniformly at random

To run the approximation algorithm (faster) type

```shell
python lp.py -d [data] -l [label_index] -k [kernel] -f [function] -o [out_file] -n [num_workers] 
-p [prune] -x [num_samples] -t [triangle] -e [epsilon]
```

* `[data]`: the path to the data file (comma delimited)
* `[label_index]`: the index of the column that contains the labels (default 0)
* `[kernel]`: one of `{linear, gaussian}`
* `[function]`: one of `{linear, quadratic, cubic, exponential, logarithm}`
* `[num_workers]`: the number of workers for parallelization (default 1)
* `[prune]`: number of flat clusters to prune to
* `[num_samples]`: on large sets, sample this many points uniformly at random
* `[triangle]`: if true, then use lazy triangle inequality (faster)
* `[epsilon]`: algorithm parameter (`0.5` is usually a safe choice)
