# HDBSCAN* - Python Version

This code is a version of the HDBSCAN* algorithm [[1]](#1).

## Installation

Please follow the steps below to install 

### GitHub Checkout

```
git checkout git@github.com:antoniocavalcante/hdbscan_python.git
```

### Create Virtual Environment:

```
python -m venv env
```

### Activate Virtual Environment:
```
source env/bin/activate
```

### Install Requirements:

```
pip install -r requirements.txt
```

```
python3 -m pip install -r requirements.txt
```

## Usage



## Troubleshooting

### `fatal error: numpy/arrayobject.h: No such file or directory`

Create a symbolic link from the Numpy installation in the Python version being used (e.g. 3.8) to `/usr/include/numpy`

```
sudo ln -s  /usr/local/lib/python3.8/dist-packages/numpy/core/include/numpy /usr/include/numpy
```

## References
<a id="1">[1]</a> 
Ricardo J. G. B. Campello, Davoud Moulavi, Arthur Zimek, Joerg Sander. 
Hierarchical Density Estimates for Data Clustering, Visualization, and Outlier Detection.
ACM Trans. Knowl. Discov. Data 10(1): 5:1-5:51 (2015)