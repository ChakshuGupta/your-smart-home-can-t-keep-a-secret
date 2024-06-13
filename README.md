# Your Smart Home Can't Keep a Secret Implementation

Implementation of the paper Your Smart Home Can't Keep a Secret: . 
The link to the paper: https://dl.acm.org/doi/10.1145/3320269.3384732.

## Install the pre-requisites
```
pip install -r requirements.txt
```


## How to run the code:
1. Add the path to the dataset in the config file, config.yml. (Train and test datasets can be the same.)
```
dataset-path:
  train: <path to the training dataset>
  test: <path to the testing dataset>
```

2. Add the path file containing the mapping between devices and mac addresses.
```
device-file: <path to the text file>
```

3. Run the code using the following command:
```
python main.py config.yml
```