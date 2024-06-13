# Your Smart Home Can't Keep a Secret Implementation

Implementation of the paper __"Your Smart Home Can't Keep a Secret: Towards Automated Fingerprinting of IoT Traffic"__.

Citation:
```
Shuaike Dong, Zhou Li, Di Tang, Jiongyi Chen, Menghan Sun, and Kehuan Zhang. 2020. Your Smart Home Can't Keep a Secret: Towards Automated Fingerprinting of IoT Traffic. In Proceedings of the 15th ACM Asia Conference on Computer and Communications Security (ASIA CCS '20).
```

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