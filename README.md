## FeLeDetect: Federated Learning for Error Detection over Different Data Sources

FeLeDetect, a federatedlearning-based error detection approach, which utilizes different data sources to improve the quality of error detection without privacy leakage. First, a graph-based error detection model GEDM is presented to capture sufficient data features from each data source for FeLeDetect. Then, an information-lossless federated learning mechanism is proposed to collaboratively train GEDM over different data sources without privacy leakage. Furthermore, we design a series of optimizations to reduce the communication cost during
the federated learning and the manual labeling effort. 

## Requirements

* Python 3.7
* PyTorch 1.7.1
* torch_scatter 2.0.7
* CUDA 10.1
* 4 NVIDIA GeForce RTX2080ti GPU

Please refer to the source code to install all required packages in Python

## Datasets

We conduct experiments on three real-life datasets with differnet types of data errors, including substitute errors, missing values, violated attribute
dependencies, and format issues. 

## Run Experimental Case

To train the FeLeDetect for error detection over different data sources in the Federated senario on DA_5:
```
python fed_main.py -dataset DA_5
```

To train the GEDM for error detection over single data source DA_5 in the Cetralized senario:
```
python main.py -dataset DA_5 -whole true
```

To train the GEDM for error detection over single data source DA_5_1 in the Local senario:
```
python main.py -dataset DA_5_1
```


## Acknowledgement

We use the code of [Raha](https://github.com/BigDaMa/raha).

The original datasets DBLP-ACM is from https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md

The original dataset flights is from http://lunadong.com/fusionDataSets.htm
