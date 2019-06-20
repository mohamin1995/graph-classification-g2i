# graph-classification-g2i
A Novel Functional Brain Connectivity Network Classification Method


#### Story:
Connections in the human brain can be examined eﬃciently using brain imaging techniques such as Diﬀusion Tensor Imaging (DTI), Resting-State fMRI. Brain connectivity networks are constructed by using image processing and statistical methods, these networks explain how brain regions interact with each other. Brain networks can be used to train machine learning models that can help the diagnosis of neurological disorders.

In functional brain graphs, the nodes describe the regions and the edge weights correspond to the values of correlation coefficients of the time-series of the two nodes associated with the edges.


#### Task: 
*Functional Brain Connectivity Network Classification for ASD Screening* 

#### Method: 
Convert graph classification to image classification based on this [paper](https://arxiv.org/abs/1804.06275).you can see an image obtained from the brain functional network:

![Brain Functional Network](http://s5.picofile.com/file/8364197492/Capture.PNG)

#### Dataset:
[UCLA Dataset](http://umcd.humanconnectomeproject.org/)

#### Requirments:
- python3
- sklearn
- networkx 2.1
- numpy

#### Run:
Download ASD and TD FMRI connectivity networks from [UCLA Dataset](http://umcd.humanconnectomeproject.org/) and set paths to those folders in conf.ini file.

command:</br> *python Demo.py -c [path to conf.ini]*

#### Output:
.......... [report] ..........
+ accuracy = 0.7272727272727273
+ precision = 0.7878787878787878
+ recall = 0.65
