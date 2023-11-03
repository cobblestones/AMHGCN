## AMHGCN: Adaptive Multi-level Hypergraph Convolution Network for Human Motion Prediction
This is the code for the paper

Jinkai Li, Jinhua Wang, Lian Wu, Xin Wang, Xiaoling Luo, Yong Xu
[_"AMHGCN: Adaptive Multi-level Hypergraph Convolution Network for Human Motion Prediction"_]. In Neural Networks (NN) 2023.

### Network Architecture
------
![image](.github/network.png)

This is the network architecture of our proposed adaptive multi-level hypergraph convolution network (AMHGCN) for human motion prediction. 

### Experiment Performance Visualization

![image](.github/Smoking.gif)

This is the experiment visualization of our proposed AMHGCN in the scenario “smoking.” In the historical human motion sequence, the groundtruth is marked as magenta-black with a complete line. While at the predicted human motion sequence, the groundtruth is marked as magenta-black with a dashed line, while our prediction results are shown as blue-green with a complete line.

![image](.github/Walking_dog.gif)

This is the experiment visualization of our proposed AMHGCN in the scenario “walking together.” In the historical human motion sequence, the groundtruth is marked as magenta-black with a complete line. While at the predicted human motion sequence, the groundtruth is marked as magenta-black with a dashed line, while the comparative results are shown as blue-green with a complete line.

 


### Dependencies

* cuda 11.4
* Python 3.8
* Pytorch 1.7.0

### Get the data

[Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).

Directory structure: 
```shell script
h3.6m
|-- S1
|-- S5
|-- S6
|-- ...
`-- S11
```
[CMU mocap](http://mocap.cs.cmu.edu/) was obtained from the [repo](https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics) of ConvSeq2Seq paper.
Directory structure:
```shell script
cmu_mocap
|-- test
|-- train
```

[3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) from their official website.

Directory structure: 
```shell script
3dpw
|-- imageFiles
|   |-- courtyard_arguing_00
|   |-- courtyard_backpack_00
|   |-- ...
`-- sequenceFiles
    |-- test
    |-- train
    `-- validation
```
Put the all downloaded datasets in ./datasets directory.

### Training
All the running args are defined in [opt.py](utils/opt.py). We use following commands to train on different datasets and representations.
To train,
```bash
python main_h36m_3d.py python train_H36M.py  --num 64  --reverse_weight 0.05
```
```bash
python main_cmu_mocap_3d.py --num 48  --reverse_weight 0.05
```
```bash
python main_3dpw_3d.py --num 48  --reverse_weight 0.05
```
### Evaluation
To evaluate the pretrained model,
```bash
python test_H36M.py
```

### Citing

If you use our code, please cite our work

```

```

### Acknowledgments

