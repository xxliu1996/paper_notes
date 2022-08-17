# An Efficient FPGA-Based Convolutional Neural Network for Classification: Ad-MobileNet

[An Efficient FPGA-Based Convolutional Neural Network for Classification: Ad-MobileNet](https://www.mdpi.com/2079-9292/10/18/2272)

# Abstract

Convolutional Neural Networks (CNN) continue to dominate research in the area of hardware acceleration using Field Programmable Gate Arrays (FPGA), proving its effectiveness in a variety of computer vision applications such as object segmentation, image classification, face detection, and traffic signs recognition, among others. However, there are numerous constraints for deploying CNNs on FPGA, including limited on-chip memory, CNN size, and configuration parameters. This paper introduces Ad-MobileNet, an advanced CNN model inspired by the baseline MobileNet model. The proposed model uses an Ad-depth engine, which is an improved version of the depth-wise separable convolution unit. Moreover, we propose an FPGA-based implementation model that supports the Mish, TanhExp, and ReLU activation functions. The experimental results using the CIFAR-10 dataset show that our Ad-MobileNet has a classification accuracy of 88.76% while requiring little computational hardware resources. Compared to state-of-the-art methods, our proposed method has a fairly high recognition rate while using fewer computational hardware resources. Indeed, the proposed model helps to reduce hardware resources by more than 41% compared to that of the baseline model.

# Article Structure

### Introduction

<aside>
💡 This paper proposes Ad-MobileNet, which is based on MobileNet. Contributions of this paper include: 1) propose Ad-depth unit, which is better suited for hardware implementation; 2) layers having negative impact on the recognition rate have been removed; 3) more activation functions have been tested.

</aside>

### Related Work

<aside>
💡 Implementing CNNs on FPGA has the following advantages: flexibility, high throughput, fine-grain parallelism, and energy efficiency. Constraints: limitations in memory resources, computing power, and power consumption restrictions.

</aside>

<aside>
💡 Several works on implementing CNNs on FPGA were introduced, the author compared their advantages and disadvantages. Here are FPGA devices mentioned in this section: [Xilinx ZCU102](https://www.xilinx.com/products/boards-and-kits/ek-u1-zcu102-g.html), [Xilinx VC709](https://www.xilinx.com/products/boards-and-kits/dk-v7-vc709-g.html), [Xilinx Zynq XC7Z020](https://www.xilinx.com/products/boards-and-kits/1-1bkpiyc.html).

</aside>

### Background

<aside>
💡 An introduction of convolution, activation, pooling and fully-connected layers.

- Convolution Layer. input array @ convolution filter —> feature map.
- Activation Functions. ReLU, Mish, TanhExp.
- Methods for approximating activation functions: polynomial approximation technique (piecewise linear approximation) or quadratic approximation.
- Look-up-table or coordinate rotation digital computer (CORDIC).
- Pooling layers.
- Fully-connected layers.
</aside>

<aside>
💡 Compare depthwise separable convolution and conventional convolution. Talk about its advantages and disadvantages.

- DSC is divided into a depthwise convolution (Dwcv) and a pointwise convolution (Pwcv).
</aside>

<aside>
💡 This work uses float-point format for representing a real number.

</aside>

### Enhanced Architecture for Real-World Application: Ad-MobileNet Model

<aside>
💡 The Baseline MobileNet Model. MobileNet-V1 uses DSC units, so it has small size. The main idea of the DSC unit is to havedifferent layers for combining and different layers for filtering. To further explain this,each input channel needs to be filtered separately in Dwcv, followed by Pwcv. In addition,the objective behind using Pwcv is to linearly combine all outputs of Dwcv.

</aside>

<aside>
💡 The Proposed Model: Ad-MobileNet. Three changes compared to the baseline model: 1) DSC unit —> Ad-depth unit; 2) using various types of activation functions, including Mish, TanhExp, ReLU; 3) removing all of the extraneous layers.

Ad-depth is made up of two Pwcv on the edges and one Dwcv in the middle.

![Untitled](An%20Efficient%20FPGA-Based%20Convolutional%20Neural%20Netwo%2004850d75d6a24aa5b1965ac138636ab5/Untitled.png)

ReLU has dying ReLU problem. So, we use three activation functions: TanhExp, ReLU, and Mish. TanhExp(x) = x * Tanh(exp(x)); Mish(x) = x * Tanh(softplus(x)).

Dropping all the unnecessary layers.

Ablation study was conducted to evaluate the importance of three modifications to the recognition rate and computational cost of this model.

![Untitled](An%20Efficient%20FPGA-Based%20Convolutional%20Neural%20Netwo%2004850d75d6a24aa5b1965ac138636ab5/Untitled%201.png)

</aside>

### Hardware Implementation of Ad-MobileNet on FPGA

<aside>
💡 Implementing ReLU only requires registers (FF) and look-up tables (LUTs). Implementing Mish and TanhExp requires an approximation. Mish(X) = X * tanh(ln(1 + exp(X)); TanhExp(X) = X * tanh(exp(X)).  Implementing Mish and TanhExp on FPGA device needs one adder, one multiplier, and registers for each approach function.

</aside>

<aside>
💡 Hardware Implementation of Ad-Depth Unit. In this study, the depthwise unit is composed of:

- 32 slices of line buffer
- 32 slices of multiplier array
- adder
- normalization block
- an activation block

Depthwise convolution is performed by k×k convolution operationswith spatial proximity. When the input data are passed through the buffer in a row-major layout, sliding window selection is released by the line buffer on the input image.Furthermore, multiple rows of pixel values can be buffered for simultaneous access.

![Untitled](An%20Efficient%20FPGA-Based%20Convolutional%20Neural%20Netwo%2004850d75d6a24aa5b1965ac138636ab5/Untitled%202.png)

The pointwise convolution unit (Pwcv) is composed of:

- a PE unit (operate a 1 x 1 convolution)
- adder (add bias)
- batch-normalization block (Norm)
- activation function module
- ram
- registers

![Untitled](An%20Efficient%20FPGA-Based%20Convolutional%20Neural%20Netwo%2004850d75d6a24aa5b1965ac138636ab5/Untitled%203.png)

</aside>

<aside>
💡 Hardware Implementation of Max-Pooling Layer.

![Untitled](An%20Efficient%20FPGA-Based%20Convolutional%20Neural%20Netwo%2004850d75d6a24aa5b1965ac138636ab5/Untitled%204.png)

The Ad-MobileNet module uses the global average pooling (GAP) instead of the fully connected layer.

</aside>

### Experiment and Results

<aside>
💡 Experiments were conducted with four networks: baseline MobileNet, MAd-MobileNet, TAd-MobileNet, and RAd-MobileNet. The fpga device used is xc7vx980t of the Virtex-7 family. Image classification task on Cifar-10 dataset. 

Results analysis:

- The baseline MobileNet model has the lowest accuracy.
- MAd-MobileNet and TAd-MobileNet outperform the two other models.
- The RAd-MobileNet has the lowest number of hardware resources.

Hardware resources needed for implementing four network models on fpga device.

![Untitled](An%20Efficient%20FPGA-Based%20Convolutional%20Neural%20Netwo%2004850d75d6a24aa5b1965ac138636ab5/Untitled%205.png)

</aside>

### Discussion

<aside>
💡 Comparison of four network models.

![Untitled](An%20Efficient%20FPGA-Based%20Convolutional%20Neural%20Netwo%2004850d75d6a24aa5b1965ac138636ab5/Untitled%206.png)

</aside>

### Conclusions

<aside>
💡 In this study, we intended to build a network with a few hardware resources whileproviding a high classification rate without losing information and implementing it onan FPGA.

</aside>

# Notes

- Field Programmable Gate Array.
    
    [Field-programmable gate array - Wikipedia](https://en.wikipedia.org/wiki/Field-programmable_gate_array)
    
- Shallow and lightweigt CNN models: MobileNet, ShuffleNet.
- Ad-MobileNet, based on MobileNet. It keeps a balance of low computational cost and high accuracy.
- Ad-depth unit. It replaces the traditional depthwise separable convolution.
- Three advantages:
    - More suited for hardware implementation without significantly increasing the computational cost.
    - Layers having a negligible negative impact on the recognition rate have been removed.
    - More advanced activation functions are compatible.
- Activation functions. Sigmoid, ReLU, Tanh, Tanh Exponential (TanhExp), Mish.
- Floating-point representation.
- Piecewise linear method (PWL) is used to approximate nonlinear activations.
    
    [Piecewise linear function - Wikipedia](https://en.wikipedia.org/wiki/Piecewise_linear_function)
    
- GoogLeNet, DenseNet, VGG, ResNet.
- MobileNet, ShuffleNet.
- Convolution output size:
    
    ![Untitled](An%20Efficient%20FPGA-Based%20Convolutional%20Neural%20Netwo%2004850d75d6a24aa5b1965ac138636ab5/Untitled%207.png)
    
- The main role of the activation function is to introduce nonlinearity into the output of a neuron. Without an activation function, a neuron network is simply a linear regression model that does not know how to solve complex tasks such as image recognition.
- CORDIC
    
    [CORDIC - Wikipedia](https://en.wikipedia.org/wiki/CORDIC)
    
- Pooling layer helps to dimish the spatial size of the convolved feature. It aids in the reduction of feature dimensions while extracting the dominant characteristics to achieve a significant recognition rate.
- Fully-connected layer maps feature maps into output labels.
- DSC can reduce the number of parameters and network calculations, but this also decreases the accuracy.
    
    ![Screen Shot 2022-07-13 at 1.28.37 PM.png](An%20Efficient%20FPGA-Based%20Convolutional%20Neural%20Netwo%2004850d75d6a24aa5b1965ac138636ab5/Screen_Shot_2022-07-13_at_1.28.37_PM.png)
    
- Dwcv uses a filter on each input channel to create the same number of outputs. Pwcv is a version of the standard convolution that uses a 1x1 filter.
- MACC mathematical operation. In computing, especially digital signal processing, the multiply–accumulate (MAC) or multiply-add (MAD) operation is a common step that computes the product of two numbers and adds that product to an accumulator. The hardware unit that performs the operation is known as a multiplier–accumulator (MAC unit); the operation itself is also often called a MAC or a MAD operation.

# Star References

- MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
    
    [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
    
- A High-Performance CNN Processor Based on FPGA for MobileNets
    
    [A High-Performance CNN Processor Based on FPGA for MobileNets](https://ieeexplore.ieee.org/document/8891993)
    
- ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
    
    [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)
    
- Rectified Linear Units Improve Restricted Boltzmann Machines
    
    [](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf)
    
- TanhExp: A Smooth Activation Function with High Convergence Speed for Lightweight Neural Networks
    
    [TanhExp: A Smooth Activation Function with High Convergence Speed for Lightweight Neural Networks](https://arxiv.org/abs/2003.09855)
    
- Mish: A Self Regularized Non-Monotonic Activation Function
    
    [Mish: A Self Regularized Non-Monotonic Activation Function](https://arxiv.org/abs/1908.08681)
    

# Academic Vocabulary

| proximity |  |  |
| --- | --- | --- |