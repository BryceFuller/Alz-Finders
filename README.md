# Alz-Finders
Predicting the onset of Alzheimer's Disease using MRI &amp; PET scans. 

# Final Project Planning

# <font color='red'>First Steps:</font> 

**For every step, make certain that you document your steps and save your code in a jupyter notebook. **
  
```diff
- (0)
```
* GET ACCESS TO ADNI (Ghosh has been emailed)

* Figure out how the data is formatted and how we access it

* If neccessary, write a pice of code which reformats or gives access to the Dataset

* Research how the authors removed pixels outside the brain (CHECK INTO MRIQC)

* Find tool to downsample images to 128x128x128 voxels

* Find (or write) tool to segment 3D images into 27 blocks



### <font color='red'>(1) </font> 
* Understand how to the Kernel Filters are learned

* Find a robust package for 3D CNN's, ideally with the flexibility to use Lui's parameters

* Optional but probably neccesary: Figure out a clean way to distribute the training of these models (all 27 3D-CNN's can be trained in parallel)


### <font color='red'>(2) </font> 
* This one seems eerily straightfoward to me, I guess we need to write some infrastructure code to collect the outputs of all 27 models as input to the fully connected layer. 

# Process Flow From:  
#### Multi-Modality Cascaded Convolutional Neural Networks for Alzheimer’s Disease Diagnosis 

### (0)    Preprocessing
* Raw images are 256x256x**54**, Lui resamples to 256x256x**256** at a density of 1mm$^3$

    * https://fsl.fmrib.ox.ac.uk/.

* Downsample both the MRI and PET images into 128x128x128 images.

* Remove pixels outside of brain. Remaining images are size 100x81x80

* Slice Images into 3x3x3 = 27 blocks of size 50x41x40 (the blocks overlap in all directions)



### (1)   3D CNN's for feature extraction

* Both MRI and PET Images are fed into separate 3D CNN models for feature extraction

>"The deep
3D CNN is built by alternatively stacking convolutional and
sub-sampling layers to hierarchically learn the multi-level features
of multi-modality brain images, which is followed by the
fully connected and softmax layers for image classification..." 

> Page: 300, Section: Feature Learning with 3D CNNs, End of First Paragraph

* Apply Learned Kernel Filters to Images. 


>In our implementation, each
deep CNN is built by stacking 4 convolutional layers, 3 max
pooling layers, a fully connected layers and a softmax layer.
The sizes of all convolutional filters are set to 3 × 3 × 3 and the
numbers of filters are set to 15, 25, 50, 50 for 4 convolution
layers, respectively. Max pooling is applied for each 2 × 2 × 2
region. Tanh function is adopted as the activation function in
these layers. The 3D convolutional kernels are randomly initialized
in the Gaussian distribution. The other trainable parameters
of the networks are tuned using the standard backpropagation
with stochastic gradient descent by minimizing
the loss of cross entropy. In addition, the dropout strategy is
employed to reduce the co-adaption of intermediate features
and overfitting problem, and improve the generalization
capability. 

> Page: 301, Section: Feature Learning with 3D CNNs, End of Last Paragraph

### (2)   Multi-Modality Cascaded CNNs 

Purpose: We want to build a model combines the features learned from both the PET scan and MRI scans.



* For each of the 27 blocks, we will run the corresponding MRI and PET outputs through a series of 2D CNN's.

* The outputs of all 27 2D-CNN's are then fed into a fully-connected layer followed by a softmax layer.

* The output of the softmax layer is the class likelihoods. 
