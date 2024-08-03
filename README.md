# SAR-Target-Classification
Repository for SAR Target Classification - Project A at the ECE Faculty at the Technion.<br>
This project is conducted by Avichay Ashur, with the supervision of Dr. Meir Bar zohar. <br>

![image](https://github.com/user-attachments/assets/ba95e305-9f0f-4230-b926-a57d26ebc99b)

# About
This project’s goal is to develop a classifier based on convolution neural network to classify Synthetic Aperture Radar (SAR) targets using deep learning.
Deep learning is a powerful technique that can be used to train robust classifier. It has shown its effectiveness in diverse areas ranging from image analysis to natural language processing. These developments have huge potential for SAR data analysis and SAR technology in general, slowly being realized. A major task for SAR-related algorithms has long been object detection and classification, which is called automatic target recognition (ATR). 
To illustrate this workflow, we will use the Moving and Stationary Target Acquisition and Recognition (MSTAR) Mixed Targets dataset published by the Air Force Research Laboratory [1]. Our goal is to develop a model to classify ground targets based on SAR imagery.

# Background

<p>
Deep neural networks, particularly those related to machine learning, simulate the structure and function of the human brain's neurons for the purpose of data processing and pattern recognition. These networks are composed of multiple layers of neurons organized hierarchically. Each layer is responsible for processing information and passing it on to the next layer. The ability of deep neural networks to learn complex, multi-layered representations of data makes them a powerful tool across a wide range of fields, including image recognition, natural language processing, speech recognition, and more.
</p>

<p>
The development of deep neural networks has resulted from the combination of large, accessible data repositories, improved algorithms for learning, and increased computational power. These networks "learn" by updating their parameters to minimize a specific error function, enabling them to handle various problems effectively.
</p>

<p>
In the realm of Synthetic Aperture Radar (SAR), a crucial challenge is the detection and classification of objects, commonly known as Automatic Target Recognition (ATR). Before diving into the details of ATR, it's important to first understand the fundamental principles of SAR.
</p>

<p>
Synthetic Aperture Radar (SAR) is a type of imaging radar that generates images of the Earth's surface. Unlike optical imaging systems, SAR operates by emitting radio waves and recording the reflected signals, making it capable of imaging the Earth in all weather conditions, day or night.
</p>

<p>
Traditional imaging systems, such as those operating in the visible or infrared spectrum, rely on lenses or mirrors to project the light onto a two-dimensional array of detectors, creating an image. The spatial resolution of these systems depends on the distance between the camera and the scene being imaged. In contrast, <b>SAR imaging does not depend on the distance between the radar and the target.</b>
</p>

<p>
SAR works by moving the radar antenna along a flight path and collecting data over time. The motion of the antenna allows SAR to simulate a large antenna aperture, leading to high-resolution images. The key advantage of SAR is its ability to produce detailed images of the Earth's surface, regardless of weather conditions or lighting, making it an invaluable tool for remote sensing applications.
</p>

<p>
The principle of SAR involves combining the radar signals collected over a synthetic aperture to create an image. This process includes several steps: transmitting radio waves, collecting the reflected signals, and processing the data to produce a final image. The quality of the image depends on various factors, including the pitch angle, altitude, and velocity of the radar platform. SAR also employs advanced signal processing techniques to enhance image resolution and reduce noise.
</p>

<p>
SAR technology is widely used in various fields, including environmental monitoring, disaster management, and military reconnaissance. Its ability to provide high-resolution images under all weather conditions makes it a powerful tool for detecting and analyzing surface features on the Earth.
</p>

### Basic Principles of SAR

<p>
The basic operation of Synthetic Aperture Radar (SAR) can be outlined in the following stages:
</p>

<ol>
  <li>
    <strong>Transmission:</strong> The radar transmits radio waves at frequencies ranging from 100 kHz to 10 GHz, depending on the radar's operational requirements.
  </li>
  <li>
    <strong>Signal Reception:</strong> The radar collects the reflected signals that return from the target on the ground.
  </li>
  <li>
    <strong>Data Collection:</strong> The radar processes the collected reflections in terms of time and space.
  </li>
  <li>
    <strong>Signal Processing:</strong> At this stage, advanced signal processing techniques and algorithms are employed to reduce noise and improve the quality of the collected data.
  </li>
</ol>

<p>
As we have seen in the basic principles of imaging radar, to successfully process and collect the data, it is necessary to know the following parameters:
</p>

<ol>
  <li>Target azimuth - Pitch Angle.</li>
  <li>Flight altitude - Z Height.</li>
  <li>Flight velocity vector.</li>
</ol>

<p>
To enhance the quality of the final image, reflections are collected over time and space, a principle known as Multi-view SAR. This principle is illustrated in the figure bellow.
</p>

![image](https://github.com/user-attachments/assets/4ea68ab9-7075-4415-84e0-d44d4faa89e0)

<h3>Advantages and Disadvantages of Synthetic Aperture Radar (SAR)</h3>

<h4>Advantages of SAR:</h4>
<ol>
  <li>The radar can be mounted on any moving platform, particularly aircraft or satellites.</li>
  <li>SAR is capable of producing very high-resolution images.</li>
  <li>SAR is minimally affected by weather conditions (heavy rain may reduce image quality due to unwanted reflections caused by the transmitted waves hitting the raindrops).</li>
  <li>SAR can operate both during the day and at night.</li>
  <li>The resolution is not dependent on flight altitude (a mathematical development and explanation for this is provided in section 1.4).</li>
</ol>

<h4>Disadvantages of SAR:</h4>
<ol>
  <li>It takes a long time (minutes to hours) to produce a single image because multiple reflections must be collected over time to generate an image.</li>
  <li>The image quality is dependent on environmental characteristics—the image depends on the reflectivity coefficient of the objects, so the image characteristics will vary depending on the reflective environment.</li>
  <li>Several motion characteristics of the radar (altitude, velocity vector, and target angle) must be known to interpret the signals.</li>
  <li>SAR only works on static targets.</li>
  <li>Interpreting the image is challenging without special training and general knowledge of imaging characteristics. It should be noted that the goal of this project is to attempt to mitigate this disadvantage.</li>
  <li>Extensive calculations are required to process the received data in order to produce a single image.</li>
</ol>

<h3>Synthesis of Synthetic Aperture Radar (SAR)</h3>

<p>
The synthesis of SAR refers to a specific application of imaging radar systems, which uses the movement of the radar platform and special signal processing to create high-resolution images. Before the discovery of SAR synthesis, imaging radars operated using the principle of real aperture and were known as Side-Looking Airborne Radars (SLAR). Karl Wiley from Goodyear Aircraft Corporation is considered the first researcher to describe the use of Doppler frequency analysis of signals from a coherent radar in motion to improve resolution in the flight direction. Wiley noted that two targets at different positions in the flight direction would be at different angles relative to the aircraft's velocity vector, resulting in different Doppler frequencies (the Doppler effect is the phenomenon that causes a change in the pitch of a vehicle's horn as it passes by a stationary observer). Using this effect, targets in the flight direction can be separated based on their different Doppler frequencies. This technique was originally known as Doppler beam sharpening, but later became known as Synthetic Aperture Radar.
</p>

<p>
The main difference between real and synthetic aperture radar is the way azimuthal resolution is achieved. The radar range resolution equation derived earlier for real aperture radar still applies here. However, the imaging mechanism in the flight direction and the resolution obtained in the flight direction differ in the cases of real and synthetic aperture radar.
</p>

![image](https://github.com/user-attachments/assets/04edd113-1c26-4031-9ae2-66e7be78b23c)


<p>
As the radar moves along the flight path, it transmits pulses of energy and records the reflected signals, as shown in the figure above. When processing the radar data, the radar platform's position is taken into account when summing the signals to combine the energy in the flight direction. As the radar moves along the flight path, the distance between the radar and the target changes, with the minimum distance occurring when the scatterer is exactly parallel to the radar platform. The phase of the radar signal is given by: 4πλR(s). The change in distance between the radar and the scatterer means that after range compression, the phase of the signal will differ for different positions along the flight path.
</p>

<p>
The change in distance can be described as:
</p>

<p align="center">
$\ R(s) = \sqrt{R_0^2 + v^2s^2} $
</p>

<p>
where R<sub>0</sub> denotes the closest distance of the radar to the scatterer, v denotes the radar's velocity, and s denotes time along the flight path (also known as slow time). At the closest point, s, the time will be zero. In a good approximation for remote sensing radar, we can assume that vs≪R<sub>0</sub> (note that this may not hold for the general case, but the basic principle remains the same). In this case, we can approximate the range as a function of slow time using the Taylor approximation:
</p>


<p align="center">
$\ R(s) ≈ R_0 + \frac{v^2s^2}{2R_0} $
</p>

<p>
The phase of the range compression is:
</p>

<p align="center">
$\ ϕ(s) = -\frac{4πR(s)}{\lambda} ≈ -\frac{4\pi R_0}{\lambda} - \frac{2\pi v^2s^2}{R_0\lambda} $
</p>

<p>
The instantaneous frequency of the signal is:
</p>

<p align="center">
$\ f(s) = -\frac{1}{2\pi} \cdot \frac{∂ϕ(s)}{∂s} = -\frac{2v^2s}{R_0\lambda} $
</p>

<p>
This is the pattern of an ultra-short pulse (linear chirp pattern). To find the bandwidth of this signal, we need to determine the maximum time we can use in signal processing. This maximum time is called the "integration time," and it is determined by the time during which the scatterer is within the antenna beam. For an antenna of physical length L, the half-power horizontal beamwidth is: θ<sub>a</sub> = $\ \frac{λ}{L}$
so the scatterer at the closest distance R<sub>0</sub> is illuminated for a duration of:
</p>

<p align="center">
$\ s_{tot} = \frac{\lambda R_0}{Lv}$
</p>

<p>
Half of this time occurs as the radar approaches the closest range, and the other half occurs as it moves away from the closest range. Therefore, the bandwidth of the signal, which is the bandwidth of the signal in Synthetic Aperture Radar, is:
</p>

<p align="center">
$\ B_D = \frac{2v}{L}$
</p>

<p>
If this signal is filtered using a suitable filter, the compressed signal obtained will have a time width of 1/B<sub>D</sub>. Since the radar platform moves at velocity v, this will result in the flight direction resolution being described by:
</p>

<p align="center">
$\ Δ_a = \frac{v}{B_D} = \frac{L}{2}$
</p>

<p>
The combination of all the above results in the conclusion that the azimuth resolution (or flight direction resolution) for Synthetic Aperture Radar is equal to half the physical antenna size and is independent of the distance between the sensor and the ground! At first glance, this result may seem strange, as it implies that a smaller antenna provides better resolution. This can be explained as follows: the smaller the physical antenna, the larger its footprint, allowing for longer observation time for each point on the ground (i.e., a longer synthetic array can be synthesized). A longer synthetic array allows for a broader Doppler frequency bandwidth and, therefore, finer ground resolution. Similarly, if the distance between the sensor and the ground increases, the physical footprint increases, leading to longer observation time and broader Doppler frequency bandwidth, which balances the increase in distance. This result is crucial and allows the radar to be mounted on satellites in addition to aircraft.
</p>

### Speckle Noise in SAR Images
<p>
In their article, T. Zheng et al. examine the modeling of speckle noise in Synthetic Aperture Radar (SAR) images by demonstrating the effects of different noise levels, represented by various 𝐿 values, on the original images. They present the equations used to model this noise as follows:

<p align="center">
$\ p(u) = L^L\cdot\frac{u^{L-1}}{\Gamma (L)}\cdot e^{-Lu}$
</p>

The figures provided at the end of the article illustrate how SAR images are affected by varying levels of speckle noise, highlighting the differences in image quality and clarity as 𝐿 changes. This systematic approach helps in understanding and addressing speckle interference in SAR imagin
</p>
<div align="center">
<img src="https://github.com/user-attachments/assets/3d9943a3-510b-4c70-9e40-503c9fecd64d">
</div>
<p align="center">
(a) L=0.2 ; (b) L=1; (c) L=5 (d) Original SAR image
</p>

# MSTAR Dataset
<p>
MSTAR Dataset contain 4 different pitch angles {15,17,30,45}. For 15° and 17° there are 11 different classes, while 30° have 5 classes and 45° only have 4 classes.
this is shown in the following figure
</p>
<div align="center">
<img src="https://github.com/user-attachments/assets/ccfa40b8-fa85-4a88-878f-fd59ee51663e">
</div>

<p>
The common practice in literature is that the training set is always 17° – as it is the largest (see Table 1), while the test set is 15°. In some papers, tests are also conducted on larger imaging angles (30° and 45°). In these cases, it is common to keep the training set at 17°, and the test set at 30° and 45° as appropriate for the test.

Table bellow shows the number of samples for each class depending on the radar imaging angle in the dataset.
</p>

| Angle | ZIL131 | BMP2 | BTR60 | BTR70 | D7 | T62 | T72 | 2S1 | BRDM2 | ZSU234 | SLICY | Total |
|-------|--------|------|-------|-------|----|-----|-----|-----|-------|--------|-------|-------|
| 15°   | 274    | 195  | 195   | 196   | 274| 273 | 196 | 274 | 274   | 274    | 274   | 2,425 |
| 17°   | 299    | 233  | 256   | 233   | 299| 299 | 232 | 299 | 298   | 299    | 298   | 2,747 |
| 30°   | 0      | 0    | 0     | 0     | 0  | 0   | 288 | 288 | 420   | 406    | 288   | 1,402 |
| 45°   | 0      | 0    | 0     | 0     | 0  | 0   | 0   | 303 | 423   | 422    | 303   | 1,148 |
| Total | 573    | 428  | 451   | 429   | 573| 572 | 716 |1,164| 1,415 | 1,401  | 1,163 | 7,722 |

<p>
During this project I will work with a subset of the MSTAR dataset called 10 class MSTAR dataset. in this subset we exclude the class "SLICY".
This is because SLICY class have different data distribution (it is a bunker while the other classes are vehicles), i will further explain this decision in the Data Visualization section.
</p>

# Data Visualization
<p>
To gain a comprehensive understanding of the dataset, I plan to apply several dimensionality reduction algorithms:

1. **PCA (Principal Component Analysis)**: This linear method will be used first to reduce the data dimensions while retaining as much variance as possible.

2. **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Following PCA, t-SNE will help in visualizing clusters by preserving local data relationships.

This multi-step approach will enable a thorough exploration of the dataset’s underlying patterns and structures, allowing us to visualize it effectively in 2D.
As I will show, only the t-SNE gave satisfying results and therefore will be used mostly.
</p>

### PCA Non-Normalized 
<img src="https://github.com/user-attachments/assets/631ce0d4-760a-4511-a750-1c67b6880b36" alt="image" width="800" />

### PCA Normalized
<img src="https://github.com/user-attachments/assets/d26d630c-6d45-4f60-9210-b2d98f96517d" alt="image" width="800" />

<p>
As we can see, PCA both normalized and Non-Normalized didnt result a good image that can help us understand the distribution of the data so I decided to use t-SNE as well.
</p>

### t-SNE
<img src="https://github.com/user-attachments/assets/5ce3a861-6c12-482c-ae67-b7bd1bdd75c9" alt="image" width="800" />

<p>
The t-SNE algorithm resulted a more clear view of the dataset, as shown in the figure above, the SLICY have different distribution compared to the other classes. I will show in the result section, when training the network with the SLICY class, our results are improving because the data distribution is very distinct compared to the other classes, therefore on our choosen solution, we wont be using the SLICY class.
</p>

<p>
In order to show why in this project I will be focusinjg on 15° as the test set and 17° as training set, I will show the data distribution between the differentt angles.
</p>

<img src="https://github.com/user-attachments/assets/8d1aba30-babd-4623-b6b9-520dab5e4f71" alt="image" width="800" />

<img src="https://github.com/user-attachments/assets/993a3457-1608-4878-a2bb-0355fb357a5f" alt="image" width="800" />

<img src="https://github.com/user-attachments/assets/c43cb8c3-4b72-4fe6-b8ff-634d705ccd3c" alt="image" width="800" />

<p>
As we can see in the different figures, while the data distribution for 15° vs 17° seems to be very similiar, the bigger the difference in the angles, the more difference we see in the data distribution.
While the images in low dimensions for 15° vs 17° are very close to each other, when we look at  17° vs 30°/45°, the difference is getting bigger and it will be a very challanging task to learn on 17° and make predictions over 30°/45°.
</p>

# Results
### MATLAB Demo Network
<p>
As part of the initial project goals, I will run the MATLAB demo network for SAR image classification using deep learning, for the final results, I will rewrite the code in Python and compare the result of the given network to other solutions. The MATLAB demo architecture is:
</p>
<div align="center">
<img src="https://github.com/user-attachments/assets/30776c71-089c-475c-b344-a79f0dd9a756">
</div>
<p>
Each Block contain Conv2d layer with ReLU Activation -> Another Conv2d layer with ReLU activation -> Batch Norm -> Max Pooling 2x2
</p>
<p>
The initial MATLAB Demo split the dataset: training set - 80% ; Validation set - 10% ; Test set - 10%. This is not the standard that is used in the literature, the results are shown in the following table:
</p>

| Run # | 9    | 8    | 7    | 6    | 5    | 4    | 3    | 2    | 1    | 0    |
|-------|------|------|------|------|------|------|------|------|------|------|
| Accuracy | 97.6 | 98.0 | 97.7 | 98.3 | 95.7 | 97.7 | 97.6 | 97.1 | 97.8 | 98.3 |

Mean: 97.58% Accuracy | Variance: 0.49

Confusion Matrix:
<div align="center">
  <img src="https://github.com/user-attachments/assets/c4391f0f-f702-4ffd-9c28-97b50e022073" alt="image" width="600">
</div>

<p>
As I show above, the given network with the current dataset split is doing very well and able to classify the image with 97.58% average accuracy over 10 different seeds. though this is not the ussual data split, in the following table are the results when using 15° vs 17° as test and training.
</p>

| Run #   | 9    | 8    | 7    | 6    | 5    | 4    | 3    | 2    | 1    | 0    |
|---------|------|------|------|------|------|------|------|------|------|------|
| Accuracy| 83.1 | 86.9 | 86.0 | 74.4 | 80.4 | 76.1 | 82.0 | 88.6 | 81.6 | 83.9 |

Mean: 82.3% | Variance 18.3

Confusion Matrix:
<div align="center">
  <img src="https://github.com/user-attachments/assets/9e9155ec-b5fb-4f72-95c4-b5cca0e2efa5" alt="image" width="600">
</div>

### Classical Methods
<p>
As part of the testing I did, I've tried Two classical methods for this task. The first is Random Forest. This method results 70.72% Accuracy. <br>
The Second method I used is KNN, the results for this method is shown in the following table:  
</p>

<div align="center">
  
| Accuracy | K  |
|----------|----|
| 90.98    | 1  |
| 89.18    | 2  |
| 88.86    | 3  |
| 87.39    | 4  |
| 87.28    | 5  |
| 86.36    | 6  |
| 85.32    | 7  |
| 85.04    | 8  |
| 83.95    | 9  |
| 82.47    | 10 |
| 79.99    | 15 |
| 79.36    | 20 |
| 79.25    | 25 |
| 68.27    | 30 |
| 67.43    | 35 |

</div>

<p>
As we can see, the bigger the K, the lower the model accuracy is.
</p>

### Augmentations
<p>
The training set contain only 2,747 Images. in order to improve the network's accuracy I will try to use data augmentation in order to increase the training set size. I've tested different augmentations using the MATLAB Demo network (implemeted in python), the results are:
</p>

<div align="center">
  
| Accuracy | Data Augmentation                           |
|----------|---------------------------------------------|
| 62.60%   | Random Rotation (0,360) [10]                |
| 83.67%   | No Augmentation                             |
| 90.52%   | Gaussian Noise [11] – N(0,0.01)             |
| 96.21%   | Image Patching [12]                         |

</div>
