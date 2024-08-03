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
