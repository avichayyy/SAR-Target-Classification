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

<p>
<span style="display:flex; justify-content:center;">R(s) = \sqrt(R<sub>0</sub>² + v²s²)</span>
</p>

<p>
where R<sub>0</sub> denotes the closest distance of the radar to the scatterer, v denotes the radar's velocity, and s denotes time along the flight path (also known as slow time). At the closest point, s, the time will be zero. In a good approximation for remote sensing radar, we can assume that vs≪R<sub>0</sub> (note that this may not hold for the general case, but the basic principle remains the same). In this case, we can approximate the range as a function of slow time using the Taylor approximation:
</p>

<p>
<span style="display:block; text-align:center;">R(s) ≈ R<sub>0</sub> + (v²s²)/(2R<sub>0</sub>)</span>
</p>

<p>
The phase of the range compression is:
</p>

<p>
<span style="display:block; text-align:center;">ϕ(s) ≈ -(4πR<sub>0</sub>)/λ - (2πv²s²)/(R<sub>0</sub>λ)</span>
</p>

<p>
The instantaneous frequency of the signal is:
</p>

<p>
<span style="display:block; text-align:center;">f(s) = (1/2π) ∙ ∂ϕ(s)/∂s = -(2v²s)/(R<sub>0</sub>λ)</span>
</p>

<p>
This is the pattern of an ultra-short pulse (linear chirp pattern). To find the bandwidth of this signal, we need to determine the maximum time we can use in signal processing. This maximum time is called the "integration time," and it is determined by the time during which the scatterer is within the antenna beam. For an antenna of physical length L, the half-power horizontal beamwidth is:
</p>

<p>
<span style="display:block; text-align:center;">θ<sub>a</sub> = λ/L,</span>
</p>

<p>
so the scatterer at the closest distance R<sub>0</sub> is illuminated for a duration of:
</p>

<p>
<span style="display:block; text-align:center;">s<sub>tot</sub> = (λR<sub>0</sub>)/Lv.</span>
</p>

<p>
Half of this time occurs as the radar approaches the closest range, and the other half occurs as it moves away from the closest range. Therefore, the bandwidth of the signal, which is the bandwidth of the signal in Synthetic Aperture Radar, is:
</p>

<p>
<span style="display:block; text-align:center;">B<sub>D</sub> = 2v/L.</span>
</p>

<p>
If this signal is filtered using a suitable filter, the compressed signal obtained will have a time width of 1/B<sub>D</sub>. Since the radar platform moves at velocity v, this will result in the flight direction resolution being described by:
</p>

<p>
<span style="display:block; text-align:center;">Δ<sub>a</sub> = v/B<sub>D</sub> = L/2.</span>
</p>

<p>
The combination of all the above results in the conclusion that the azimuth resolution (or flight direction resolution) for Synthetic Aperture Radar is equal to half the physical antenna size and is independent of the distance between the sensor and the ground! At first glance, this result may seem strange, as it implies that a smaller antenna provides better resolution. This can be explained as follows: the smaller the physical antenna, the larger its footprint, allowing for longer observation time for each point on the ground (i.e., a longer synthetic array can be synthesized). A longer synthetic array allows for a broader Doppler frequency bandwidth and, therefore, finer ground resolution. Similarly, if the distance between the sensor and the ground increases, the physical footprint increases, leading to longer observation time and broader Doppler frequency bandwidth, which balances the increase in distance. This result is crucial and allows the radar to be mounted on satellites in addition to aircraft.
</p>
