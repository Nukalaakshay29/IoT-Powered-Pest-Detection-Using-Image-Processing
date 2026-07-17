<p align="center">
    <img src="https://github.com/Amrita-TIFAC-Cyber-Blockchain/.github/blob/main/profile/img/AVV_CYS_Logo.png" alt ="Amrita TIFAC" width="700" />
</p>

# 24CYS333 – Internet of Things

## IoT#17 – IoT-Powered Pest Detection and Control System Using Image Processing

Pest infestation and plant diseases are major contributors to reduced agricultural productivity and economic loss. Early detection is critical for minimizing damage, yet current monitoring practices largely depend on manual field inspection, which is time-consuming, labor-intensive, inconsistent, and often reactive rather than preventive. By the time visible symptoms are noticed, crop damage may already be significant.

Existing automated pest and disease detection systems typically fall into two categories. The first relies solely on environmental sensors such as temperature, humidity, and soil moisture to estimate risk levels, but these systems lack visual confirmation and often produce inaccurate or delayed assessments. The second category uses deep learning–based image classification models; however, many of these solutions depend on computationally heavy architectures and cloud-based processing, increasing latency, infrastructure cost, and dependency on continuous internet connectivity. Such systems are not practical for small or medium-scale agricultural setups.

There is a clear need for a low-cost, real-time, and scalable pest detection system that combines visual analysis with environmental data and performs efficient local processing without relying entirely on cloud infrastructure.

This project proposes an IoT-integrated pest detection system that utilizes an ESP32 camera module along with environmental sensors to capture real-time crop data. The visual data is processed using a lightweight convolutional neural network (TinyLite) inspired by MobileNetV2 architecture principles, incorporating efficient depthwise separable convolutions to reduce computational complexity while maintaining classification performance. The trained model runs on a local computing system to classify plant conditions as “Healthy” or specific disease categories. The results, including predicted class, confidence score, and recommended control actions, are displayed on a dashboard. Upon detecting a disease condition, the system generates an alert signal to the ESP32 to trigger a visual indicator.

The proposed system aims to provide early detection, reduce dependency on manual inspection, minimize infrastructure cost, and enable timely intervention to prevent large-scale crop damage.

---

## Hardware Requirements

| Availability | Component                     | Specification / Model     | Purpose                                  |
|--------------|--------------------------------|----------------------------|------------------------------------------|
|       ✅       | ESP32-CAM Module              | AI Thinker ESP32-CAM       | Captures plant images                    |
|       ✅       | Leaf Wetness Sensor          | Capacitive Sensor          | Measures Leaf Wetness level             |
|       ✅      | Humidity Sensor              | DHT11 / DHT22              | Monitors environmental conditions        |
|        ✅      | LED Indicator (RED AND GREEN) | 5mm LED                    | Visual alert for disease detection       |
|        ✅      | Resistors & Jumper Wires      | Standard                   | Circuit connections                      |
|       ✅       | Power Supply                  | 5V Adapter / USB           | Powers ESP32 module                      |
|      ✅        | Laptop / PC                   | Minimum 12GB RAM            | Runs model and dashboard              |
|      ✅       | Arduino UNO                   |  BASIC ONE                 | File Upload to the ESP32 CAM MODULE     |
|      ✅           |  TTL to RS-485 transceiver |    MAX485E           | translator between your ESP32-CAM and the industrial JXCT sensor |


**NOTE: LEAF WETNESS SENSOR IS TAKEN FROM THE DEPARTMENT**

---

## Software Requirements

| Availability | Software / Tool     | Version (Recommended) | Purpose                                      |
|--------------|---------------------|------------------------|----------------------------------------------|
|  ✅            | Python              | 3.9+                   | Core programming language                    |
|  ✅            | TensorFlow / Keras  | 2.x                    | Model training and inference                 |
|  ✅            | NumPy               | Latest                 | Numerical computations                       |
|  ✅           | Matplotlib/Seaborn  | Latest                 | Visualization of training metrics            |
|  ✅           | Kaggle API          | Latest                 | Dataset download                             |
|  ✅           | KAGGLE RESOURCES(GPU)| Latest                 | Model training environment                   |
|  ✅            | Arduino IDE         | Latest                 | ESP32 firmware programming                   |
|  ✅            | Streamlit / Flask   | Latest                 | Dashboard interface                          |

---

### Literature Survey
**BY AKSHAY**
- Edge AI enables real-time agricultural monitoring without continuous cloud dependency, reducing latency and improving response speed in low-connectivity rural environments.
- Hybrid edge–cloud architecture improves system reliability by performing fast local inference while using the cloud for storage, analytics, and long-term monitoring.
- Environmental sensing (temperature, humidity, soil moisture) provides critical context for crop health and significantly improves prediction reliability.
- CNN-based image analysis effectively detects visible disease symptoms such as spots, discoloration, and texture variations in plant leaves.
- Image-only models are reactive, while combining sensor data with image features enables predictive and context-aware disease detection.
- Multimodal data fusion (sensor + image) reduces false positives and improves overall classification accuracy in real-world agricultural conditions.
- Image preprocessing techniques such as noise removal, normalization, and segmentation are essential for improving feature extraction and model performance.
- Dataset quality and diversity strongly influence deep learning model accuracy and robustness.
- Continuous IoT-based monitoring allows early disease detection, reducing crop damage and minimizing manual inspection.
- Transmitting processed insights instead of raw sensor/image data optimizes bandwidth, reduces power consumption, and improves system efficiency.
- Smart agriculture systems focus on prediction and early intervention rather than simple disease detection, enabling targeted pesticide use and sustainable farming.
- Energy-efficient lightweight AI models are necessary for deployment on resource-constrained edge devices such as IoT nodes and embedded systems.

---

**BY Govardhan**

- IoT-based pest detection systems integrate sensors and image processing to enable real-time monitoring of agricultural fields.
- PIR sensors detect pest movement based on infrared heat radiation, enabling automatic activation of control mechanisms.
- Image processing techniques such as grayscale conversion, background subtraction, and pixel comparison improve pest detection accuracy.
- K-Means clustering and feature extraction methods enhance pest identification and classification from captured images.
- Environmental sensing (temperature, humidity, soil moisture, CO₂, light intensity) provides critical context for understanding pest growth patterns.
- Raspberry Pi and STM32 microcontrollers are widely used as central controllers for data processing and device management.
- Cloud-based data storage using MQTT protocol enables remote monitoring, historical data analysis, and seasonal pest prediction.
- Automated pest density estimation reduces manual counting errors and improves monitoring efficiency in large-scale farms.
- Physical pest control methods such as ultrasonic waves, ozone sterilization, and LED light traps reduce dependency on chemical pesticides.
- Wireless communication technologies (Wi-Fi, GPRS, ZigBee) support real-time data transmission and multi-device coordination.
- Noise filtering and image preprocessing techniques significantly improve feature extraction and model performance.
- Continuous IoT-based monitoring allows early pest detection, minimizing crop damage and improving yield.
- Integrated edge-device processing reduces bandwidth usage by transmitting processed insights instead of raw image data.
- Eco-friendly pest management approaches promote sustainable agriculture and reduce environmental pollution.
- Smart agricultural systems focus on automation, predictive analysis, and remote management to enhance productivity and reduce labor cost.
---


**BY TAGORE**

- IoT-based plant disease detection systems enable real-time crop monitoring without manual field inspection.
- Environmental sensors (temperature, humidity, soil moisture) provide essential context for accurate disease prediction.
- Camera modules capture leaf images periodically for automated visual analysis of plant health.
- Image processing techniques detect visible symptoms such as spots, lesions, discoloration, and texture changes.
- Convolutional Neural Networks (CNNs) improve disease classification accuracy by learning complex visual patterns.
- Combining sensor data with image features enables predictive and context-aware disease detection.
- Multimodal data fusion (sensor + image) reduces false positives and improves reliability in real field conditions.
- Edge computing allows local processing on devices like Raspberry Pi, reducing latency and internet dependency.
- Cloud integration supports long-term storage, analytics, and remote monitoring of crop health data.
- Automated alert systems notify farmers immediately when disease symptoms are detected.
- Continuous monitoring enables early disease detection, reducing crop damage and yield loss.
- Dataset quality and diversity significantly affect machine learning model performance and robustness.
- Image preprocessing techniques such as noise removal, normalization, and segmentation improve detection accuracy.
- Smart agriculture systems focus on early intervention rather than reactive treatment after severe infection.
- Energy-efficient lightweight AI models are necessary for deployment on resource-constrained IoT devices.

---
### Proposed Solution

The proposed system is a real-time IoT-based pest and plant disease detection platform that integrates image capture, environmental sensing, and a lightweight deep learning model for efficient crop health monitoring.

The system uses an ESP32 module with an integrated camera and environmental sensors to capture plant images along with humidity and soil moisture data. The collected data is transmitted to a laptop over WiFi, where a lightweight TinyLite CNN model (inspired by MobileNetV2) performs image classification. The system then displays results on a dashboard and triggers an alert if a disease is detected.


### Solution Flow

```
+-------------------------------------------+
| ESP32 + Camera Module                    |
| - Captures Leaf Images                   |
| - Reads Humidity & Leaf Moisture         |
+-------------------------------------------+
                    |
                    |  WiFi Transmission
                    v
+-------------------------------------------+
| Laptop / Local Processing System         |
| - TinyLite CNN Model                     |
| - Image Preprocessing                    |
| - Disease Classification                 |
+-------------------------------------------+
                    |
                    v
+-------------------------------------------+
| Dashboard Interface                      |
| - Disease Name                           |
| - Confidence Score                       |
| - Affected Area (%)                      |
| - Recommended Control Actions            |
+-------------------------------------------+
                    |
                    v
+-------------------------------------------+
| LED Alert via ESP32                      |
| (Activated if Disease Detected)          |
+-------------------------------------------+
```


### Working Principle

1. The ESP32 module captures real-time plant leaf images.
2. Environmental sensors measure humidity and Leaf moisture levels.
3. Image and sensor data are transmitted to the laptop via WiFi.
4. The TinyLite CNN model processes the image and classifies it as “Healthy” or a specific disease.
5. The system calculates the confidence score and estimates the affected area percentage.
6. Results are displayed on the dashboard with recommended control measures.
7. If a disease condition is detected, a signal is sent back to the ESP32 to activate the LED alert.


### Key Design Features

- Lightweight TinyLite CNN inspired by MobileNetV2 for efficient computation  
- Local processing to reduce cloud dependency and latency  
- Real-time WiFi-based data transmission  
- Automated alert mechanism for quick response  
- Scalable and cost-effective IoT architecture  

This solution enables early pest detection, faster intervention, and reduced crop loss through intelligent and efficient monitoring.


#### Architecture Diagram
![Architecture Diagram](./images/Architecture_Diagram.png)

---

#### Usecases
- Real-time pest and plant disease detection using ESP32 camera and TinyLite CNN model.
- Early warning system for farmers through dashboard alerts and LED indication.
- Smart greenhouse monitoring by combining image analysis with humidity and moisture data.
- Decision support system providing disease name, confidence score, and recommended control actions.
- Low-cost IoT-based crop health monitoring solution without full cloud dependency.
- Educational and research prototype for lightweight deep learning in agriculture.

---

#### Deliverables

- Working ESP32-CAM module for real-time plant image capture  
- Integrated Leaf moisture and humidity sensing system  
- TinyLite CNN model (MobileNetV2-inspired) for disease classification  
- Trained and saved model file (.keras) for deployment  
- Real-time image inference pipeline  
- Dashboard displaying disease name, confidence score, and control recommendations  
- LED-based alert mechanism triggered on disease detection  
- Complete end-to-end system integration (ESP32 → Model → Dashboard & Alert)


---

### Mapping the Project to Relevant Sustainable Development Goals (SDGs)

| SDG No. | SDG Title                                      | Project Alignment                                                                 |
|:------|-----------------------------------------------|-----------------------------------------------------------------------------------|
| SDG 2  | Zero Hunger                                   | Pest attacks reduce crop yield; early detection minimizes crop loss and improves food security |
| SDG 12 | Responsible Consumption and Production        | Reduces excessive pesticide use through targeted pest control, lowering chemical waste |
| SDG 9  | Industry, Innovation and Infrastructure       | Uses AI/ML and IoT technologies to bring innovation and modern infrastructure to agriculture |


---

### Collaboration

| Team Member | Module & Scope | Contribution |
|:-----------:|----------------|:-------------|
|             |                |              |

---

### References
- [AI and IoT-powered edge device optimized for crop pest and disease detection](https://www.nature.com/articles/s41598-025-06452-5)
- [Pest detection and extraction using image processing techniques](https://www.ijcce.org/papers/317-CS038.pdf)
- [Raspberry Pi Camera-Based Leaf Pest Detection By Using Classical Image Processing With CNN Algorithm](https://ieeexplore.ieee.org/document/10895083)


















































