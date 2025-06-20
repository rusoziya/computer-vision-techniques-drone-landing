<<<<<<< HEAD
# Computer Vision and Sensing Techniques for Autonomous Drone Landing in Space Exploration
![ORB_Matching](https://github.com/user-attachments/assets/a1536f23-f220-435b-ad71-cf89fd6a198b)

## Overview
This project focused on developing robust computer vision and sensing algorithms to estimate the geometry, rotation, and motion of a suspended Astronomical Object (AO) model. The insights gained from this analysis can be applied to autonomous drone navigation and landing strategies in dynamic space exploration environments.

## Objectives
- Accurately segment and isolate the AO from images.
- Analyze the AO's geometric properties, such as its center and height above the ground.
- Estimate the AO's rotation cycle using video processing.
- Derive surface velocity parameters to guide drone navigation.

## Features
- **Image Segmentation**: Implemented HSV and Hough Transform-based methods, enhanced by combined approaches for improved accuracy.
- **Geometric Analysis**: Tracked the AO's centroid and estimated its height above the ground using stereo depth estimation.
- **Rotation Cycle Estimation**: Used ORB-based feature matching and RANSAC for robust rotation time detection.
- **Drone Navigation Parameters**: Calculated the AO's diameter and surface velocities as a function of latitude.

## Methodology
### Task 1: Image Segmentation
- **Techniques**: HSV segmentation, Hough Transform, and combined segmentation.
- **Evaluation**: Assessed segmentation accuracy using metrics like ROC curves, IoU, F1 score, and AUC.
<img width="480" alt="image" src="https://github.com/user-attachments/assets/2e5afa63-126f-425f-a132-f8c98a6e1d1e" />

### Task 2: Geometric Analysis
- **Center Tracking**: Extracted and tracked the AO's centroid over time, identifying sinusoidal motion patterns.
- **Height Estimation**: Applied stereo depth estimation with calibrated cameras to compute the AO's height.
<img width="640" alt="image" src="https://github.com/user-attachments/assets/5030fb92-afca-47d1-8340-fb5401cf710e" />

### Task 3: Rotation Cycle Estimation
- **Feature Matching**: Employed ORB and RANSAC to detect rotation peaks.
- **Real-Time Processing**: Automated rotation cycle estimation for dynamic inputs.
<img width="785" alt="image" src="https://github.com/user-attachments/assets/e96bc24c-79cd-4abc-adb1-0cd08ea31851" />


### Task 4: Drone Navigation Parameters
- **Diameter Measurement**: Used the Pinhole Camera Model to convert pixel measurements to real-world distances.
- **Velocity Calculation**: Computed surface velocities for various latitudes using rotation period and radius.
<img width="566" alt="image" src="https://github.com/user-attachments/assets/80d431e5-fd42-4495-87b1-6403696f2c2f" />
<img width="674" alt="image" src="https://github.com/user-attachments/assets/4adac2ea-b93c-4dca-ae9b-073e33164936" />

## Tools and Libraries
- **Languages**: Python 3.11.x
- **Libraries**:
- OpenCV: Advanced image processing.
- NumPy: Numerical computations.
- Matplotlib: Data visualization.
- scikit-learn: Performance evaluation metrics.
- PyAutoGUI: Screenshot capturing.

## Results
- Achieved robust segmentation with high IoU and AUC scores.
- Extracted near precise geometric measurements and tracked motion dynamics.
- Estimated rotation cycles with high accuracy after accounting for environmental noise.
- Derived navigation parameters for drone applications.

## Challenges
- Variations in lighting conditions and background complexity impacted segmentation accuracy.
- Non-ideal motion of the AO introduced noise in centroid tracking.
- Processing limitations were addressed through frame skipping and filtering techniques.

## Future Work
- Incorporate machine learning models (e.g., U-Net, Mask R-CNN) for more robust segmentation.
- Optimize the pipeline for real-time operations using GPU acceleration.
- Extend the methodology to handle more complex and irregular rotating objects.

## Acknowledgments

- This project was developed as part of the assessment for the [COMP0241 - Computer Vision and Sensing](https://www.ucl.ac.uk/module-catalogue/modules/computer-vision-and-sensing-COMP0241) module at University College London. 
- It was a collaborative effort by myself, [@ziyaruso](), and [@lorenzouttini](https://github.com/lorenzouttini).
- Special thanks to the teaching team for providing guidance and resources throughout the project.

## License
[MIT License](LICENSE)
=======
# computer-vision-techniques-drone-landing
>>>>>>> d816578e23d46e5f61d0a3f0ab61f22134788d6b
