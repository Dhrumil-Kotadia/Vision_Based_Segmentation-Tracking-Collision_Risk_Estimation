# Vision Based Segmentation, Tracking, and Collision Risk Estimation

This repository implements computer vision techniques for object segmentation, tracking, and collision risk estimation. The methods in this project are aimed at improving safety in applications such as autonomous vehicles, robotics, and advanced driver-assistance systems (ADAS).

<p align="center">
 <img src="output/output.gif" width = "900"/>
</p>
<p align="center"><em>Example Output</em></p>

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

In modern intelligent systems, real-time decision making is critical. This project provides an integrated framework that:
- **Segments objects** in video frames.
- **Tracks object movements** over time.
- **Estimates collision risk** based on object trajectories and distances.

By combining these components, the system enhances situational awareness and assists in proactive collision avoidance.

## Features

- **Segmentation:** MASK-RCNN from Detectron2 to accurately separate objects from the background.
- **Tracking:** Robust tracking algorithms to monitor objects through successive frames.
- **Collision Risk Estimation:** Analyzes object trajectories to calculate the likelihood of collisions.
- **Modular Design:** Each component is designed to work independently and can be adapted to different environments or datasets.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Dhrumil-Kotadia/Vision_Based_Segmentation-Tracking-Collision_Risk_Estimation.git
   cd Vision_Based_Segmentation-Tracking-Collision_Risk_Estimation
   ```

2. **Install the dependencies:**

   Ensure you have Python 3 installed. Then run:

   ```bash
   pip install -r requirements.txt
   ```

   *(Detectron2 Installation: [Link](#https://detectron2.readthedocs.io/en/latest/tutorials/install.html))*

## Usage

After installing the dependencies, update the left and right frames in the data folder and execute code/3d_tracking.py

## Contributing

Contributions are welcome! If you have suggestions or improvements, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Submit a pull request.

Please ensure your contributions adhere to the project’s coding style and include appropriate tests or documentation updates.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.