<a id="readme-top"></a>



[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Lab Website][lab-website-shield]][lab-website-url]
[![Twitter][twitter-shield]][twitter-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/JayRGopal/FaceEmotionDetection">
    <img src="readme_assets/FaceDx_Logo.png" alt="Logo" width="385" height="270">
  </a>

  <h1 align="center">FaceDx</h1>

  <p align="center">
    <h3>A Markerless Computer Vision Approach For Continuous Quantification of Internal States and Affective Behaviors in Clinical Settings</h3>
    <a href="https://github.com/JayRGopal/FaceEmotionDetection/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/JayRGopal/FaceEmotionDetection/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#project-overview">Project Overview</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contacts">Contacts</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- PROJECT OVERVIEW -->
## Project Overview

![Product Demo](readme_assets/product_demo.png)

FaceDx is a fully integrated computer vision workflow to analyze internal states, affect, pain, and short-term behaviors in the clinical setting. Importantly, our approach is markerless and does not require any fine-tuning, meaning FaceDx can automatically process any videos in which the given subject's face is visible. We bring together pre-trained, open source models that output facial action units (AUs) and emotions on a frame-by-frame basis. 

FaceDx has been shown to accurately decode self-reported long-term mood scores, as well as short-term behaviors such as smiles, frowns, or neutral expressions. Importantly, we are one of the first to conduct this validation in a clinical setting without the need for significant human intervention and analysis.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

We use a combination of the most widely employed deep learning and computer vision libraries for our custom clinical monitoring pipeline.
* **PyTorch**: MTCNN, OpenGraphAU
* **TensorFlow**: HSEmotion, DeepFace & Partial Verify,
* **OpenCV**: Video Processing, Intermediate Image Saving, Visualizer

<br />
<div align="center">
  <a href="https://github.com/JayRGopal/FaceEmotionDetection">
    <img src="readme_assets/pytorch.png" alt="Logo" width="451" height="112">
  </a>
  <br />
  <a href="https://github.com/JayRGopal/FaceEmotionDetection">
    <img src="readme_assets/tensorflow.png" alt="Logo" width="422" height="270">
  </a>

  <a href="https://github.com/JayRGopal/FaceEmotionDetection">
    <img src="readme_assets/opencv.png" alt="Logo" width="262" height="347">
  </a>

</div>



<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

TODO

### Prerequisites

TODO

### Installation

TODO

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

TODO

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

TODO

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACTS -->
## Contacts

Yuhao "Danny" Huang, MD - [@YuhaoHuangMD](https://twitter.com/YuhaoHuangMD) - Yhhuang@Stanford.edu

Jay Gopal - [@JayRGopal](https://twitter.com/JayRGopal) - Jay_Gopal@Brown.edu & JGopal@Stanford.edu

Corey Keller, MD, PhD - [@DrCoreyKeller](https://twitter.com/DrCoreyKeller) - CKeller1@Stanford.edu


Project Link: [https://github.com/JayRGopal/FaceEmotionDetection](https://github.com/JayRGopal/FaceEmotionDetection)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments


* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Font Awesome](https://fontawesome.com)
* [README Template](https://github.com/othneildrew/Best-README-Template)
* [OpenGraphAU](https://github.com/lingjivoo/OpenGraphAU)
* [HSEmotion](https://github.com/av-savchenko/hsemotion)
* [DeepFace](https://github.com/serengil/deepface)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/JayRGopal/FaceEmotionDetection
[contributors-url]: https://github.com/JayRGopal/FaceEmotionDetection/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/JayRGopal/FaceEmotionDetection
[forks-url]: https://github.com/JayRGopal/FaceEmotionDetection/network/members
[stars-shield]: https://img.shields.io/github/stars/JayRGopal/FaceEmotionDetection
[stars-url]: https://github.com/JayRGopal/FaceEmotionDetection/stargazers
[issues-shield]: https://img.shields.io/github/issues/JayRGopal/FaceEmotionDetection
[issues-url]: https://github.com/JayRGopal/FaceEmotionDetection/issues
[lab-website-shield]: https://img.shields.io/website?url=https%3A%2F%2Fprecisionneuro.stanford.edu%2F
[lab-website-url]: https://precisionneuro.stanford.edu/
[linkedin-shield]: https://img.shields.io/badge/LinkedIn-JayRGopal-blue
[linkedin-url]: https://linkedin.com/in/jay-gopal/
[twitter-shield]: https://img.shields.io/twitter/follow/JayRGopal
[twitter-url]: https://twitter.com/JayRGopal
