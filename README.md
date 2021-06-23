[![LinkedIn][linkedin-shield]][linkedin-url]
<!--
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


[![Github][github-shield]][github.com/zoumson?tab=repositories]
[![Stack Overflow][stackoverflow-shield]][stackoverflow.com/users/11175375/adam]
[![Leetcode][leetcode-shield]][eetcode.com/Hard_Code/]
-->
## Augmented Reality
![output](https://user-images.githubusercontent.com/38358621/123144894-c88c1400-d48e-11eb-94b2-0f55cabeb920.jpg)
<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#file-structure">Files Structure</a>
      <ul>
        <li><a href="#folders">Folders</a></li>
        <li><a href="#entire-files-structure">Entire Files Structure</a></li>
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
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->
Mouse cursor position prediction.
<!--Built with -->
### Built With

<br>

* [opencv](https://opencv.org/)
* [cmake](https://cmake.org/)

<br>

## File Structure

### Folders

* [resource/](resource/): images.
* [src/](src/): c++ definitions.


### Entire Files Structure 

```
.
├── CMakeLists.txt
├── README.md
├── resource
│   └── image
│       ├── augmented-reality-example.jpg
│       ├── me.jpg
│       ├── new_scenery.jpg
│       ├── test_ar_out_cpp.jpg
│       └── test.jpg
└── src
    └── demo.cpp

3 directories, 8 files

```


<!-- GETTING STARTED -->
## Getting Started

This is a sample code of how you may use  the opencv aruco for augmented reality.
To get a local copy up and running follow these simple steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* cmake
  ```sh
  sudo apt-get install cmake
  ```

 * opencv4
 ```sh
 sudo su
 ```
 ```sh
git clone https://github.com/zoumson/OpencvInstall.git     \
&& cd OpencvInstall && chmod +x install.sh && ./install.sh
 ```
### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/zoumson/AugmentedReality.git
   ```
2. Go to the project directory source
   ```sh
   cd AugmentedReality
   ```
3. Create empty directories 
   ```sh
   mkdir build &&  mkdir bin && mkdir result && cd result && mkdir image && cd ..
   ```
5. Generate the exectutable and move it to `bin`
   ```sh
   cd build && cmake .. && make -j4 && cd ..
   ```

<!-- USAGE EXAMPLES -->
### Usage

1. Run the executable 
 ```sh
   ./bin/demo -o=1 -i=./resource/image/test.pg -n=./resource/image/me.jpg
```
2. Original Image

![test](https://user-images.githubusercontent.com/38358621/123138664-e30ebf00-d487-11eb-80ef-fc605b143b34.jpg)

3. Image used to replace the existing scenaring in the square

![me](https://user-images.githubusercontent.com/38358621/123144847-ba3df800-d48e-11eb-8f63-badc55b7f21f.jpg)

4. Output image 

![output](https://user-images.githubusercontent.com/38358621/123144894-c88c1400-d48e-11eb-94b2-0f55cabeb920.jpg)

5. Back to the initial file structure configuration
   ```sh
   rm -r bin build result 
   ```
<!-- ROADMAP -->
## Roadmap

All the headers files are well docummented, read through the comments

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Adama Zouma - <!-- [@your_twitter](https://twitter.com/your_username) -->- stargue49@gmail.com

Project Link: [https://github.com/zoumson/AugmentedReality](https://github.com/zoumson/AugmentedReality.git)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Google](https://www.google.com/)
* [Stack Overflow](https://stackoverflow.com/)
* [Github](https://github.com/)




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: linkedin.com/in/adama-zouma-553bba13a
[product-screenshot]: images/screenshot.png

