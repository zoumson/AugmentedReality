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
## Mouse Next Position Prediction 
![mouse](https://user-images.githubusercontent.com/38358621/122943458-9fde1e80-d3a9-11eb-8de8-6e91af8ae3fe.png)
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

* [include/](include/): c++ header files.
* [src/](src/): c++ definitions.


### Entire Files Structure 

```
.
├── CMakeLists.txt
├── include
│   ├── ConstantsMouse.h
│   └── Mouse.h
├── README.md
└── src
    ├── demo
    │   ├── CMakeLists.txt
    │   └── demo.cpp
    └── lib
        ├── CMakeLists.txt
        └── mouse
            └── Mouse.cpp

5 directories, 8 files
```


<!-- GETTING STARTED -->
## Getting Started

This is a sample code of how you may use  the opencv libs to predict mouse position.
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
   git clone https://github.com/zoumson/TrackMousePosition.git
   ```
2. Go to the project directory source
   ```sh
   cd TrackMousePosition
   ```
3. Create empty directories 
   ```sh
   mkdir build &&  mkdir bin && mkdir lib
   ```
5. Generate the exectutable and move it to `bin`
   ```sh
   cd build && cmake .. && make -j4 && cd ..
   ```

<!-- USAGE EXAMPLES -->
### Usage

2. Run the executable 
 ```sh
   ./bin/demo 
```
3. Output
```sh
current position        = (521, 470)
next predicted position = (609, 450)
--------------------------------------------------
current position        = (591, 412)
next predicted position = (672, 372)
--------------------------------------------------
current position        = (609, 354)
next predicted position = (663, 303)
--------------------------------------------------
current position        = (589, 300)
next predicted position = (610, 245)
--------------------------------------------------
current position        = (551, 280)
next predicted position = (540, 238)
--------------------------------------------------
current position        = (441, 252)
next predicted position = (383, 218)
--------------------------------------------------
current position        = (311, 248)
next predicted position = (216, 229)
--------------------------------------------------
current position        = (179, 264)
next predicted position = (61, 262)
--------------------------------------------------
current position        = (123, 286)
next predicted position = (24, 296)
--------------------------------------------------
current position        = (55, 340)
next predicted position = (-28, 371)
--------------------------------------------------
current position        = (35, 402)
next predicted position = (-18, 449)
--------------------------------------------------
current position        = (61, 462)
next predicted position = (46, 518)
--------------------------------------------------
current position        = (97, 484)
next predicted position = (108, 529)
--------------------------------------------------
```
4. Screen saved for `current position = (97, 484)` and `next predicted position = (108, 529)`
   ```sh
   White ===> Current mouse position
   Red ===> Next predicted mouse position
   ```

![Screen](https://user-images.githubusercontent.com/38358621/122940820-65738200-d3a7-11eb-85bf-a4184efcb06d.png)
6. Back to the initial file structure configuration
   ```sh
   rm -r bin build result lib 
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

Project Link: [https://github.com/zoumson/TrackMousePosition](https://github.com/zoumson/TrackMousePosition.git)



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

