Structure from Motion:
SfM can be said as a process of predicting a 3D structure from a set of overlapping 2D images. This project actually imitates a former Microsoft application called "Photsynth". A similar web application performing sfm is VisualSfm (http://ccwu.me/vsfm/). A few years ago, Agarwal et. al published Building Rome in a Day in which they reconstructed the entire city just by using a large collection of photos from the Internet.(https://grail.cs.washington.edu/rome/rome_paper.pdf)

1) To run the Structure from Motion script:

Run `python3 main.py` in the `./Code/SfM` folder.

This will result in all resulting images given in the report in the `./Code/SfM` folder.
There is also `SfM.ipynb` file which was initially written and tested on present in the folder.

NeRF

Go to the `./Code/Nerf` folder
Run `wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz` to get the data file used for NeRF training.
Run `python3 nerf.py`

This will result in the test images predicted and shown in the report.


### Structure from Motion ###

RANSAC

<p align="center">
  <img width="600" height="300" src="https://user-images.githubusercontent.com/55713396/217652416-38f74c01-9a02-4506-80f9-c58c6279b582.png">
</p>

4 Sets of World Points
<p align="center">
  <img width="400" height="300" src="https://user-images.githubusercontent.com/55713396/217653636-825f3ac8-776e-4d50-a434-4729dc94af50.png">
</p>

Disambiguated Best World Points:
<p align="center">
  <img width="400" height="300" src="https://user-images.githubusercontent.com/55713396/217653635-be75763f-e1e8-4bc0-891a-6a112596b985.png">
</p>

Non-Linear Triangulation vs Linear Triangulation:
<p align="center">
  <img width="400" height="300" src="https://github.com/venk221/Structure-from-Motion-NeRF/assets/46212911/3c4fcbf8-935d-417d-ad10-551156a832e0"> 

PnP and Bundle Adjustment: 
<p align="center">
  <img width="400" height="300" src="https://github.com/venk221/Structure-from-Motion-NeRF/assets/46212911/6d3c7e05-ced7-49b0-a1c0-bfd8ce98800e">
</p>



