<p align="center">
<img src="https://github.com/tryingsomestuff/Morph/blob/main/demo/video_output2.gif" width="350"> 
</p>

# Tell me how to run it !

To morph between two images just do
```python src/morph.py --imgA A.jpg --imgB B.jpg --dir ./output/ --output video.mp4 --duration 4 --frame 20```

In order to morph many images in one directory do
```python src/morph.py --dir path/to/images --dir ./output/ --output video.mp4 --duration 2 --frame 24```

# What's needed ?
You'll need ffmpeg install (like `sudo apt-get install ffmpeg` or similar).  
You'll need cv2, PIL, numpy and dlib (you can use pip to get them).

# Usage
- ```--imgA```: Image A (not necessary when ```--dir``` is used)
- ```--imgB```: Image B (not necessary when ```--dir``` is used)
- ```--dir```: The directorey containing multiple images to morph (not necessary when ```--imgA``` and ```--imgB``` are used).
- ```--duration```: Length in second of each morphing from one face to the other (default is 5 seconds).
- ```--frame```: Frame rate of the encoding (default is 24 fps).
- ```--output```: Output video name.
- ```--tmpdir```: Directory to store all the videos in ```--dir```mode.
- ```--shuffle```: Shuffle image order in directory mode
- ```--with_triangle```: Display the triangulation
- ```--multiple```: Experimental multi face support (must have same number of faces in all images of course...)

Many image formats are supported.

In case you run the script on many images in a directory, you can them use a video editing tool to work with them or just try in command line:
```ffmpeg -f concat -safe 0 -i video_list.txt -c copy output.mp4```

# How does it work ?

The whole algorithm has 3 phases :
- detect face(s) in the two images and get the control points
- triangulate the image using the face control points
- interpolate the image sequence based on a linear regression between to points position on the two images and blend the two images together to get a smooth morphing

## Detect face and get the control points

Detecting facial landmarks is a subset of the shape prediction problem. Given an input image, a shape predictor attempts to localize key points of interest along the shape (for us, here, a face).

So our goal is to detect important facial structures on the face using shape prediction methods.
Detecting facial landmarks is a two step process:
- Localize the face in the image.
- Detect the key facial structures on the face.

The tool use here is based on the HOG (Histogram of Oriented Gradients) feature descriptor with a linear SVM machine learning algorithm to perform face detection (using `get_frontal_face_detector` dlib interface and the `shape_predictor_68_face_landmarks` shape predictor) but we might even use some more robust deep learning-based algorithms for face localization (like the `mmod_human_face_detector.dat` model for instance using the `cnn_face_detection_model_v1`dlib interface). In either case, the actual algorithm used to detect the face in the image doesn’t matter. Instead, what’s important is that through some method we obtain first the face(s) bounding box.

Then, given such a face region we can detect key facial structures in the face region. There are a variety of facial landmark detectors, but all methods essentially try to localize and label the following facial regions:
- Mouth
- Right eyebrow
- Left eyebrow
- Right eye
- Left eye
- Nose
- Jaw

The detector might struggle if the face is not straight-on view or too much rotated, also if the person is wearing glasses. Using the `shape_predictor_68_face_landmarks` model, we will get 68 control points as shown in the following picture.

<p align="center">
<img src="https://github.com/tryingsomestuff/Morph/blob/main/demo/68_points.png" width="550"> 
</p>

## Triangulate the image based on the control points

Once we have the 68 control points, we add 8 more : 4 at the image corners and 4 at the middle of each images size; this is enough to triangulate the image using the `cv2.Subdiv2D` tool (when there is only one face on the image, for more faces we had a bit more control points in the middle, see the multi-face experimental feature). We then keep track of the connectivity : which points is in which triangle. That's all ...

## Interpolate the image sequence

Based on the current frame number we construct the triangle interpolated between the one on image A and the one on image B. Then the transformation of corresponding triangles from image A and B to this interpolated one is obtained using `cv2.getAffineTransform` and applied using `cv2.warpAffine`. The full intermediate image is then reconstructed by adding all those small triangles together and alpha-blending both transformed triangle from image A and B. This way, both the position and the color are interpolated between the two images. Each such intermediate images is then piped to ffmpeg to build the video clip.

## Example

Let's use Dana and Fox pictures

<p align="center">
<img src="https://github.com/tryingsomestuff/Morph/blob/main/demo/dana.jpg" height="350"> 
<img src="https://github.com/tryingsomestuff/Morph/blob/main/demo/fox.jpg" height="350"> 
</p>

In this clip the triangles are printed to better understand what is going on. More other the following clip is a sequence of two morphings, the first from Dana to Fox and the second from Fox to Dana. One can see that the triangulation is not exactly the same for both way.

<p align="center">
<img src="https://github.com/tryingsomestuff/Morph/blob/main/demo/video_output.gif" width="350"> 
</p>

## Multi-face support (experimental)

<p align="center">
<img src="https://github.com/tryingsomestuff/Morph/blob/main/demo/video_output3.gif" width="550"> 
</p>

If the same number of faces are present in each image and people are nearly at the same place, there is a chance that `--multiple` option will do the trick as shown here with Hermione, Harry and Ron.

In this case 4 more control points are added at 1/3 and 2/3 of the image and faces found are sorted according to their position in the image (from left to right). This will not always work.

## Troubleshooting

A lot of things can go wrong in the process. Here are some possible issue :
- the detector cannot find a face on the picture (`NoFaceFound` exception) : maybe the face it not exactly front oriented or too rotated in the picture. You can try to edit the picture to make the face well oriented and crop the picture by yourself
- the detector finds more than one person (`MoreThanOneFaceFound` exception) and ```--multiple``` was not set : if there are more than one person on the picture, the algorithm will fail.
- If ```--multiple``` (experimental) is set but a different number of faces is found on each image (`PointsSizeError` exception) : be sure number of faces are the same, try to put them nearly as the same place in all pictures.
- any other issue (out of bound, ffmpeg issue, ...) : try to use a simpler picture, with a uniform background, be sure each face is entirely inside the picture, that hairs does not cover the face too much, ...

# Credits

This little tool is more or less a rewrite of https://github.com/provostm/face-morphing-multiple-images as a way to learn a bit about dlib.

Usefull documentation / examples :
- http://dlib.net/python/index.html
- https://pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/
- https://github.com/italojs/facial-landmarks-recognition/blob/master/main.py
- https://towardsdatascience.com/facial-mapping-landmarks-with-dlib-python-160abcf7d672

# Why so much Harry Potter here ?

Because of my lovely daughter !
