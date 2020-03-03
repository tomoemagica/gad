# gad
 Gender age detection
 
 Download
 Gender and Age Detection Python Project
 https://data-flair.training/blogs/python-project-gender-age-detection/
 
 Download this zip
 https://drive.google.com/file/d/1yy_poZSFAPKi0y2e2yj9XDe1N8xXYuKB/view
 
Unzip the file,
Place in DFL\workspace.

opencv_face_detector.pbtxt
opencv_face_detector_uint8.pb
age_deploy.prototxt
age_net.caffemodel
gender_deploy.prototxt
gender_net.caffemodel

You can delete the sample image file.


Download gad.py and gad.bat.
Download URL
https://github.com/tomoemagica/gad

Put gad.py and gad.bat in DFL\workspace.

how to use
Double click gad.bat to execute.

In the image in the DFL\workspace\data_src folder,
Video frames in which Female, 0-2 years old or 4-6 years old or 8-12 years old or 15-20 years old or 25-32 years old, are detected,

Move to the workspace\data_src\match folder.

Very young ages are included because some Japanese are detected as very young.
If you are targeting Europeans and Americans, you can correct the age judgment in gad.py.

Unfortunately, the original gad.py covers frames in video images,
An error occurred when uploading the face image.

Attempting to process images in workspace\data_src\aligned will result in an error.

The basename is required for the operation of the program.
This can be done, for example, by installing msys64.
