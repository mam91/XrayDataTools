# XrayDataTools
This repo contains various classes and functions are converting and managing datasets.  A few notable examples are:
* Converting from vgg format (one file for entire directory) to custom format (one annotation file per image file)
* Binary Mask conversion for easier at rest storage
* Converting dataset to tensorflow record format.  (This needs to be validated as functioning correctly).

#### Xray_data.py notes
If it isn't obvious, this class was created as a means to have a sort of agnostic dataset class with which we can convert to different formats.  With this said, the paths of improvements should be very clear.
