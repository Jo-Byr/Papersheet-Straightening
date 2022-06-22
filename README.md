# Papersheet-Straightening
Python function straightening a photo of a papersheet

Algorithm inspired by this article : https://hypjudy.github.io/2017/03/28/cvpr-A4-paper-sheet-detection-and-cropping/

The borders detection is unperfect but correct.
The correspondance between source pixels and target pixels are simpling made by rounding positions as interpolations did not improve significantly the resability of the image, while increasing the run time.
