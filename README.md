# Blood Vessel Image Detection

This code performs a detection of shapes in a set of images using OpenCV.<br>
The specific application was performing detection of blood vessels (myocardial & skeletal muscle capillaries) in slide 
images. This was later expanded to include colour-based weighting using cosine similarity for expanded use in histology.
<br>

This code should only be used with express consent from its authors.

### Quickstart Usage
- Ensure requirements are installed
- Create a folder called images in the same folder as the code
- Place all the images in that folder
- Open a command line/terminal/PowerShell window to the location of the folder with the code (cd = change directory)
- Type `python ./bulk_detection.py`
- See `results.csv` for results

### Requirements
- Python 3.x
- OpenCV
- Pandas
- numpy

### Usage in Publications
[Loai S, Zhou YQ, Vollett KDW, Cheng HM. “Skeletal muscle microvascular dysfunction manifests early in diabetic cardiomyopathy,” Frontiers in Cardiovascular Medicine 8, 715400, 2021.](https://www.frontiersin.org/articles/10.3389/fcvm.2021.715400/full)

### File Descriptions
- bulk_detection.py and sample_detection.py look for shapes in an original image and count them
- bulk_detection_colour.py and sample_detection_colour.py weight images to extract desired colours using cosine
  similarity before looking for shapes and ocunting them.