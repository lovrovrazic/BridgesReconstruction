# BridgesReconstruction
Reconstruction of bridges in aerial LiDAR point cloud data.

Example of how to run the script:

```bash
pip install laspy
pip install pyshp
pip install pyrr
pip install numpy
python BridgeReconstruction.py 25 data/TM_463_104.las  out 
```

The first argument sets the D value, which is the density of the points. Second argument sets the input `.las` file. And the third argument sets the folder to where the `.las` files with new points will be stored.

The script also needs topology files stored in directory `shapefiles/TN_CESTE_L.*` from [egp.gu.gov.si](https://egp.gu.gov.si/egp/).
