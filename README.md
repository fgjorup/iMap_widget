# iMap_widget
Interactive GUI for segmentation of non-rectangular μXRD and μXRF mapping data - Developed for the DanMAX beamline
The iMap interactive map tool is a gui intended for preparing irregular mesh scans (imesh) from regular coarse overview scans. It can be used for both 2D raster scan mapping and 3D xrd-ct scanning. In the latter case the imesh map is made from an xrd-ct sinogram.

## DISCLAIMER
The iMAP GUI is part of the work flow at the DanMAX beamline, MAX IV, Sweden, and as such, includes several DanMAX-specific hard-coded parts.  
The repository has been included as a source of inspiration for others that wish to reproduce a similar workflow for non-rectangular raster scanning.

## Quick guide
### Segmentation
Load the coarse overview scan data by typing in the file path to the scan-*.h5 master file or finding the scan from the browse file dialog.  

Select the fast and slow axis motor names from the drop-down menus.  

Select the signal (xrd, xrf, other) from the drop-down menu.  

The 3D data (motor1, motor2, signal) is reduced to a 2D image, which is in turn used for the segmentation. Use the drop-down menu to select the reduction method (standard deviation, mean, integral, sum, min, max).

The scale value can be used to digitally increase the resolution of the reduced image.

Use the vertical lines in the pattern plot to select the signal range to use for the image reduction.

Use the horizontal lines in the histogram to adjust upper and lower thresholds.

Use the tools in the left toolbar to adjust the mask (grow, shrink, draw/erase, close holes, open holes).

Use the roi tool buttons to add adjustable rois and press apply or invert roi(s) to include or exclude the rois from the mask.

Click a pixel to get its motor coordinates (lower left corner of the main window)

Once the segmentation is done and the appropriate area of interest has been found, set the imesh scan parameters

![iMap screenshot](/media/iMap_screenshot.png)

### imesh parameters
Set the fast motor and slow motor resolutions.

Set the acquisition time or acquisition frequency (per pixel).

The binary image show the area of interest that will be scanned during the measurement. Note that the scan does not allow gaps in the fast axis direction, which will be automatically filled during the scan parameters calculation.

Set the number of splits to avoid to large datasets. The calculator will automatically suggest splits for scans larger than 50,000 points. The split position is indicated by vertical yellow lines in the binary image. Splits are mandatory for areas with gaps in the slow axis direction.

Once the imesh parameters are found, the scan setting are saved as *.json file(s) in /data/visitors/danmax/proposal/visit/imesh_jsons. By default the json files are named after the overview scan used during segmentation. Change the save file name in the text field to create custom json file names. The template name should include "{i}" to account for splits.

### Sequence
The sequence tap is used to create text sequence scripts for the text_sequence macro.

Available imesh json files are automatically found from the /data/visitors/danmax/proposal/visit/imesh_jsons directory. Drag-drop or double-click the relevant json files to add them to the sequence. You can add full scans or individual splits.

Use the new destination button to add a setFolder macro command to the sequence.

Use the green arrow buttons to change the order of the sequence commands.

Save the sequence text file and run it with the text_sequence macro in SPOCK.

