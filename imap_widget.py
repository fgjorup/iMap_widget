import sys
import os
import warnings
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout, QWidget, QSpacerItem, QMessageBox, QTabWidget, QTreeWidget, QTreeWidgetItem,QAbstractItemView, QGroupBox, QPushButton, QToolButton, QLabel
from PyQt6 import QtCore
import pyqtgraph as pg
import numpy as np
import h5py as h5
import skimage as ski
import resources
from scipy.interpolate import RegularGridInterpolator
import json
import glob
import re
import tango

from pyqtgraph import ColorMap
# custom colormap "grayscale_clip_upper_lower"
cmaps = {'gray': ColorMap([0, 1], [[0, 0, 0], [1, 1, 1]]),
         'grayscale_clip_upper_lower': ColorMap([0, 1e-6, 1-1e-6, 1], ['#AA0000', '#000000', '#FFFFFF', '#AA0000']),
         'transparent_red': ColorMap([0, 1], ['#00000000', '#AA0000AA'])}

# TO-DO
# - aspect ratio tool button icon
# - predict rotational motors and uncheck keep_aspect
# - add option to visualize and generate maps from pcap_mean values
#   perhaps by adding additional comboboxes for the json generation
# - add doc strings
# - add mouse hover pixel coordinates?
# - change the colors of the roi tool button icons
# - add mask transparency slider
# - merge imap_widget with mapping_tool

MAX_PX_PER_STITCH = 50000
MAX_VELOCITY = 10 # mm/s
LINE_OVERHEAD = 1.32 # s
LATENCY_TIME = 0.001 # s
DEFAULT_RAW_PATH = '/data/visitors/danmax/'
DEFAULT_JSON_DIR = '/data/visitors/danmax/'
DEFAULT_SCAN_CMD = 'imeshct'

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice")  # suppress RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")  # suppress RuntimeWarning

try:
    pathfixer = tango.DeviceProxy("b304a/ctl/sdm-01")
except:
    pathfixer = None

if pathfixer:
    DEFAULT_RAW_PATH = pathfixer.Path.split('/raw')[0]+'/raw/'
    DEFAULT_JSON_DIR = DEFAULT_RAW_PATH.split('/raw/')[0]+'/imesh_jsons'

def irregular_flat2regular_grid(I,x,y,x_res,y_res,filler=0.,corners=False):
    """
    Remap irregular flat mapping data to a regular grid
    
    Parameters
    ----------
    I       -  1d or 2d ndarray
            Intensity (or signal) of shape (n,) or (n,m)
    x       -  1d ndarray
            Measured x-positions (n,)
    y       -  1d ndarray
            Measured y-positions (n,)
    x_res   -  float
            Resolution of the x-positions
    y_res   -  float
            Resolution of the y-positions
    filler  -  float or numpy.nan
            Value to fill all the empty positions of the 
            regular grid with
    corner  -  bool
            Return the xx and yy grids as pixel corners
            instead of pixel centers (u+1,v+1)

    Return
    ---------
    I_map   - 3d ndarray
            Intensity (or signal) on a regular grid (u,v,m)
    xx      - 2d ndarray
            Grid of x-positions. Uses the measured x-positions
            where applicable, equidistant grid everywhere else
            based on the provided x_res. (u,v) or (u+1,v+1) if
            corners=True
    yy      - 2d ndarray
            Grid of y-positions. Uses the measured y-positions
            where applicable, equidistant grid everywhere else
            based on the provided y_res. (u,v) or (u+1,v+1) if
            corners=True
    """
    if len(I.shape)==1:
        I = np.atleast_2d(I).T
    # get the signal shape
    signal_shape = I.shape[1:]
    # calculate the pixel indices based on the absolute x- and y-positions
    x_index = np.round((x-x.min())/x_res).astype(int)
    y_index = np.round((y-y.min())/y_res).astype(int)
    # guess the map shape from the indices
    map_shape = (x_index.max()+1,y_index.max()+1)
    # initialize a map array
    I_map = np.full((*map_shape,*signal_shape),filler,dtype=I.dtype)
    # populate the appropriate indices
    I_map[x_index,y_index]=I
    if corners:
        map_shape = map_shape[0]+1,map_shape[1]+1
    # make grids of the nominal pixel indices
    xx,yy = np.mgrid[0:map_shape[0]:1., 0:map_shape[1]:1.]
    # convert to absolute (nominal) positions
    xx = (xx*x_res+x.min()).astype(x.dtype)
    yy = (yy*y_res+y.min()).astype(x.dtype)
    # populate with the measured positions
    xx[x_index,y_index]=x
    yy[x_index,y_index]=y
    if corners:
        xx -= x_res/2
        yy -= y_res/2
    return I_map,xx,yy

def interpolateToFineGrid(fast_res,slow_res,fast_pos,slow_pos,mask):

    # make a regular fine grid
    fast_fine = np.arange(fast_pos.min(),fast_pos.max()+fast_res,fast_res)
    slow_fine = np.arange(slow_pos.min(),slow_pos.max()+slow_res,slow_res)
    xx,yy = np.meshgrid(slow_fine,fast_fine)

    # initialize grid interpolator
    interp = RegularGridInterpolator((slow_pos,fast_pos), mask,bounds_error=False,fill_value=None)

    # interpolate the coarse mask to the fine grid
    fine_mask = interp((xx,yy))>0
    fine_mask = fine_mask.T
    return fast_fine, slow_fine, fine_mask

def guessRes(x,decimals=3):
            """Guess the resolution based on the median of the absolute steps"""
            dx = np.abs(np.diff(x))
            return np.round(np.median(dx[dx>0.0001]),decimals)

def duration2string(dur):
    """Convert a duration in hours to a string of the form 'xx h yy min zz s'"""
    return f'{int(dur):.0f} h {int(dur%1*60):.0f} min {(dur%1*60)%1*60:.0f} s'

def string2duration(string):
    """Convert a string of the form 'xx h yy min zz s' to a duration in hours"""
    h, m, s = string.split(' ')[::2]
    dur = float(h) + float(m)/60 + float(s)/3600
    return dur

class IMaskWidget(QWidget):
    """Interactive mask widget for mapping data with a third signal axis"""
    sigMouseClicked = pg.QtCore.Signal(object)
    def __init__(self, parent=None):
        super(IMaskWidget, self).__init__(parent)
        
        # Initialize the image, mask and roi arrays
        self.img = np.ones((10,10,2),dtype=np.float32)
        self.mask_thres = np.zeros((10,10),dtype=np.int8)   # threshold mask
        self.mask_roi = np.zeros((10,10),dtype=np.int8)     # roi mask
        self.mask_invert = np.zeros((10,10),dtype=np.int8) # invert mask
        self.mask_tot = np.zeros((10,10),dtype=np.int8)     # total mask

        self.reduction_mode = 'std'
        self.rois = {}
        
        # Create the main layout for the widget (vertical)
        layout = QVBoxLayout(self)   
        
        # create layout for buttons and input fields
        self.button_layout_1 = QHBoxLayout()
        layout.addLayout(self.button_layout_1)

        # Create a plot item for the line plot
        self.plot_item = pg.PlotWidget()
        layout.addWidget(self.plot_item, 1)

        # hide the y-axis ticks
        self.plot_item.getAxis('left').setTicks([])

        # add three draggable vertical lines
        _dashLine = QtCore.Qt.PenStyle.DashLine
        self.vLine_g = pg.InfiniteLine(angle=90, movable=True, pen='g', hoverPen='w')
        self.plot_item.addItem(self.vLine_g)
        self.vLine_r = pg.InfiniteLine(angle=90, movable=True, pen='r', hoverPen='w')
        self.plot_item.addItem(self.vLine_r)
        self.vLine_c = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen(color='y',style=_dashLine), hoverPen='w')
        self.plot_item.addItem(self.vLine_c)

        # connect the vertical lines to the updateImage function
        self.vLine_g.sigPositionChanged.connect(self.updateImage)
        self.vLine_r.sigPositionChanged.connect(self.updateImage)
        self.vLine_c.sigPositionChanged.connect(self.updateVlines)
        self.vLine_c.sigClicked.connect(self.clineClicked)
        
        # create a plot item for the pattern plot
        self.pattern = self.plot_item.plot()
        #self.pattern.sigClicked.connect(self.patternClicked)
        self.plot_item.getPlotItem().scene().sigMouseClicked.connect(self.patternClicked)


        # create a horizontal layout for the toolbar and image view
        layout_h = QHBoxLayout()
        layout.addLayout(layout_h,4)

        # create a vertical toolbar for the widget
        self.toolbar = pg.QtWidgets.QToolBar()
        self.toolbar.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.toolbar.setStyleSheet('QToolBar::separator {background:#4a4a4a; height: 1px; margin: 1px 5px;}')
        self.toolbar.setAutoFillBackground(True)

        layout_h.addWidget(self.toolbar)
        #self.toolbar.setFloatable(True)
        #self.toolbar.setMovable(True)
        
        # Create an image item
        self.image_view = pg.ImageView(view=pg.PlotItem())
        self.image_view.setMinimumWidth(350)
        self.image_view.setMinimumHeight(350)
        layout_h.addWidget(self.image_view, 4)
        self.image_view.view.setAutoFillBackground(True)

        # add a label at the bottom
        self.fraction_label = pg.QtWidgets.QLabel('Masked fraction: 0.0%', alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.fraction_label)


        # create a mask image item
        self.mask_overlay = pg.ImageItem()
        self.image_view.view.addItem(self.mask_overlay)
        self.mask_overlay.setColorMap(cmaps['transparent_red'])
        self.mask_overlay.setLevels([0,1])

        # hide the ROI and menu buttons
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()

        # diable the gradientEditor add
        # Set the colormap
        #self.image_view.setColorMap(cmaps['grayscale_clip_upper_lower'])

        self.image_view.ui.histogram.gradient.allowAdd = False
        self.image_view.ui.histogram.gradient.showTicks(False)

        self.image_view.ui.histogram.sigLevelsChanged.connect(self.updateThreshold)
        self.image_view.view.scene().sigMouseClicked.connect(self.mouseClicked)

        self._initButtons()

        self.initToolbar()

        self.button_layout_1.addStretch(1)

        self.updateImage()

    def _initButtons(self):
        """Initialize the buttons and input fields"""
        # displayed mask combobox
        self.mask_combobox = pg.QtWidgets.QComboBox()
        self.mask_combobox.addItems(['total', 'threshold', 'roi', 'invert'])
        self.mask_combobox.currentTextChanged.connect(self.updateMask)
        self.button_layout_1.addWidget(self.mask_combobox)

        # reduction mode input
        self.reduction_mode_input = pg.QtWidgets.QComboBox()
        self.reduction_mode_input.addItems(['std', 'integral', 'mean', 'sum', 'max', 'min'])
        self.reduction_mode_input.currentTextChanged.connect(self.setReductionMode)
        self.button_layout_1.addWidget(self.reduction_mode_input)

        # scale image spinbox
        self.scale_image_spinbox = pg.QtWidgets.QSpinBox()
        self.scale_image_spinbox.setRange(1,10)
        self.scale_image_spinbox.setValue(1)
        self.scale_image_spinbox.setToolTip('Scale image \nUse to increase the image resolution for a better mask')
        self.scale_image_spinbox.valueChanged.connect(self.updateImage)
        self.button_layout_1.addWidget(self.scale_image_spinbox)

        # replace pattern
        self.replace_pattern_button = pg.QtWidgets.QToolButton()
        self.replace_pattern_button.clicked.connect(self.replacePattern)
        pmap = pg.QtGui.QPixmap(':/icons/replace_pattern')
        icon = pg.QtGui.QIcon(pmap)
        self.replace_pattern_button.setIcon(icon)
        self.replace_pattern_button.setIconSize(QtCore.QSize(24,24))
        self.replace_pattern_button.setToolTip('Replace pattern')
        self.button_layout_1.addWidget(self.replace_pattern_button)

    def initToolbar(self):
        """Initialize the toolbar with buttons"""
        
        # grow mask
        self.grow_mask_button = pg.QtWidgets.QToolButton()
        self.grow_mask_button.clicked.connect(self.growMask)
        # set the button icon from resources
        pmap = pg.QtGui.QPixmap(':/icons/grow_mask')
        icon = pg.QtGui.QIcon(pmap)
        self.grow_mask_button.setIcon(icon)
        self.grow_mask_button.setIconSize(QtCore.QSize(24,24))
        self.grow_mask_button.setToolTip('Grow mask')
        self.toolbar.addWidget(self.grow_mask_button)

        # shrink mask
        self.shrink_mask_button = pg.QtWidgets.QToolButton()
        self.shrink_mask_button.clicked.connect(self.shrinkMask)
        # set the button icon from resources
        pmap = pg.QtGui.QPixmap(':/icons/shrink_mask')
        icon = pg.QtGui.QIcon(pmap)
        self.shrink_mask_button.setIcon(icon)
        self.shrink_mask_button.setIconSize(QtCore.QSize(24,24))
        self.shrink_mask_button.setToolTip('Shrink mask')
        self.toolbar.addWidget(self.shrink_mask_button)

        # reset mask
        self.reset_mask_button = pg.QtWidgets.QToolButton()
        self.reset_mask_button.clicked.connect(self.resetMask)
        pmap = pg.QtGui.QPixmap(':/icons/clear_brush')
        icon = pg.QtGui.QIcon(pmap)
        self.reset_mask_button.setIcon(icon)
        self.reset_mask_button.setIconSize(QtCore.QSize(24,24))
        self.reset_mask_button.setToolTip('Reset mask')
        self.toolbar.addWidget(self.reset_mask_button)

        self.toolbar.addSeparator()

        # draw/erase tool
        self.draw_erase_button = pg.QtWidgets.QToolButton()
        self.draw_erase_button.setCheckable(True)
        self.draw_erase_button.clicked.connect(self.toggleDrawErase)
        pmap = pg.QtGui.QPixmap(':/icons/draw_erase')
        icon = pg.QtGui.QIcon(pmap)
        self.draw_erase_button.setIcon(icon)
        self.draw_erase_button.setIconSize(QtCore.QSize(24,24))
        self.draw_erase_button.setToolTip('Draw/Erase mask')
        self.toolbar.addWidget(self.draw_erase_button)

        # add rectangle roi
        self.add_rectangle_roi_button = pg.QtWidgets.QToolButton()
        self.add_rectangle_roi_button.setCheckable(True)
        self.add_rectangle_roi_button.clicked.connect(self.toggleRectangleRoi)
        # set the button icon from resources
        pmap = pg.QtGui.QPixmap(':/icons/rect_roi')
        icon = pg.QtGui.QIcon(pmap)
        self.add_rectangle_roi_button.setIcon(icon)
        self.add_rectangle_roi_button.setIconSize(QtCore.QSize(24,24))
        self.add_rectangle_roi_button.setToolTip('Rectangle ROI')
        self.toolbar.addWidget(self.add_rectangle_roi_button)

        # add circle roi
        self.add_circle_roi_button = pg.QtWidgets.QToolButton()
        self.add_circle_roi_button.setCheckable(True)
        self.add_circle_roi_button.clicked.connect(self.toggleCircleRoi)
        # set the button icon from resources
        pmap = pg.QtGui.QPixmap(':/icons/circle_roi')
        icon = pg.QtGui.QIcon(pmap)
        self.add_circle_roi_button.setIcon(icon)
        self.add_circle_roi_button.setIconSize(QtCore.QSize(24,24))
        self.add_circle_roi_button.setToolTip('Circle ROI')
        # self.button_layout_2.addWidget(self.add_circle_roi_button)
        self.toolbar.addWidget(self.add_circle_roi_button)

        # add ellipse roi
        self.add_ellipse_roi_button = pg.QtWidgets.QToolButton()
        self.add_ellipse_roi_button.setCheckable(True)
        self.add_ellipse_roi_button.clicked.connect(self.toggleEllipseRoi)
        # set the button icon from resources
        pmap = pg.QtGui.QPixmap(':/icons/ellipse_roi')
        icon = pg.QtGui.QIcon(pmap)
        self.add_ellipse_roi_button.setIcon(icon)
        self.add_ellipse_roi_button.setIconSize(QtCore.QSize(24,24))
        self.add_ellipse_roi_button.setToolTip('Ellipse ROI')
        self.toolbar.addWidget(self.add_ellipse_roi_button)
        
        self.toolbar.addSeparator()

        # apply rois
        self.apply_rois_button = pg.QtWidgets.QToolButton()
        self.apply_rois_button.clicked.connect(lambda: self.applyRois(1))
        # set the button icon from resources
        pmap = pg.QtGui.QPixmap(':/icons/apply_roi')
        icon = pg.QtGui.QIcon(pmap)
        self.apply_rois_button.setIcon(icon)
        self.apply_rois_button.setIconSize(QtCore.QSize(24,24))
        self.apply_rois_button.setToolTip('Apply ROI(s)')
        self.toolbar.addWidget(self.apply_rois_button)

        # invert rois
        self.invert_rois_button = pg.QtWidgets.QToolButton()
        self.invert_rois_button.clicked.connect(lambda: self.applyRois(-1))    
        # set the button icon from resources
        pmap = pg.QtGui.QPixmap(':/icons/invert_roi')
        icon = pg.QtGui.QIcon(pmap)
        self.invert_rois_button.setIcon(icon)
        self.invert_rois_button.setIconSize(QtCore.QSize(24,24))
        self.invert_rois_button.setToolTip('Invert ROI(s)')
        self.toolbar.addWidget(self.invert_rois_button)

        self.toolbar.addSeparator()

        # close holes i.e. remove small "dark" spots
        self.close_holes_button = pg.QtWidgets.QToolButton()
        self.close_holes_button.clicked.connect(self.closeHoles)
        # set the button icon from resources
        pmap = pg.QtGui.QPixmap(':/icons/close_holes')
        icon = pg.QtGui.QIcon(pmap)
        self.close_holes_button.setIcon(icon)
        self.close_holes_button.setIconSize(QtCore.QSize(24,24))
        self.close_holes_button.setToolTip('Close mask holes')
        self.toolbar.addWidget(self.close_holes_button)

        # open holes i.e. remove small "bright" spots
        self.open_holes_button = pg.QtWidgets.QToolButton()
        self.open_holes_button.clicked.connect(self.openHoles)
        # set the button icon from resources
        pmap = pg.QtGui.QPixmap(':/icons/open_holes')
        icon = pg.QtGui.QIcon(pmap)
        self.open_holes_button.setIcon(icon)
        self.open_holes_button.setIconSize(QtCore.QSize(24,24))
        self.open_holes_button.setToolTip('Open mask holes')
        self.toolbar.addWidget(self.open_holes_button)

        # smooth the mask overlay
        self.smooth_mask_button = pg.QtWidgets.QToolButton()
        self.smooth_mask_button.clicked.connect(self.smoothMask)
        pmap = pg.QtGui.QPixmap(':/icons/smooth_mask')
        icon = pg.QtGui.QIcon(pmap)
        self.smooth_mask_button.setIcon(icon)
        self.smooth_mask_button.setIconSize(QtCore.QSize(24,24))
        self.smooth_mask_button.setToolTip('Smooth mask')
        self.toolbar.addWidget(self.smooth_mask_button)


    def toggleDrawErase(self):
        """Toggle the draw/erase mode"""
        if self.draw_erase_button.isChecked():
            self.image_view.view.setMenuEnabled(False)
        else:
            self.image_view.view.setMenuEnabled(True)

    def mouseClicked(self, event):
        """Handle mouse clicks on the image view"""

        event.accept()
        pos = self.image_view.view.getViewBox().mapSceneToView(event.scenePos())
        x,y = int(pos.x()),int(pos.y())
        if self.draw_erase_button.isChecked():
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.mask_roi[x,y] = 1
            elif event.button() == QtCore.Qt.MouseButton.RightButton:
                self.mask_roi[x,y] = 0
            self.updateMask()
        # emit a signal with the x and y coordinates
        self.sigMouseClicked.emit((x,y))
        
    def setData(self, img,aspect=1):
        """Set the image data and update the image view"""

        img = np.atleast_3d(img).astype(np.float32)
        self.img = img

        
        if img.shape[2] > 1:
            y = np.nanmean(img, axis=(0,1))
            y = (y-y.min())/(y.max()-y.min())*100
            x = np.arange(0,y.shape[0])
            # set the y data in the pattern plot
            self.pattern.setData(x=x,
                                 y=y)

            # set the vertical lines to the start and end of the pattern
            # suppress the sigPositionChanged signal
            self.vLine_g.sigPositionChanged.disconnect(self.updateImage)
            self.vLine_r.sigPositionChanged.disconnect(self.updateImage)
            self.vLine_c.sigPositionChanged.disconnect(self.updateVlines)
            self.vLine_g.setValue(x[0])
            self.vLine_r.setValue(x[-1])
            self.vLine_c.setValue(x[int(x.shape[0]/2)])
            self.vLine_g.sigPositionChanged.connect(self.updateImage)
            self.vLine_r.sigPositionChanged.connect(self.updateImage)
            self.vLine_c.sigPositionChanged.connect(self.updateVlines)

            # limit the ranges of the view and the position of the vertical lines
            self.plot_item.setLimits(xMin=-5, xMax=x[-1]+5, yMin=-2, yMax=102)
            self.vLine_g.setBounds([-1,x[-1]+1])
            self.vLine_r.setBounds([-1,x[-1]+1])

            # show the vertical lines
            self.vLine_g.show()
            self.vLine_r.show()
            self.vLine_c.show()

            # enable the reduction mode input
            self.reduction_mode_input.setEnabled(True)
            self.reduction_mode_input.setCurrentText(self.reduction_mode)
            if self.reduction_mode_input.findText('None') >= 0:
                self.reduction_mode_input.removeItem('None')
            
            # change the default reduction mode for small datasets
            if img.shape[2] < 10:
                self.reduction_mode_input.setCurrentText('max')
            
        else:
            # hide the vertical lines
            self.vLine_g.hide()
            self.vLine_r.hide()
            self.vLine_c.hide()

            # disable the reduction mode input
            self.reduction_mode_input.setEnabled(False)
            self.reduction_mode_input.addItem('None')
            self.reduction_mode_input.setCurrentText('None')
        
        im = self.reduceImage(img, self.reduction_mode)
        
        self.mask_thres = np.zeros_like(im,dtype=np.int8)
        self.mask_roi = np.zeros_like(im,dtype=np.int8)
        self.mask_invert = np.zeros_like(im,dtype=np.int8)
        self.mask_tot = np.zeros_like(im,dtype=np.int8)

        # Set the image in the image item
        self.image_view.setImage(im)
        # set the mask overlay
        self.mask_overlay.setImage(self.mask_roi, autoRange=False, autoLevels=True)

        # set the aspect ratio of the image view
        self.image_view.view.setAspectLocked(True,ratio=aspect)
    
    def setReductionMode(self, mode):
        """Set the reduction mode for the image"""
        self.reduction_mode = mode
        self.updateImage()

    def reduceImage(self, img, reduction_mode='std'):
        """
        Reduce the image along the signal axis to a 2d image. Reduction options are
        'std', 'integral', 'mean', 'sum', 'max', 'min'. The reduction is done along the
        third axis of the image in the interval defined by the pattern vertical lines.
        The 'integral' reduction includes a simple background subtraction.
        
        Parameters
        ----------
        img : ndarray
            3d image of shape (n,m,p)
        reduction_mode : str
            Mode of reduction. Options are 'std', 'integral', 'mean', 'sum', 'max', 'min'
        Returns
        -------
        im : ndarray
            2d image of shape (n,m)
        """
        
        # get the pattern roi from the vertical lines
        x0 = min(self.vLine_g.value(), self.vLine_r.value())
        x1 = max(self.vLine_g.value(), self.vLine_r.value())
        
        # set the position of the center line
        self.vLine_c.sigPositionChanged.disconnect(self.updateVlines)
        self.vLine_c.setValue((x0+x1)/2)
        self.vLine_c.sigPositionChanged.connect(self.updateVlines)

        # ensure valid indices
        x0 = int(max(0, x0))
        x1 = int(max(0,min(img.shape[2], x1+1)))
        if x0 == x1:
            x1 = x0 + 1

        if reduction_mode == 'std':
            im = np.nanstd(img[:,:,x0:x1], axis=2,dtype=np.float32)
        elif reduction_mode == 'integral':
            # subtract a simple background
            img = img - np.nanmean(img[:,:,[x0,x1-1]], axis=2,keepdims=True)
            im = np.trapz(img[:,:,x0:x1], axis=2).astype(np.float32)
        elif reduction_mode == 'mean':
            im = np.nanmean(img[:,:,x0:x1], axis=2,dtype=np.float32)
        elif reduction_mode == 'sum':
            im = np.nansum(img[:,:,x0:x1], axis=2,dtype=np.float32)
        elif reduction_mode == 'max':
            im = np.nanmax(img[:,:,x0:x1], axis=2).astype(np.float32)
        elif reduction_mode == 'min':
            im = np.nanmin(img[:,:,x0:x1], axis=2).astype(np.float32)
        elif reduction_mode == 'None':
            im = img[:,:,0]
        else:
            raise ValueError('reduction_mode not recognized')
        return im
    
    def updateVlines(self):
        """Update the vertical lines. Called when the center line is moved"""
        # get the distance between the green and red lines
        # such the distance is positive if green is left of red
        d = self.vLine_r.value()-self.vLine_g.value() # 
        # set the red and green lines to be symmetric around the center line
        self.vLine_g.sigPositionChanged.disconnect(self.updateImage)
        self.vLine_r.sigPositionChanged.disconnect(self.updateImage)
        self.vLine_g.setValue(self.vLine_c.value()-d/2)
        self.vLine_r.setValue(self.vLine_c.value()+d/2)
        self.vLine_g.sigPositionChanged.connect(self.updateImage)
        self.vLine_r.sigPositionChanged.connect(self.updateImage)
        self.updateImage() 
    
    def clineClicked(self, line, event):
        """Handle double clicks on the center line"""
        # check if the event was a double click
        if event.double():
            ## move the green and red lines close to the center line
            # find the range of the pattern
            x_range = self.pattern.xData.max()-self.pattern.xData.min()
            dx = x_range/40
            # set the green and red lines to be symmetric around the center line
            self.vLine_g.sigPositionChanged.disconnect(self.updateImage)
            self.vLine_r.sigPositionChanged.disconnect(self.updateImage)
            self.vLine_g.setValue(self.vLine_c.value()-dx)
            self.vLine_r.setValue(self.vLine_c.value()+dx)
            self.vLine_g.sigPositionChanged.connect(self.updateImage)
            self.vLine_r.sigPositionChanged.connect(self.updateImage)
            self.updateImage()

    def patternClicked(self, event):
        """Handle clicks on the pattern plot"""
        # check if the event was a double click
        if event.double():
            pos = self.plot_item.getPlotItem().vb.mapSceneToView(event.scenePos()).x()
            self.vLine_c.setValue(pos)
        

    def updateImage(self):
        """Update the image view with the current reduction mode and pattern roi"""
        scale = self.scale_image_spinbox.value()
        im = self.reduceImage(self.img, self.reduction_mode)
        autoRange = False
        if scale > 1:
            shape = im.shape
            im = ski.transform.rescale(im, scale, anti_aliasing=False)
        if not self.mask_roi.shape == im.shape:
            self.mask_roi = ski.transform.resize(self.mask_roi, im.shape, order=0, anti_aliasing=False)
            self.mask_invert = ski.transform.resize(self.mask_invert, im.shape, order=0, anti_aliasing=False)
            self.mask_thres = ski.transform.resize(self.mask_thres, im.shape, order=0, anti_aliasing=False)
            autoRange = True
        self.image_view.setImage(im, autoRange=autoRange, autoLevels=True)
    
    def replacePattern(self):
        """Update the image and pattern plot with the current mask"""
        img = self.img
        mask = self.getCurrentMask()
        img[mask>0] = np.nan
        self.setData(img)
        self.updateMask()
        # if len(self.img.shape) == 3:
        #     y = np.mean(self.img[self.mask_tot<1], axis=0)
        #     y = (y-y.min())/(y.max()-y.min())*100
        #     x = np.arange(0,y.shape[0])
        #     # set the y data in the pattern plot
        #     self.pattern.setData(x=x,
        #                          y=y)


    def updateMask(self):
        """Update the mask overlay and the masked fraction label"""
        # get the mask to display
        mask = self.getCurrentMask()
        if -1 in mask:
            mask = 1+mask
        self.mask_overlay.setImage(mask, autoRange=False, autoLevels=True)
        # calculate the masked fraction
        frac = 100*np.sum(mask>0)/mask.size
        self.fraction_label.setText(f'Masked fraction: {frac:.1f}%')

    def updateThreshold(self):
        """Update the threshold mask based on the histogram levels"""
        # the the current threshold values from the histogram
        levels = self.image_view.ui.histogram.getLevels()
        im = self.image_view.getImageItem().image
        self.mask_thres = np.isnan(im).astype(np.int8)
        self.mask_thres += np.int8(im < levels[0])
        self.mask_thres += np.int8(im > levels[1])
        self.mask_thres = np.clip(self.mask_thres,0,1)
        self.updateMask()
    
    def getMask(self):
        """Get the total mask"""
        mask = np.clip(self.mask_roi + self.mask_thres,0,1)
        mask = np.clip(mask + self.mask_invert,0,1)
        self.mask_tot = mask
        return mask

    def getCurrentMask(self):
        """Get the current mask selected in the mask combobox"""
        selection = self.mask_combobox.currentText()
        if selection == 'threshold':
            return self.mask_thres
        elif selection == 'roi':
            return self.mask_roi
        elif selection == 'invert':
            return 1+self.mask_invert
        elif selection == 'total':
            return self.getMask()

    def setCurrentMask(self, mask):
        """Set the current mask selected in the mask combobox"""
        selection = self.mask_combobox.currentText()
        if selection == 'threshold':
            self.mask_thres = mask
        elif selection == 'roi':
            self.mask_roi = mask
        elif selection == 'invert':
            self.mask_invert = mask
        elif selection == 'total':
            self.mask_tot = mask

    def growMask(self):
        """Grow the current mask(s)"""
        if self.mask_combobox.currentText() in ['total', 'threshold']:
            self.mask_thres = ski.morphology.binary_dilation(self.mask_thres)
        if self.mask_combobox.currentText() in ['total', 'roi']:
            self.mask_roi = ski.morphology.binary_dilation(self.mask_roi)
        else: # invert
            self.mask_invert = -ski.morphology.binary_dilation(-self.mask_invert)
        self.updateMask()


    def shrinkMask(self):
        """Shrink the current mask(s)"""
        if self.mask_combobox.currentText() in ['total', 'threshold']:
            self.mask_thres = ski.morphology.binary_erosion(self.mask_thres)
        if self.mask_combobox.currentText() in ['total', 'roi']:
            self.mask_roi = ski.morphology.binary_erosion(self.mask_roi)
        else: # invert
            self.mask_invert = -ski.morphology.binary_erosion(-self.mask_invert)
        self.updateMask()

    def closeHoles(self):
        """Close small holes in the threshold mask"""
        mask = self.mask_thres
        mask = ski.morphology.binary_closing(mask)
        self.mask_thres = np.int8(mask)
        self.updateMask()

    def openHoles(self):
        """Open small holes in the threshold mask"""
        mask = self.mask_thres
        mask = ski.morphology.binary_opening(mask)
        self.mask_thres = np.int8(mask)
        self.updateMask()

    def smoothMask(self):
        """Smooth the current mask(s)"""
        if self.mask_combobox.currentText() in ['total', 'threshold']:
            self.mask_thres = ski.filters.median(self.mask_thres)
        if self.mask_combobox.currentText() in ['total', 'roi']:
            self.mask_roi = ski.filters.median(self.mask_roi)
        else: # invert
            self.mask_invert = ski.filters.median(self.mask_invert)
        self.updateMask()

    def resetMask(self):
        """Reset the masks"""
        self.mask_roi = np.zeros_like(self.mask_roi,dtype=np.int8)
        self.mask_invert = np.zeros_like(self.mask_invert,dtype=np.int8)
        self.mask_thres = np.zeros_like(self.mask_thres,dtype=np.int8)

        # set the histogram levels to the full range
        vmin,vmax = self.image_view.getImageItem().image.min(), self.image_view.getImageItem().image.max()
        self.image_view.ui.histogram.setLevels(vmin,vmax)
        self.updateMask()

    def toggleCircleRoi(self):
        """Toggle the circle roi"""
        if 'circle' in self.rois:
            self.image_view.removeItem(self.rois['circle'])
            del self.rois['circle']
            self.add_circle_roi_button.setChecked(False)
        else:
            self.addCircleRoi()
            self.add_circle_roi_button.setChecked(True)

    def toggleEllipseRoi(self):
        """Toggle the ellipse roi"""
        if 'ellipse' in self.rois:
            self.image_view.removeItem(self.rois['ellipse'])
            del self.rois['ellipse']
            self.add_ellipse_roi_button.setChecked(False)
        else:
            self.addEllipseRoi()
            self.add_ellipse_roi_button.setChecked(True)

    def toggleRectangleRoi(self):
        """Toggle the rectangle roi"""
        if 'rectangle' in self.rois:
            self.image_view.removeItem(self.rois['rectangle'])
            del self.rois['rectangle']
            self.add_rectangle_roi_button.setChecked(False)
        else:
            self.addRectangleRoi()
            self.add_rectangle_roi_button.setChecked(True)

    def addCircleRoi(self):
        """Add a circle roi to the image view"""
        roi = pg.CircleROI([0, 0], [10, 10],scaleSnap=True,translateSnap=True,rotateSnap=True)
        self.image_view.addItem(roi)
        # count the number of circle rois already present
        n = sum([1 for key in self.rois if 'circle' in key])
        self.rois[f'circle'] = roi
    
    def addEllipseRoi(self):
        """Add an ellipse roi to the image view"""
        roi = pg.EllipseROI([0, 0], [10, 10],scaleSnap=True,translateSnap=True,rotateSnap=True)
        self.image_view.addItem(roi)
        # count the number of ellipse rois already present
        n = sum([1 for key in self.rois if 'ellipse' in key])
        self.rois[f'ellipse'] = roi
    
    def addRectangleRoi(self):
        """Add a rectangle roi to the image view"""
        roi = pg.RectROI([0, 0], [10, 10],scaleSnap=True,translateSnap=True,rotateSnap=False)
        roi.addRotateHandle([1, 0], [0.5, 0.5])
        #roi = pg.ROI([0, 0], [10, 10],scaleSnap=True,translateSnap=True,rotateSnap=True)
        self.image_view.addItem(roi)
        # count the number of rectangle rois already present
        n = sum([1 for key in self.rois if 'rectangle' in key])
        self.rois[f'rectangle'] = roi

    def applyRois(self,sign):
        """Apply the rois to the mask. sign=1 for apply, sign=-1 for invert"""
        ones = np.ones_like(self.mask_roi,dtype=np.int8)
        mask = np.zeros_like(self.mask_roi,dtype=np.int8)
        for key, roi in self.rois.items():
            m, coords = roi.getArrayRegion(ones, self.image_view.imageItem,returnMappedCoords=True)
            coords = np.round(coords).astype(int)
            coords = coords[:,m>0]
            mask[coords[0],coords[1]] += 1
        mask = np.clip(mask,0,1)
        mask = sign*ski.morphology.binary_closing(mask).astype(np.int8)

        self.mask_roi = np.clip(self.mask_roi + mask,0,1)
        self.mask_invert = np.clip(self.mask_invert + mask,-1,0)
        self.updateMask()

    def getScaleFactor(self):
        """Get the scale factor of the image view"""
        return self.scale_image_spinbox.value()

class MiniMapWidget(pg.PlotWidget):
    
    def __init__(self, parent=None):
        super(MiniMapWidget, self).__init__(parent)

        self.image_item = pg.ImageItem()
        self.addItem(self.image_item)

        self.invertY(True)
        self.setAspectLocked(True)
        self.setAutoFillBackground(True)

        # remove axis ticks
        self.getAxis('left').setTicks([])
        self.getAxis('bottom').setTicks([])
        self.getAxis('left').setPen(None)
        self.getAxis('bottom').setPen(None)

        self.vlines = []
        
    def setImage(self,im,aspect=1):
        self.image_item.setImage(im.astype(np.int8),levels=[-0.1,1])
        self.setAspectLocked(True,ratio=aspect)
        self.autoRange()

    def getImage(self):
        return self.image_item.image
    
    def setVLines(self, x):
        for line in self.vlines:
            self.removeItem(line)
        self.vlines = []
        for xi in x:
            line = pg.InfiniteLine(angle=90, movable=False)
            line.setValue(xi)
            self.addItem(line)
            self.vlines.append(line)
        
class IMapWidget(QWidget):
    sigIMaskClicked = pg.QtCore.Signal(object)
    def __init__(self,parent=None):
        super(IMapWidget, self).__init__(parent)
        # self.setWindowTitle("PyQtGraph ImageWidget Example")

        self.fast_pos = None
        self.slow_pos = None
        self.mask_aspect_ratio = 1
        self.json_dict = None
        self.json_save_dir = None

        layout_cental = QHBoxLayout(self)

        # imask layout
        layout_imask = QVBoxLayout()
        layout_cental.addLayout(layout_imask,2)
        self._initImaskLayout(layout_imask)

        # use mask button
        self.use_mask_button = pg.QtWidgets.QToolButton()
        self.use_mask_button.clicked.connect(self.useMask)
        self.use_mask_button.setText('>')
        self.use_mask_button.setMinimumHeight(75)
        layout_cental.addWidget(self.use_mask_button,0)

        # map layout
        layout_map  = QVBoxLayout()
        layout_cental.addLayout(layout_map,1)
        self._initMapLayout(layout_map)


    def _initImaskLayout(self,layout):

        # add input layout
        layout_input = QHBoxLayout()
        layout.addLayout(layout_input,1)

        # lineEdit for the filename
        self.fname_input = pg.QtWidgets.QLineEdit()
        self.fname_input.setText(DEFAULT_RAW_PATH)
        self.fname_input.returnPressed.connect(self.loadImage)
        layout_input.addWidget(self.fname_input,4)

        # add browse button
        self.browse_button = pg.QtWidgets.QToolButton()
        self.browse_button.clicked.connect(self.browseFile)
        pmap = pg.QtGui.QPixmap(':/icons/browse')
        icon = pg.QtGui.QIcon(pmap)
        self.browse_button.setIcon(icon)
        self.browse_button.setIconSize(QtCore.QSize(24,24))
        layout_input.addWidget(self.browse_button,0)

        # save mask
        self.save_mask_button = pg.QtWidgets.QToolButton()
        self.save_mask_button.clicked.connect(self.saveMask)
        pmap = pg.QtGui.QPixmap(':/icons/save')
        icon = pg.QtGui.QIcon(pmap)
        self.save_mask_button.setIcon(icon)
        self.save_mask_button.setIconSize(QtCore.QSize(24,24))
        self.save_mask_button.setToolTip('Save mask')
        layout_input.addWidget(self.save_mask_button,0)

        layout_input.addStretch(1)

        # add layout for scan-specic input
        layout_params = QHBoxLayout()
        layout.addLayout(layout_params)
        
        # fast and slow axis combobox
        label = pg.QtWidgets.QLabel('Fast axis:')
        layout_params.addWidget(label)
        self.fast_axis_combobox = pg.QtWidgets.QComboBox()
        self.fast_axis_combobox.setEnabled(False)
        self.fast_axis_combobox.currentTextChanged.connect(self.updateImage)
        layout_params.addWidget(self.fast_axis_combobox,1)

        label = pg.QtWidgets.QLabel('Slow axis:')
        layout_params.addWidget(label)
        self.slow_axis_combobox = pg.QtWidgets.QComboBox()
        self.slow_axis_combobox.setEnabled(False)
        self.slow_axis_combobox.currentTextChanged.connect(self.updateImage)
        layout_params.addWidget(self.slow_axis_combobox,1)

        # signal type combobox
        label = pg.QtWidgets.QLabel('Signal:')
        layout_params.addWidget(label)
        self.signal_combobox = pg.QtWidgets.QComboBox()
        self.signal_combobox.setEnabled(False)
        self.signal_combobox.currentTextChanged.connect(self.updateImage)
        layout_params.addWidget(self.signal_combobox,1)

        self.keep_aspect_button = pg.QtWidgets.QPushButton('Keep aspect')
        self.keep_aspect_button.setCheckable(True)
        self.keep_aspect_button.setChecked(True)
        self.keep_aspect_button.clicked.connect(self.updateImage)
        layout_params.addWidget(self.keep_aspect_button,1)


        layout_params.addStretch(1)

        
        # Create an instance of the custom ImageWidget and add it to the layout
        self.imask_widget = IMaskWidget()
        self.imask_widget.sigMouseClicked.connect(self.iMaskClicked)
        layout.addWidget(self.imask_widget)

        # Set the image
        #im = self.openH5(fname)
        #self.imask_widget.setData(im)
        #self.imask_widget.setData(np.mean(im, axis=2))
        # im = np.stack([np.mean(im[:,:,::2],axis=2),np.mean(im[:,:,1::2],axis=2)])
        # im = np.stack([np.mean(im[:,:,::3],axis=2),np.mean(im[:,:,1::3],axis=2),np.mean(im[:,:,2::3],axis=2)])
        # im = np.transpose(im,axes=(1,2,0))
        # print(im.shape)
        #self.imask_widget.setData(im)

    def _initMapLayout(self,layout):

        ## GRID LAYOUT
        # add a grid layout for paramaters
        layout_params = QGridLayout()
        # layout_params.setColumnStretch(1,1)
        layout.addLayout(layout_params,1)

        # add labels and spinboxes for the target resolutions
        label = pg.QtWidgets.QLabel('Fast res.:')
        layout_params.addWidget(label,0,0)
        self.fast_res_input = pg.QtWidgets.QDoubleSpinBox()
        self.fast_res_input.setDecimals(1)
        self.fast_res_input.setRange(0.1,10000)
        self.fast_res_input.setSingleStep(10)
        self.fast_res_input.setValue(100)
        self.fast_res_input.setSuffix(' μm')
        self.fast_res_input.editingFinished.connect(self.applyMapParams)
        layout_params.addWidget(self.fast_res_input,0,1,1,1)

        label = pg.QtWidgets.QLabel('Slow res.:')
        layout_params.addWidget(label,0,2)
        self.slow_res_input = pg.QtWidgets.QDoubleSpinBox()
        self.slow_res_input.setDecimals(1)
        self.slow_res_input.setRange(0.1,10000)
        self.slow_res_input.setSingleStep(10)
        self.slow_res_input.setValue(100)
        self.slow_res_input.setSuffix(' μm')
        self.slow_res_input.editingFinished.connect(self.applyMapParams)
        layout_params.addWidget(self.slow_res_input,0,3,1,1)

        # add labels and spinboxes for the target acquisition time and frequency
        label = pg.QtWidgets.QLabel('Acq. time:')
        layout_params.addWidget(label,1,0)
        self.acq_time_input = pg.QtWidgets.QDoubleSpinBox()
        self.acq_time_input.setDecimals(3)
        self.acq_time_input.setRange(0.004,10)
        self.acq_time_input.setSingleStep(1)
        self.acq_time_input.setValue(1/10)
        self.acq_time_input.setSuffix(' s')
        self.acq_time_input.valueChanged.connect(self.setAcqFreq)
        self.acq_time_input.editingFinished.connect(self.applyMapParams)
        layout_params.addWidget(self.acq_time_input,1,1,1,1)

        label = pg.QtWidgets.QLabel('Acq. freq.:')
        layout_params.addWidget(label,1,2)
        self.acq_freq_input = pg.QtWidgets.QDoubleSpinBox()
        self.acq_freq_input.setDecimals(2)
        self.acq_freq_input.setRange(0.1,1/0.004)
        self.acq_freq_input.setSingleStep(1)
        self.acq_freq_input.setValue(10)
        self.acq_freq_input.setSuffix(' Hz')
        self.acq_freq_input.valueChanged.connect(self.setAcqTime)
        self.acq_freq_input.editingFinished.connect(self.applyMapParams)
        layout_params.addWidget(self.acq_freq_input,1,3,1,1)

        # add a split map spinbox
        label = pg.QtWidgets.QLabel('Split:')
        layout_params.addWidget(label,2,0)
        self.split_map_input = pg.QtWidgets.QSpinBox()
        self.split_map_input.setRange(0,99)
        self.split_map_input.setValue(0)
        self.split_map_input.editingFinished.connect(self.applyMapParams)
        layout_params.addWidget(self.split_map_input,2,1,1,1)

        # add toggle smooth checkbox
        self.smooth_image_checkbox = pg.QtWidgets.QCheckBox('Smooth')
        self.smooth_image_checkbox.setChecked(True)
        layout_params.addWidget(self.smooth_image_checkbox,2,2,1,1)

        # add an apply button
        self.apply_button = pg.QtWidgets.QPushButton('Apply')
        self.apply_button.clicked.connect(self.applyMapParams)
        layout_params.addWidget(self.apply_button,2,3,1,1)

        # add a snake checkbox
        self.snake_checkbox = pg.QtWidgets.QCheckBox('Snake mode')
        self.snake_checkbox.setChecked(True)
        layout_params.addWidget(self.snake_checkbox,3,1,1,1)


        ## MINI MAP
        # add a mini map widget
        self.mini_map_widget = MiniMapWidget(self)
        layout.addWidget(self.mini_map_widget,3)


        ## MAP SUMMARY
        # add a text browser
        self.summary_textbrowser = pg.QtWidgets.QTextBrowser()
        layout.addWidget(self.summary_textbrowser,2)

        # add a text edit
        #self.summary_textedit = pg.QtWidgets.QTextEdit()
        #layout.addWidget(self.summary_textedit,1)

        ## SAVE MAP
        ## a horizontal layout for the save map button and the save map line edit
        layout_save = QHBoxLayout()
        layout.addLayout(layout_save,1)

        # add a line edit for the save map path
        self.save_map_lineedit = pg.QtWidgets.QLineEdit()
        self.save_map_lineedit.setPlaceholderText('scan-xxxx_01_of_01.json')
        layout_save.addWidget(self.save_map_lineedit,4)

        # add a browse button
        self.save_map_button = pg.QtWidgets.QToolButton()
        # self.save_map_button.clicked.connect(self.saveMap)
        pmap = pg.QtGui.QPixmap(':/icons/save')
        icon = pg.QtGui.QIcon(pmap)
        self.save_map_button.setIcon(icon)
        self.save_map_button.setIconSize(QtCore.QSize(24,24))
        self.save_map_button.setToolTip('Save map')
        self.save_map_button.clicked.connect(self.saveJson)
        layout_save.addWidget(self.save_map_button,0)

    def iMaskClicked(self, pos):
        if self.fast_pos is None or self.slow_pos is None:
            return
        x,y = pos
        scale = self.imask_widget.getScaleFactor()
        x = x//scale
        y = y//scale
        # check if the x and y coordinates are within the bounds of the image
        if x < 0 or y < 0 or x >= self.fast_pos.shape[0] or y >= self.slow_pos.shape[1]:
            self.sigIMaskClicked.emit(None)
            return


        # get the fast and slow positions
        slow_pos = self.slow_pos[x,0]
        fast_pos = self.fast_pos[0,y]

        # emit a signal with the x and y coordinates
        self.sigIMaskClicked.emit((slow_pos,fast_pos))

    def setAcqTime(self):

        freq = self.acq_freq_input.value()
        self.acq_time_input.valueChanged.disconnect(self.setAcqFreq)
        self.acq_time_input.setValue(1/freq)
        self.acq_time_input.valueChanged.connect(self.setAcqFreq)

    def setAcqFreq(self):
        time = self.acq_time_input.value()
        self.acq_freq_input.valueChanged.disconnect(self.setAcqTime)
        self.acq_freq_input.setValue(1/time)
        self.acq_freq_input.valueChanged.connect(self.setAcqTime)
        
    def useMask(self):
        # get the (inverted) mask from the imask widget
        mask = 1-self.imask_widget.getCurrentMask()


        # if not self.fast_pos is None and not self.slow_pos is None:
        #     self.fast_res_input.setValue(np.diff(self.fast_pos[0,:]).mean()*1e3)
        #     self.slow_res_input.setValue(np.diff(self.slow_pos[:,0]).mean()*1e3)

        # # remove empty rows and columns
        # mask = mask[~np.all(mask==0,axis=1)]
        # mask = mask[:,~np.all(mask==0,axis=0)]

        # get the index of the first and last non-zero elements in the mask
        first = np.argmax(mask,axis=1)
        last = (mask.shape[1]-1)-np.argmax(mask[:,::-1],axis=1)
        empty_lines = np.sum(mask,axis=1)<1

        # fill between the first and last non-zero elements using slicing
        mask[np.arange(mask.shape[0])[:, None], np.arange(mask.shape[1])] = (np.arange(mask.shape[1]) >= first[:, None]) & (np.arange(mask.shape[1]) <= last[:, None])

        mask[empty_lines] = 0

        # # apply the  mask to the mini map
        # if self.smooth_image_checkbox.isChecked():
        #     mask = ski.filters.median(mask)
        self.mini_map_widget.setImage(mask,aspect=self.mask_aspect_ratio)

        #self.updateSummary()

    def applyMapParams(self):
        # ensure the mask is up to date
        self.useMask()

        # get the mask from the minimap widget
        mask = self.mini_map_widget.getImage()
        if mask is None or self.fast_pos is None or self.slow_pos is None:
            return

        # get the target resolutions
        fast_res = self.fast_res_input.value()*1e-3 # convert to mm
        slow_res = self.slow_res_input.value()*1e-3 # convert to mm

        # get the fast and slow positions
        fast_pos = self.fast_pos[0,:]
        slow_pos = self.slow_pos[:,0]
        if fast_pos.shape[0] != mask.shape[1] or slow_pos.shape[0] != mask.shape[0]:
            # ensure the fast and slow positions are the same length as the mask
            # incase the mask has been resized
            fast_pos = np.linspace(fast_pos[0],fast_pos[-1],mask.shape[1])
            slow_pos = np.linspace(slow_pos[0],slow_pos[-1],mask.shape[0])

        # interpolate the mask to the target resolutions
        fast_fine, slow_fine, fine_mask = interpolateToFineGrid(fast_res,slow_res,fast_pos,slow_pos,mask)

        if self.smooth_image_checkbox.isChecked():
            mask = ski.filters.median(fine_mask)

        self.mini_map_widget.setImage(fine_mask,aspect=slow_res/fast_res)

        # find first and last point along each line
        first_pt = np.nanargmax(fine_mask,axis=1)
        last_pt = (fine_mask.shape[1]-1)-np.nanargmax(fine_mask[:,::-1],axis=1)

        # find non-empty lines
        non_empty_lines = np.sum(fine_mask,axis=1)>0

        regions = np.diff(np.append(0,non_empty_lines.astype(int)))
        start_indices = np.where(regions==1)[0]
        stop_indices = np.where(regions==-1)[0]
        if non_empty_lines[-1] == 1:
            stop_indices = np.append(stop_indices,non_empty_lines.shape[0]-1)
                    

        # find the first and last (non-empty) lines 
        x_start_index = np.argmax(non_empty_lines)
        x_stop_index = (non_empty_lines.shape[0]-1)-np.argmax(non_empty_lines[::-1])

        # set the first and last points to zero for empty lines
        first_pt[~non_empty_lines] = 0
        last_pt[~non_empty_lines] = 0
        

        # find the start y-values
        start = fast_fine[first_pt]
        # find the stop y-values
        stop = fast_fine[last_pt]
        if self.snake_checkbox.isChecked():
            start[1::2] = fast_fine[last_pt][1::2]
            stop[1::2] = fast_fine[first_pt][1::2]

        # calculate the number of pixels per line
        px_per_line = np.round(np.abs(stop-start)/fast_res).astype(int)

        if len(start_indices)>1:
            # determine the required number of stitches based on max_px_per_stitch per region
            stitch_indices = np.append(start_indices,stop_indices)
            for i in range(len(start_indices)):
                s_ = np.s_[start_indices[i]:stop_indices[i]]
                n_px = np.sum(px_per_line[s_])
                stitches = int(np.ceil(n_px/MAX_PX_PER_STITCH))
                px_per_stitch  = int(np.ceil(n_px/stitches))
                cutoff = np.cumsum(px_per_line[s_])%px_per_stitch
                if cutoff[-1] == 0:
                    cutoff = cutoff[:-1]
                cutoff_indices = np.arange(1,len(cutoff))[np.diff(cutoff)<0]+start_indices[i]
                stitch_indices = np.append(stitch_indices,cutoff_indices)
            stitch_indices = np.sort(stitch_indices)
            stitches = len(stitch_indices)
            self.split_map_input.setValue(stitches-1)
            self.split_map_input.setEnabled(False)
        else:
            # determine the required number of stitches based on max_px_per_stitch
            n_px = np.sum(px_per_line[x_start_index:x_stop_index+1])
            stitches = int(np.ceil(n_px/MAX_PX_PER_STITCH))
            stitches = max(stitches,self.split_map_input.value()+1)
            px_per_stitch  = int(np.ceil(n_px/stitches))

            # determine the stitch cutoff to ensure complete lines
            cutoff = np.cumsum(px_per_line[x_start_index:x_stop_index+1])%px_per_stitch
            if cutoff[-1] == 0:
                cutoff = cutoff[:-1]
                #cutoff = cutoff[:]

            # determine the start and end indices for each stitch
            stitch_indices = np.arange(1,len(cutoff))[np.diff(cutoff)<0]
            stitch_indices = np.insert(stitch_indices,0,0)
            stitch_indices += x_start_index
            stitch_indices = np.append(stitch_indices,x_stop_index+1)
            self.split_map_input.setValue(stitches-1)
            self.split_map_input.setEnabled(True)
        
        # update the number of splits
        self.mini_map_widget.setVLines(stitch_indices[1:-1])

        # get the acquisition time and frequency
        acq_time = self.acq_time_input.value()

        fast_velocity = fast_res/acq_time

        s_ = np.s_[stitch_indices[0]:stitch_indices[-1]]
        n_px = np.sum(px_per_line[s_])
        n_line = x_stop_index-x_start_index+1

        return_dur=0
        if not self.snake_checkbox.isChecked():
            return_dur = np.sum(np.abs(stop-start)/MAX_VELOCITY+LINE_OVERHEAD)

        exposure_dur = n_px*(acq_time-LATENCY_TIME)/60/60 # h
        overhead_dur = (n_px*(LATENCY_TIME)+(n_line-1)*LINE_OVERHEAD+return_dur)/60/60 # h
        total_dur = (exposure_dur+overhead_dur) # h
        # tot_dur = ((n_px*(acq_time)+(n_line-1)*LINE_OVERHEAD))/60/60 # h
        

        summary_text = ''
        summary_text += f'Estimated duration: {duration2string(total_dur)}' + '\n'
        summary_text += f'          overhead: {duration2string(overhead_dur)} ({overhead_dur/total_dur*100:.0f} %)' + '\n'
        summary_text += f'          exposure: {duration2string(exposure_dur)} ({exposure_dur/total_dur*100:.0f} %)' + '\n'
        summary_text += f'Pixels: {n_px}, lines: {n_line}' + '\n'
        summary_text += f'Fast motor velocity: {fast_velocity:.2f} mm/s' + '\n'
        
        for i in range(len(stitch_indices)-1):
            x_start = slow_fine[stitch_indices[i]]
            x_end = slow_fine[stitch_indices[i+1]-1]
            s_ = np.s_[stitch_indices[i]:stitch_indices[i+1]]
            n_px = np.sum(px_per_line[s_])
            n_line = stitch_indices[i+1]-stitch_indices[i]
            summary_text += f'slow motor range: {x_start:.4f} -> {x_end:.4f} ({n_px} px) ({n_line} lines)'+'\n'

        self.summary_textbrowser.setText(summary_text)

        if fast_velocity>MAX_VELOCITY:
            QMessageBox.warning(self, 'Warning', f'The fast scan velocity ({fast_velocity:.2f} mm/s) exceeds the maximum allowed velocity ({MAX_VELOCITY} mm/s)')

        motor1 = self.fast_axis_combobox.currentText()
        motor2 = self.slow_axis_combobox.currentText()
        self.json_dict = self._prepareJson(stitch_indices,start,stop,px_per_line,slow_fine,acq_time-LATENCY_TIME,LATENCY_TIME,motor1,motor2)
        
        #self.summary_textedit.setText(json.dumps(self.json_dict,indent=4).replace('"[','[').replace(']"',']'))
        save_text = os.path.basename(self.fname_input.text())
        save_text = save_text.replace('.h5','_{i}_of_') +f'{len(self.json_dict):02d}.json'
        self.save_map_lineedit.setText(save_text)

    def _prepareJson(self,stitch_indices,start,stop,px_per_line,x_fine,integ_time,latency_time,motor1,motor2):

        # Save scan commands to a separate json file for each stitch
        s = {}
        cmd_idx = 0
        non_empty_stitches = []
        for i in range(len(stitch_indices)-1):
            s_ = np.s_[stitch_indices[i]:stitch_indices[i+1]]
            non_empty_stitches.append(px_per_line[s_].sum()>0)
        for i, include in enumerate(non_empty_stitches):
            if not include:
                continue
            s_ = np.s_[stitch_indices[i]:stitch_indices[i+1]]
            scan_cmd={f'{i+1:02d} of {np.sum(non_empty_stitches):02d}':{
                    #"m1_start_pos": f"{list(np.round(start[s_],4))}",
                    #"m1_final_pos": f"{list(np.round(stop[s_],4))}",
                    #"m1_nr_interv": f"{list(px_per_line[s_])}",
                    "m1_start_pos": f"{[float(x) for x in np.round(start[s_],4)]}",
                    "m1_final_pos": f"{[float(x) for x in np.round(stop[s_],4)]}",
                    "m1_nr_interv": f"{[int(x) for x in px_per_line[s_]]}",
                    "m2_start_pos": np.round(x_fine[stitch_indices[i]],4),
                    "m2_final_pos": np.round(x_fine[stitch_indices[i+1]-1],4),
                                }}
            scan_cmd["integ_time"] = integ_time
            scan_cmd["latency_time"] = latency_time
            scan_cmd["motor1"] = motor1 # fast motor
            scan_cmd["motor2"] = motor2 # slow motor
            cmd_idx += 1
            key = f'{cmd_idx:02d}'
            s[key] = json.dumps(scan_cmd,indent=4).replace('"[','[').replace(']"',']')
        return s

    def saveJson(self):

        if self.json_dict is None:
            return
        if self.save_map_lineedit.text() == '':
            save_text = os.path.basename(self.fname_input.text())
            save_text = save_text.replace('.h5','_{i}_of_') +f'{len(self.json_dict):02d}.json'
            self.save_map_lineedit.setText(save_text)
        
        # ASSUMING DANMAX STRUCTURE
        dst = os.path.dirname(self.fname_input.text()).split('raw/')[0]
        dst = os.path.join(dst,'imesh_jsons')
        os.makedirs(dst,exist_ok=True)
        self.json_save_dir = dst
        dst = os.path.join(dst,self.save_map_lineedit.text())

        ignore_existing = False
        for key, value in self.json_dict.items():
            _dst = dst.replace('{i}',key)
            if not ignore_existing and os.path.isfile(_dst):
                reply = QMessageBox.question(self, 'File exists', f'The file {_dst} already exists. Do you want to overwrite it?',
                                             QMessageBox.StandardButton.YesToAll |QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                             QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.No:
                    # briefly change the text color to red
                    self.save_map_lineedit.setStyleSheet('color: red')
                    # set a timer to change the text color back to the default
                    timer = QtCore.QTimer(self)
                    timer.setSingleShot(True)
                    timer.timeout.connect(lambda: self.save_map_lineedit.setStyleSheet(""))
                    timer.start(1000)
                    return
                elif reply == QMessageBox.StandardButton.YesToAll:
                    ignore_existing = True

            with open(_dst, 'w') as f:
                f.write(value)
        
        # briefly change the text color to green
        self.save_map_lineedit.setStyleSheet('color: green')
        # set a timer to change the text color back to the default
        timer = QtCore.QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: self.save_map_lineedit.setStyleSheet(""))
        timer.start(1000)


    def browseFile(self):
        default_path = os.path.dirname(self.fname_input.text())
        if not os.path.isdir(default_path):
            default_path = DEFAULT_RAW_PATH
        fname = pg.QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', default_path, 'HDF5 files (*.h5)')[0]
        self.fname_input.setText(fname)
        text = os.path.basename(fname).replace('.h5','_01_of_01.json')
        self.save_map_lineedit.setPlaceholderText(text) # 'scan-xxxx_01_of_01.json'
        self.loadImage()

    def loadImage(self):
        fname = self.fname_input.text()
        if os.path.isfile(fname):
            im = self.openH5(fname)
        else:
            # change the text color to red
            self.fname_input.setStyleSheet('color: red')
            # set a timer to change the text color back to the default
            timer = QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda: self.fname_input.setStyleSheet(''))
            timer.start(1000)

    def saveMask(self):
        fname = os.path.basename(self.fname_input.text())
        if fname.startswith('scan-') and fname.endswith('.h5'):
            fname = fname[0:9]+'_mask.npy'
        else:
            fname = 'mask.npy'
        fname = pg.QtWidgets.QFileDialog.getSaveFileName(self, 'Save mask', fname, 'Numpy files (*.npy)')[0]
        if fname:
            mask = self.imask_widget.getCurrentMask()
            np.save(fname, mask)

    def openH5(self,fname):


        
        # azint_dset = 'entry/data/I'
        # with h5.File(fname, 'r') as f:
        #     I = f[azint_dset][:]

        # ## TEST IMAGE
        # im = I.reshape(66,87,-1)

        is_xrf = '_xspress3-dtc-2d' in fname
        fname = fname.replace('_pilatus_integrated.h5','.h5')
        fname = fname.replace('_xspress3-dtc-2d.h5','.h5')
        if not os.path.isfile(fname):
            return
        meta = self.getMetaDic(fname)

        # disconnect and clear the comboboxes
        self.fast_axis_combobox.currentTextChanged.disconnect(self.updateImage)
        self.slow_axis_combobox.currentTextChanged.disconnect(self.updateImage)
        self.signal_combobox.currentTextChanged.disconnect(self.updateImage)
        self.fast_axis_combobox.clear()
        self.slow_axis_combobox.clear()
        self.signal_combobox.clear()

        # add the keys of the meta data dictionary to the comboboxes
        self.fast_axis_combobox.addItems(meta.keys())
        self.slow_axis_combobox.addItems(meta.keys())
        self.fast_axis_combobox.setEnabled(True)
        self.slow_axis_combobox.setEnabled(True)

        if 'pd_sam_y' in meta:
            self.fast_axis_combobox.setCurrentText('pd_sam_y')
        elif 'pd_sam_x' in meta:
            self.fast_axis_combobox.setCurrentText('pd_sam_x')
        if 'pd_sam_x' in meta:
            self.slow_axis_combobox.setCurrentText('pd_sam_x')
        elif 'pd_huber' in meta:
            self.slow_axis_combobox.setCurrentText('pd_huber')
        elif 'pd_omega' in meta:
            self.slow_axis_combobox.setCurrentText('pd_omega')
        elif 'pd_sam_y' in meta:
            self.slow_axis_combobox.setCurrentText('pd_sam_y')

        signals = []
        with h5.File(fname, 'r') as f:

            if 'entry/instrument/pilatus' in f:
                signals.append('xrd')
            if 'entry/instrument/xspress3-dtc-2d' in f:
                signals.append('xrf')
            if 'entry/instrument/signal' in f:
                signals.append('signal')
            
            signals += [key for key in f['entry/instrument'].keys() if 'albaem-xrd' in key]
        self.signal_combobox.addItems(signals)
        self.signal_combobox.setEnabled(True)
        if is_xrf:
            self.signal_combobox.setCurrentText('xrf')
        
        # reconnect the comboboxes
        self.fast_axis_combobox.currentTextChanged.connect(self.updateImage)
        self.slow_axis_combobox.currentTextChanged.connect(self.updateImage)
        self.signal_combobox.currentTextChanged.connect(self.updateImage)

        self.updateImage()
 
    def updateImage(self):
        fname = self.fname_input.text()
        fname = fname.replace('_pilatus_integrated.h5','.h5')
        fname = fname.replace('_xspress3-dtc-2d.h5','.h5')

        I = self.getSignal(fname)

        fast, slow = self.getMotorPos(fname)
        # guess the fast and slow axis resolution
        fast_res = guessRes(fast,decimals=4)
        slow_res = guessRes(slow,decimals=4)

        # convert the flat signal to a 2d image
        im,slow,fast = irregular_flat2regular_grid(I,slow,fast,slow_res,fast_res)

        self.fast_pos = fast
        self.slow_pos = slow
        if not self.keep_aspect_button.isChecked():
            self.mask_aspect_ratio = 1
        else:
            self.mask_aspect_ratio = slow_res/fast_res # xScale/yScale

        self.imask_widget.setData(im,aspect=self.mask_aspect_ratio)

    def getSignal(self, fname):
        signal = self.signal_combobox.currentText()
        if signal == 'xrd':
            fname = fname.replace('/raw','/process/azint').replace('.h5','_pilatus_integrated.h5')
            with h5.File(fname, 'r') as f:
                if 'entry/data/I' in f:
                    I = f['entry/data/I'][:]
                elif 'entry/data1d/I' in f:
                    I = f['entry/data1d/I'][:]
        elif signal == 'xrf':
            with h5.File(fname, 'r') as f:
                I = np.squeeze(f['entry/instrument/xspress3-dtc-2d/data'][:])
        else:
            with h5.File(fname, 'r') as f:
                I = np.atleast_2d(f['entry/instrument/'+signal+'/data'][:]).T
        return I

    def getMotorPos(self, fname):
        fast_axis = self.fast_axis_combobox.currentText()
        slow_axis = self.slow_axis_combobox.currentText()
        pos = []
        with h5.File(fname, 'r') as f:
            for ax in [fast_axis, slow_axis]:
                if ax not in f['entry/instrument/']:
                    raise ValueError(f'{ax} not found in the meta data')
                gr = f['entry/instrument/'+ax]
                if 'value' in gr:
                    pos.append(gr['value'][:])
                elif 'data' in gr:
                    pos.append(gr['data'][:])
        return pos

    def getMetaDic(self,fname):
        """Return dictionary of available meta data, reusing the .h5 dictionary keys."""
        if 'master.h5' in fname:
            fname = fname.replace('raw', 'process/azint').replace('.h5','_meta.h5')
        data = {}
        with h5.File(fname,'r') as f:
            for key in f['/entry/instrument/'].keys():
                # if key != 'pilatus' and key != 'start_positioners':
                if not key in ['pilatus', 'start_positioners', 'xspress3-dtc-2d']:
                    for k in f['/entry/instrument/'][key].keys():
                        if k in ['value','data']:
                            data[key] = f['/entry/instrument/'][key][k][:]
        return data

class SequenceWidget(QWidget):
    def __init__(self,parent=None):
        super(SequenceWidget, self).__init__(parent)

        central_layout = QHBoxLayout(self)

        # add a group box for the file tree
        self.file_tree_groupbox = QGroupBox('Available files')
        file_tree_layout = QVBoxLayout()
        self.file_tree_groupbox.setLayout(file_tree_layout)
        central_layout.addWidget(self.file_tree_groupbox,1)

        # add a file tree widget
        self.file_tree_widget = QTreeWidget()
        self.file_tree_widget.setHeaderLabels(['Name','Fast motor','Slow motor','Pixels','Lines','Duration'])
        self.file_tree_widget.setColumnWidth(0,300)
        self.file_tree_widget.setColumnWidth(1,80)
        self.file_tree_widget.setColumnWidth(2,80)
        self.file_tree_widget.setColumnWidth(3,75)
        self.file_tree_widget.setColumnWidth(4,75)
        self.file_tree_widget.setSortingEnabled(True)
        self.file_tree_widget.setDragEnabled(True)
        self.file_tree_widget.itemDoubleClicked.connect(self.addItemToSequence)
        self.file_tree_widget.setExpandsOnDoubleClick(False)

        file_tree_layout.addWidget(self.file_tree_widget,1)

        # add a group box for the sequence tree
        self.sequence_tree_groupbox = QGroupBox('Sequence')
        sequence_tree_layout = QVBoxLayout()
        self.sequence_tree_groupbox.setLayout(sequence_tree_layout)
        central_layout.addWidget(self.sequence_tree_groupbox,1)

        # add a toolbar to the sequence tree
        toolbar = pg.QtWidgets.QToolBar()        
        sequence_tree_layout.addWidget(toolbar)

        self._initSequenceToolbar(toolbar)



        # add a sequence tree widget
        self.sequence_tree_widget = QTreeWidget()
        self.sequence_tree_widget.setHeaderLabels(['Name','Fast motor','Slow motor','Pixels','Lines','Duration'])
        self.sequence_tree_widget.setColumnWidth(0,250)
        self.sequence_tree_widget.setColumnWidth(1,80)
        self.sequence_tree_widget.setColumnWidth(2,75)
        self.sequence_tree_widget.setColumnWidth(3,75)
        self.sequence_tree_widget.setColumnWidth(4,75)
        # allow items from the file tree to be dropped on the sequence tree
        self.sequence_tree_widget.setAcceptDrops(True)
        self.sequence_tree_widget.setDragEnabled(True)
        self.sequence_tree_widget.setDropIndicatorShown(True)
        # self.sequence_tree_widget.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        sequence_tree_layout.addWidget(self.sequence_tree_widget,1)

        self.sequence_tree_widget.dropEvent = self.seqTreeDropEvent

        # add a horizontal layout for total duration label and save sequence button
        layout = QHBoxLayout()
        sequence_tree_layout.addLayout(layout)

        # add a label for the total duration
        self.total_duration_label = QLabel('Total duration:')
        font = pg.QtGui.QFont()
        font.setBold(True)
        self.total_duration_label.setFont(font)
        layout.addWidget(self.total_duration_label)

        # add a stretch
        layout.addStretch(1)

        # add a save sequence button
        self.save_sequence_button = QPushButton('Save sequence')
        self.save_sequence_button.clicked.connect(self.saveSequence)
        layout.addWidget(self.save_sequence_button)
     


        # central_layout.addStretch(1)
    def _initSequenceToolbar(self,toolbar):

        # add a button to add a destination tree item
        self.add_dst_button = QToolButton()
        pmap = pg.QtGui.QPixmap(':/icons/new_folder')
        icon = pg.QtGui.QIcon(pmap)
        self.add_dst_button.setIcon(icon)
        self.add_dst_button.clicked.connect(self.add_dst_item)
        toolbar.addWidget(self.add_dst_button)

        # add a button to move the selected item up
        self.move_up_button = QToolButton()
        pmap = pg.QtGui.QPixmap(':/icons/move_up')
        icon = pg.QtGui.QIcon(pmap)
        self.move_up_button.setIcon(icon)
        self.move_up_button.clicked.connect(self.move_seq_item_up)
        toolbar.addWidget(self.move_up_button)

        # add a button to move the selected item down
        self.move_down_button = QToolButton()
        pmap = pg.QtGui.QPixmap(':/icons/move_down')
        icon = pg.QtGui.QIcon(pmap)
        self.move_down_button.setIcon(icon)
        self.move_down_button.clicked.connect(self.move_seq_item_down)
        toolbar.addWidget(self.move_down_button)

        # add a button to add the selected item to the sequence tree
        self.add_button = QToolButton()
        pmap = pg.QtGui.QPixmap(':/icons/item_add')
        icon = pg.QtGui.QIcon(pmap)
        self.add_button.setIcon(icon)
        self.add_button.clicked.connect(lambda: self.addItemToSequence(self.file_tree_widget.currentItem(),None))
        toolbar.addWidget(self.add_button)

        # add a button to delete the selected item
        self.delete_button = QToolButton()
        pmap = pg.QtGui.QPixmap(':/icons/item_delete')
        icon = pg.QtGui.QIcon(pmap)
        self.delete_button.setIcon(icon)
        self.delete_button.clicked.connect(self.del_seq_item)
        toolbar.addWidget(self.delete_button)


        # add a button to clear the sequence tree
        self.clear_button = QToolButton()
        pmap = pg.QtGui.QPixmap(':/icons/clear_brush')
        icon = pg.QtGui.QIcon(pmap)
        self.clear_button.setIcon(icon)
        self.clear_button.clicked.connect(lambda: self.sequence_tree_widget.clear())
        self.clear_button.clicked.connect(self.updateTotalDuration)
        toolbar.addWidget(self.clear_button)


    def add_dst_item(self):
        # add a destination item to the sequence tree
        # get the destination name from the user
        dst, ok = pg.QtWidgets.QInputDialog.getText(self, 'Destination', 'Enter destination name:')
        if not ok:
            return
        # replace spaces and special characters with underscores
        dst = re.sub(r'\W+', '_', dst)

        if dst == '':
            dst = 'Type here'
        font = pg.QtGui.QFont()
        font.setItalic(True)
        font.setBold(True)
        item = QTreeWidgetItem([dst])
        item.setFont(0,font)
        item.setBackground(0,pg.QtGui.QBrush(pg.QtGui.QColor(255,255,0,100)))
        item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
        self.sequence_tree_widget.addTopLevelItem(item)

    def move_seq_item_up(self):
        # move the selected item up in the sequence tree
        item = self.sequence_tree_widget.currentItem()
        if item is None:
            return
        parent = item.parent()
        if parent is None:
            index = self.sequence_tree_widget.indexOfTopLevelItem(item)
        else:
            index = parent.indexOfChild(item)
        if index > 0:
            if parent is None:
                item = self.sequence_tree_widget.takeTopLevelItem(index)
                self.sequence_tree_widget.insertTopLevelItem(index-1,item)
            else:
                item = parent.takeChild(index)
                parent.insertChild(index-1,item)
            self.sequence_tree_widget.setCurrentItem(item)

    def move_seq_item_down(self):
        # move the selected item down in the sequence tree
        item = self.sequence_tree_widget.currentItem()
        if item is None:
            return
        parent = item.parent()
        if parent is None:
            index = self.sequence_tree_widget.indexOfTopLevelItem(item)
            new_index = min(index+1,self.sequence_tree_widget.topLevelItemCount()-1)
        else:
            index = parent.indexOfChild(item)
            new_index = min(index+1,parent.childCount()-1)
        if parent is None:
            item = self.sequence_tree_widget.takeTopLevelItem(index)
            self.sequence_tree_widget.insertTopLevelItem(new_index,item)
        else:
            item = parent.takeChild(index)
            parent.insertChild(new_index,item)
        self.sequence_tree_widget.setCurrentItem(item)

    def del_seq_item(self):
        # delete the selected item from the sequence tree
        item = self.sequence_tree_widget.currentItem()
        if item is not None:
            parent = item.parent()
            if parent is None:
                self.sequence_tree_widget.takeTopLevelItem(self.sequence_tree_widget.indexOfTopLevelItem(item))
            else:
                parent.takeChild(parent.indexOfChild(item))
        self.updateTotalDuration()

    def seqTreeDropEvent(self,event):
        # check if the source is the file tree widget
        if event.source() == self.file_tree_widget:
            # add the selected item and all its children to the sequence tree
            for item in event.source().selectedItems():
                if item.parent() is not None:
                    name = item.parent().text(0)+'_'+item.text(0)
                else:
                    name = item.text(0)
                parent = QTreeWidgetItem([name]+[item.text(i) for i in range(1,6)])
                for i in range(item.childCount()):
                    child = QTreeWidgetItem([item.child(i).text(j) for j in range(6)])
                    parent.addChild(child)
                self.sequence_tree_widget.addTopLevelItem(parent)
            event.accept()
        else:
            event.ignore()
        self.updateTotalDuration()

    def addItemToSequence(self,item,_):
        # add the selected item and all its children to the sequence tree
        if item.parent() is not None:
            name = item.parent().text(0)+'_'+item.text(0)
        else:
            name = item.text(0)
        parent = QTreeWidgetItem([name]+[item.text(i) for i in range(1,6)])
        for i in range(item.childCount()):
            child = QTreeWidgetItem([item.child(i).text(j) for j in range(6)])
            parent.addChild(child)
        self.sequence_tree_widget.addTopLevelItem(parent)
        self.updateTotalDuration()

    def updateFileTree(self,src_dir):
        self.file_tree_groupbox.setTitle(f'Available files in {src_dir}')
        self.file_tree_widget.clear()
        # find all available .json files
        json_files = glob.glob(os.path.join(src_dir,'*.json'))
        json_files = [os.path.basename(f) for f in json_files]
        json_files = sorted(json_files)

        # group the json files by base name, excluding the _01_of_01.json suffix
        keys = set([f[:-14] for f in json_files])
        json_dict = {key:[] for key in keys}
        for f in json_files:
            key = f[:-14]
            # read relevant information from the json files
            fast_motor,slow_motor,n_px, n_lines, acquisition_time = self.readJson(os.path.join(src_dir,f))
            duration = (acquisition_time*n_px+LINE_OVERHEAD*(n_lines-1))/60/60 # h
            duration = duration2string(duration)
            json_dict[key].append([f[-13:],fast_motor,slow_motor,f'{n_px}',f'{n_lines}',duration,acquisition_time])

        # add the json dictionary to the tree widget
        for key, value in json_dict.items():
            # value: [suffix,fast_motor,slow_motor,n_px,n_lines]
            fast_motor = value[0][1]
            slow_motor = value[0][2]
            acquisition_time = value[0][6]
            n_px = np.sum([int(v[3]) for v in value])
            n_lines = np.sum([int(v[4]) for v in value])
            duration = (acquisition_time*n_px+LINE_OVERHEAD*(n_lines-1))/60/60 # h
            duration = duration2string(duration)
            parent = QTreeWidgetItem([key,fast_motor,slow_motor,f'{n_px}',f'{n_lines}',duration])
            for v in value:
                child = QTreeWidgetItem(v[:-1])
                parent.addChild(child)
            self.file_tree_widget.addTopLevelItem(parent)

        self.file_tree_widget.sortByColumn(0,QtCore.Qt.SortOrder.AscendingOrder)

    def readJson(self,fname):
        """
        Simple function to read a json file.
        Return the fast and slow motor positions, the number of pixels and the number of lines.
        """
        try:
            with open(fname, 'r') as f:
                data = json.load(f)
        except JSONDecodeError:
            print('Bad json file: {fname}')
        if 'motor1' in data:
            fast_motor = data['motor1']
        else:
            fast_motor = 'None'
        if 'motor2' in data:
            slow_motor = data['motor2']
        else:
            slow_motor = 'None'
        acquisition_time = data['integ_time'] + data['latency_time']
        n_px = 0
        n_lines = 0
        for key, value in data.items():
            if isinstance(value,dict):
                if "m1_nr_interv" in value:
                    n_px += np.sum(value["m1_nr_interv"])
                    n_lines += len(value["m1_nr_interv"])
        return fast_motor,slow_motor,n_px, n_lines, acquisition_time
        
    def saveSequence(self):

        # set folder cmd: 'setFolder [folder_name]'
        # imesh cmd: 'imeshct [json file path]'
        
        if self.sequence_tree_widget.topLevelItemCount() == 0:
            return
        cmd = DEFAULT_SCAN_CMD
        src_dir = self.file_tree_groupbox.title().split(' ')[-1]
        sequence_text = ''
        for i in range(self.sequence_tree_widget.topLevelItemCount()):
            item = self.sequence_tree_widget.topLevelItem(i)
            # check if the item is a destination item
            # by checking if column 1 is empty
            if item.text(1) == '':
                dst = item.text(0)
                if dst == 'Type here':
                    continue
                # replace spaces and special characters with underscores
                dst = dst.replace(' ','_').replace('.','_').replace(',', '_').replace('/','_').replace('\\','_')
                # remove any double underscores
                dst = dst.replace('__','_')
                item.setText(0,dst)
                sequence_text += '# change folder\n'
                sequence_text += 'setFolder ' + dst + '\n'
                continue
            sequence_text += '# ' + item.text(0) + '\n'
            if item.childCount() == 0:
                sequence_text += f'# {item.text(5)}'+ '\n'
                json_path = os.path.join(src_dir,item.text(0))
                sequence_text += cmd + f' "{json_path}"'  + '\n'
            for j in range(item.childCount()):
                if j<1:
                    sequence_text += f'# {item.text(5)}'+ '\n'
                child = item.child(j)
                json_path = os.path.join(src_dir,item.text(0)+'_'+child.text(0))
                sequence_text += cmd + f' "{json_path}"'  + '\n'
        sequence_text = sequence_text.strip()
        # save the sequence text to a file
        default_save_path = os.path.join(src_dir,'sequence.txt')
        fname = pg.QtWidgets.QFileDialog.getSaveFileName(self, 'Save sequence', default_save_path, 'Text files (*.txt)')[0]
        if fname:
            with open(fname, 'w') as f:
                f.write(sequence_text)
            # briefly change the text color to green
            self.save_sequence_button.setStyleSheet('color: green')
            # set a timer to change the text color back to the default
            timer = QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda: self.save_sequence_button.setStyleSheet(""))
            timer.start(1000)

    def updateTotalDuration(self):
        total_duration = 0
        for i in range(self.sequence_tree_widget.topLevelItemCount()):
            item = self.sequence_tree_widget.topLevelItem(i)
            if item.text(5) == '':
                continue
            total_duration += string2duration(item.text(5)) # h
        # round up to nearest 5 minutes
        total_duration = np.ceil(total_duration*60/5)*5/60+1e-6 # h
        total_duration = duration2string(total_duration).split('min')[0]+'min'
        self.total_duration_label.setText('Total duration: ' + total_duration)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("IMapWidget")
        self.resize(1600,1000)
        #self.central_widget = IMapWidget(self)
        #self.setCentralWidget(self.central_widget)

        # add a tab widget as the central widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # add a tab for the IMapWidget
        self.imap_widget = IMapWidget()
        self.imap_widget.sigIMaskClicked.connect(self.updateStatusBar)
        self.tab_widget.addTab(self.imap_widget,'Segmentation')

        # add a tab for the SequenceWidget
        self.sequence_widget = SequenceWidget()
        self.tab_widget.addTab(self.sequence_widget,'Sequence')
        self.tab_widget.currentChanged.connect(self.tabChanged)



        #self.statusBar().showMessage('Ready')

    def tabChanged(self,index):
        if index == 1:
            src_dir = self.imap_widget.json_save_dir
            if src_dir is None:
                src_dir = DEFAULT_JSON_DIR
            self.sequence_widget.updateFileTree(src_dir)

    def updateStatusBar(self,pos):
        if pos is None:
            self.statusBar().clearMessage()
        else:
            self.statusBar().showMessage(f'X: {pos[0]:.2f}, Y: {pos[1]:.2f}')

if __name__ == '__main__':   
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
