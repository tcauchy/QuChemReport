## -*- encoding: utf-8 -*-

from quchemreport.utility_services import units

## Default parameters

## Orbkit parameters
# default grid step size
# Spacing of cubic voxels in the grid 
obk_step = 0.075 # Depending on the molecule size, it should be adapted. 
# Oversizing the grid aroung the molecule in Bohr radii
# ORBKIT uses 5 by default, tune this as required
obk_extand = 5

## Gaussian broadening default parameters
FWHM = 3000 # 3000 cm-1 for full width at half maximum of gaussian band. 


## Mayavi parameters
# Set image size in pixels
img_width = 2400
img_height = 2400
# In mayavi the elevation is the theta angle from the z axis (physics convention) in spherical coordinates. 
# The azimuth is the phi angle on the x-y plane. 
# By default the elevation is set to 55° and azimuth to around 55°
# The thing is that the automatic positining of the z axis will depend on the molecule shape
# Case 1:
# For planar compound with a rectangular shape the z axis is PERPENDICULAR to the maximun number of atoms. Elevation near 10° is adapted
# Changing the azimuth will rotate the molecule around the camera axis favoring a portrait orientation with an azimuth 0°  
# Therefore the optimal view for a landscape orientation should correspond to an azimuth of 90°.
elev_angle_cam1 = 10
azimuth_cam1 =  90

# Case 2 : 
# For linear or planar compound with a square shape the z axis is ALONG the maximun number of atoms. Elevation near 0° is NOT adapted
# Best elevation should be around 80° to have a little bit of perspective (camera will therefore be near the x or y axis). 
# For linear compounds changing the azimuth will not  change the view : rotation around the camera axis   
# However for planar molecules with a square shape changing the azimuth to 90° or 270° and an elevation of 80° 
# will bring back the atoms again along the camera axis. Not adapted 
elev_angle_cam2 = 80
azimuth_cam2 =  10

# isosurface opacity
surf_opacity = 1

colors = {"OM" : (0.4, 0, 0.235),  #For OM positive surface Tyrian purple
          "EDD" : (0.0, 0.5, 0.5),  #For Electron density difference positive surface metallic blue
          "Oif" : (0.95, 0.5, 0.0),  #For transitions wavefunctions overlap positive surface an orange
          "NEG" : (0.95, 0.90, 0.93),  #For isosurfaces negative surfaces almost white
          "CTDIP" : (0.0, 0.5, 0.5), #For transitions charge transfer dipole same as EDD surface
          "GSDIP" : (0.1, 0.1, 0.1),
          "ELDIP" : (0, 0.6, 1),
          "VELDIP" : (0.6, 1, 0.2),
          "MAGDIP" : (1, 0.2, 0.6),
          "OVDIP" : (0.95, 0.5, 0.) #For transitions wavefunctions overlap dipole same as Oif surface 
          }
scale = {"CTDIP" : 1., #For transitions charge transfer dipole same as EDD surface
          "GSDIP" : 1.,
          "ELDIP" : 1./units.A_to_a0,
          "VELDIP" : 1./units.A_to_a0,
          "MAGDIP" : 1./units.A_to_a0,
          "OVDIP" : 1. #For transitions wavefunctions overlap dipole same as Oif surface 
          }

# TODO : 
THRESHOLD_IR_INTENSITIES = 50.0
