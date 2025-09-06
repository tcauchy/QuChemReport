
## -*- encoding: utf-8 -*-

from mayavi import mlab
import numpy as np
from sys import exit
import os
from math import sqrt, acos, atan2

from sklearn.decomposition import PCA

from quchemreport.utility_services.log_data import LogData

#from utils import A_to_a0
from quchemreport.utility_services import units

## Import mayavi defaults parameters
from quchemreport.utility_services.parameters import img_width, img_height, surf_opacity, colors, scale, azimuth_cam1, azimuth_cam2, elev_angle_cam1, elev_angle_cam2
width = img_width
height = img_height

## No screen
mlab.options.offscreen = True
#mlab.options.offscreen = False

## Initialize visualization details common to all jobs
with open('%s/../utility_services/Atoms_properties.csv' % os.path.dirname(__file__), "r") as f:
    tab = [line.split() for line in f]


def _init_scene(elements_3D_coords, atom_pairs, atoms_Z):
    u"""Initializes the MayaVi scene.

    ** Parameters **
      j_data : dict
    Data on the molecule, as deserialized from the scanlog format.

    ** Returns **
      figure : mayavi.core.scene.Scene
    The MayaVi scene with the atoms plotted.
    """
      
    if (
      elements_3D_coords == None
      or atom_pairs == None
      or atoms_Z == None
    ):
      raise ValueError(
        "Missing required data to initialize scene: "
        "elements_3D_coords, atom_pairs, and atoms_Z must all be provided."
      )
    
    figure = mlab.figure(bgcolor=(1,1,1), fgcolor=(0,0,0))
    figure.scene.disable_render= True
    geom = np.array(elements_3D_coords).reshape((-1,3))/units.A_to_a0
    conn = atom_pairs
    atom_nums = atoms_Z
    ## Draw atoms and bonds
    for i, atom in enumerate(atom_nums):
        p, color = geom[i], tuple(float(x)/255.0 for x in tab[atom][3:6])
        ## Requires >=MayaVi-4.6.0
        mlab.points3d([p[0]], [p[1]], [p[2]],
                      figure=figure, mode='sphere', color=color, resolution=15, scale_factor=0.5)
    for pair in conn:
        att1 = tab[atom_nums[pair[0]]]
        p1, p2 = geom[pair[0]], geom[pair[1]]
        color = tuple(float(x)/255.0 for x in att1[3:6])
        mlab.quiver3d([p1[0]],         [p1[1]],         [p1[2]], 
                      [p2[0] - p1[0]], [p2[1] - p1[1]], [p2[2] - p1[2]],
                      figure=figure, mode='cylinder', color=color, resolution=15, scale_factor=0.5)

    return figure

# TODO : Utiliser caméra auto seulement
def _set_cam(figure, cam):
    if cam == "cam1":
        mlab.view(azimuth=azimuth_cam1, elevation=elev_angle_cam1, figure=figure)         # First vue is almost from above our calculated normal
        #print(mlab.view(figure=figure))
    elif cam == "cam2":
        mlab.view(azimuth=azimuth_cam2, elevation=elev_angle_cam2, figure=figure)
        #print(mlab.view(figure=figure))

    return figure

## Set the Iso contour value for mayavi from a percent
# Choose the % (between 0 to 100) of the positive values to show in picture.
def CalcCutOff(data,IsoContourPercent=30):
    #data are the function values for each voxels. Transforma as a list, sort and select the positive values
    datar = data.ravel()
    datasortP = np.sort(datar[datar>0])
    datasortN = np.sort(np.abs(datar[datar<0]))
    #cumulative sum of function values, normalize
    cumdatasortP=np.cumsum(datasortP)
    cumdatasortPnorm=cumdatasortP/cumdatasortP[-1]*100.
    cumdatasortN=np.cumsum(datasortN)
    cumdatasortNnorm=cumdatasortN/cumdatasortN[-1]*100.
    return datasortP[cumdatasortPnorm>=(100.-IsoContourPercent)][0], -datasortN[cumdatasortNnorm>=(100.-IsoContourPercent)][0]

## Visualize

def get_molecule_bounds(coords):
    """Calculate the bounding box of the molecule."""
    coords = np.array(coords)
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    size = max_coords - min_coords
    max_dimension = np.max(size)
    return max_dimension

def get_pca_camera_auto(coords, fov_deg=25, margin=1.2, perspective_offset_ratio=0.05):
    """
    Automatically calculate camera position and focus to view the entire molecule.
    
    Parameters
    ----------
    coords : array-like
        Atomic coordinates
    fov_deg : float, default=25
        Vertical field of view in degrees
    margin : float, default=1.5
        Scene enlargement factor
    perspective_offset_ratio : float, default=0.1
        Lateral offset ratio for perspective
    
    Returns
    -------
    cam : np.ndarray
        Camera position
    foc : np.ndarray
        Focus point
    """
    coords = np.array(coords)
    
    # Calculate center of mass
    center = np.mean(coords, axis=0)
    coords_centered = coords - center
    
    # PCA analysis
    pca = PCA(n_components=3)
    pca.fit(coords_centered)
    
    # Get normal vector (viewing direction)
    normal = pca.components_[-1]  # Last component gives the normal to the main plane
    
    # Get the two main axes in the plane
    x_axis = pca.components_[0]
    y_axis = pca.components_[1]

    # Calculate molecule extent
    max_dimension = get_molecule_bounds(coords)
  
    # Alternative method: use the actual bounding box
    bbox_extent = max_dimension * margin
    
    # Calculate distance based on field of view
    theta = np.radians(fov_deg / 2)
    if theta > 0:
        r = bbox_extent / (2 * np.tan(theta))
    else:
        r = bbox_extent * 2  # Fallback
    
    # Ensure minimum distance
    r = max(r, max_dimension * 2)
    
    # Calculate perspective offset
    #offset = np.zeros(3)
    offset = perspective_offset_ratio * r * (x_axis + y_axis)
    
    # Camera position: along normal + small lateral offset for perspective
    cam = center + r * normal + offset
    
    # Focus on center of mass
    foc = center
    
    return cam, foc, x_axis, y_axis, normal

def add_pca_plane(coords, figure, size=3.0, resolution=10, opacity=0.3):
    center = np.mean(coords, axis=0)
    coords_centered = coords - center
    pca = PCA(n_components=3)
    pca.fit(coords_centered)
    v1, v2 = pca.components_[0], pca.components_[1]  # vecteurs dans le plan

    # Grille pour représenter le plan
    grid = np.linspace(-size, size, resolution)
    xx, yy = np.meshgrid(grid, grid)
    plane = center + xx[..., None]*v1 + yy[..., None]*v2

    # Affichage
    mlab.mesh(plane[:, :, 0], plane[:, :, 1], plane[:, :, 2],
            color=(1, 1, 0), opacity=opacity, figure=figure)
    
    return figure

def move_camera_with_pca(coords, figure, r=5.0, perspective_offset=0.05):
    cam, foc, x_axis, y_axis, normal = get_pca_camera_auto(coords, fov_deg=25, margin=1.2, perspective_offset_ratio=perspective_offset)
    
    figure.scene.disable_render = False
    scene = figure.scene
    
    scene.camera.position = list(cam)
    scene.camera.focal_point = list(foc)
    
    scene.camera.compute_view_plane_normal()
    scene.camera.view_up = tuple(y_axis / np.linalg.norm(y_axis))
    scene.render()

    return figure, normal

## Visualize
def topoWithPCA(elements_3D_coords, atom_pairs, atoms_Z, file_name=None, size=(width,height)):
    u"""Creates the topological view of the molecule.

    ** Parameters **
      j_data : dict
    Data on the molecule, as deserialized from the scanlog format.
      file_name : str, optional
    Base name of the file in which to save the image.
      size : tuple(int, int), optional
    The size of the image to save.
    """
    geom = np.array(elements_3D_coords).reshape((-1,3))/units.A_to_a0
    figure = _init_scene(elements_3D_coords, atom_pairs, atoms_Z)  
    figure, normal = move_camera_with_pca(geom, figure, r=12.0)
    label_offset = 0.5 * normal 

    ## Show labels and numbers ( = indices + 1 )
    for i, atom in enumerate(atoms_Z):
        P, label = geom[i], tab[atom][1]
        P_label = P + label_offset  # offset in camera direction
        mlab.text3d(*P_label, label + str(i + 1), color=(0, 0, 0), scale=0.2, figure=figure)
    if file_name is not None:
        mlab.savefig("{}-TOPOLOGY.png".format(file_name), figure=figure, size=size)    
    mlab.close()

def viz_MO(data, X, Y, Z, elements_3D_coords, atom_pairs, atoms_Z, file_name=None, labels=None, size=(width,height)):
    u"""Visualizes the molecular orbitals of the molecule.

    ** Parameters **
      data : list(numpy.ndarray)
    List of series of voxels containing the scalar values of the molecular orbitals to plot.
      X, Y, Z
    Meshgrids as generated by numpy.mgrid, for positioning the voxels.
      j_data : dict
    Data on the molecule, as deserialized from the scanlog format.
      file_name : str, optional
    Base name of the files in which to save the images.
      labels : list(str), optional
    Labels to append to `file_name` for each series in `data`. If None, the position of the series is appended.
      size : tuple(int, int), optional
    The size of the image to save.
    """
    
    # Skip if missing data
    if (
      data is None
      or X is None
      or Y is None
      or Z is None
      or elements_3D_coords is None
      or atom_pairs is None
      or atoms_Z is None
    ):
      return

    figure = _init_scene(elements_3D_coords, atom_pairs, atoms_Z)
    for i, series in enumerate(data):
        Cutoffp, Cutoffn = CalcCutOff(series,IsoContourPercent=20) 
        #print(Cutoffp, Cutoffn)
        MO_data = mlab.pipeline.scalar_field(X, Y, Z, series, figure=figure)        
        MOp = mlab.pipeline.iso_surface(MO_data, figure=figure, contours=[ Cutoffp ], color=colors["OM"], opacity=surf_opacity)
        MOn = mlab.pipeline.iso_surface(MO_data, figure=figure, contours=[ Cutoffn ], color=colors["NEG"], opacity=surf_opacity)
        if file_name is not None:
            figure = _set_cam(figure, "cam1")
            mlab.savefig("{}-MO-{}.png".format(file_name, labels[i] if labels is not None else i), figure=figure, size=size)
            figure = _set_cam(figure, "cam2")
            mlab.savefig("{}-MO-{}_cam2.png".format(file_name, labels[i] if labels is not None else i), figure=figure, size=size)
        MO_data.remove()
        MOp.remove()
        MOn.remove()
    mlab.close()

def viz_EDD(data, X, Y, Z, elements_3D_coords, atom_pairs, atoms_Z, et_sym, file_name=None, labels=None, size=(width,height)):
    u"""Visualizes the electron density differences for the transitions of the molecule.

    ** Parameters **
      data : list(numpy.ndarray)
    Voxels containing the scalar values of the electron density differences to plot.
      X, Y, Z : numpy.ndarray
    Meshgrids as generated by numpy.mgrid, for positioning the voxels.
      j_data : dict
    Data on the molecule, as deserialized from the scanlog format.
      file_name : str, optional
    Base name of the files in which to save the images.
      labels : list(str), optional
    Labels to append to `file_name` for each series in `data`. If None, the position of the series is appended.
      size : tuple(int, int), optional
    The size of the image to save.
    """

    figure = _init_scene(elements_3D_coords, atom_pairs, atoms_Z)
    for i, series in enumerate(data):
        Cutoffp, Cutoffn = CalcCutOff(series,IsoContourPercent=30) 
        D_data = mlab.pipeline.scalar_field(X, Y, Z, series, figure=figure)
        Dp = mlab.pipeline.iso_surface(D_data, figure=figure, contours=[ Cutoffp ], color=colors["EDD"], opacity=surf_opacity)
        Dn = mlab.pipeline.iso_surface(D_data, figure=figure, contours=[ Cutoffn ], color=colors["NEG"], opacity=surf_opacity)
        if file_name is not None:
            if 'Singlet' in et_sym:
                figure = _set_cam(figure, "cam1")
                mlab.savefig("{}-EDD-S{}.png".format(file_name, labels[i] if labels is not None else i), figure=figure, size=size)
                figure = _set_cam(figure, "cam2")
                mlab.savefig("{}-EDD-S{}_cam2.png".format(file_name, labels[i] if labels is not None else i), figure=figure, size=size)
            elif 'Triplet' in et_sym:
                figure = _set_cam(figure, "cam1")
                mlab.savefig("{}-EDD-T{}.png".format(file_name, labels[i] if labels is not None else i), figure=figure, size=size)
                figure = _set_cam(figure, "cam2")
                mlab.savefig("{}-EDD-S{}_cam2.png".format(file_name, labels[i] if labels is not None else i), figure=figure, size=size)
            else:
                figure = _set_cam(figure, "cam1")
                mlab.savefig("{}-EDD-{}.png".format(file_name, labels[i] if labels is not None else i), figure=figure, size=size)
                figure = _set_cam(figure, "cam2")
                mlab.savefig("{}-EDD-{}_cam2.png".format(file_name, labels[i] if labels is not None else i), figure=figure, size=size)
        D_data.remove()
        Dp.remove()
        Dn.remove()

    mlab.close()

def viz_Oif(data, X, Y, Z, elements_3D_coords, atom_pairs, atoms_Z, et_sym, file_name=None, labels=None, size=(width,height)):
    u"""Visualizes the overlap between the initial and final wavefunctions for the transitions of the molecule.

    ** Parameters **
      data : list(numpy.ndarray)
    Voxels containing the scalar values of the electron density differences to plot.
      X, Y, Z : numpy.ndarray
    Meshgrids as generated by numpy.mgrid, for positioning the voxels.
      j_data : dict
    Data on the molecule, as deserialized from the scanlog format.
      file_name : str, optional
    Base name of the files in which to save the images.
      labels : list(str), optional
    Labels to append to `file_name` for each series in `data`. If None, the position of the series is appended.
      size : tuple(int, int), optional
    The size of the image to save.

    ** Returns **
      figure : mayavi.core.scene.Scene
    The MayaVi scene containing the visualization.
    """

    figure = _init_scene(elements_3D_coords, atom_pairs, atoms_Z) 
    for i, series in enumerate(data):
        Cutoffp, Cutoffn = CalcCutOff(series,IsoContourPercent=30) 
        O_data = mlab.pipeline.scalar_field(X, Y, Z, series, figure=figure)
        Op = mlab.pipeline.iso_surface(O_data, figure=figure, contours=[ Cutoffp ], color=colors["Oif"], opacity=surf_opacity)
        On = mlab.pipeline.iso_surface(O_data, figure=figure, contours=[ Cutoffn ], color=colors["NEG"], opacity=surf_opacity)
        if file_name is not None:
            if 'Singlet' in et_sym:
                figure = _set_cam(figure, "cam1")
                mlab.savefig("{}-Oif-S{}.png".format(file_name, labels[i] if labels is not None else i), figure=figure, size=size)
                figure = _set_cam(figure, "cam2")
                mlab.savefig("{}-Oif-S{}_cam2.png".format(file_name, labels[i] if labels is not None else i), figure=figure, size=size)
            elif 'Triplet' in et_sym:
                figure = _set_cam(figure, "cam1")
                mlab.savefig("{}-Oif-T{}.png".format(file_name, labels[i] if labels is not None else i), figure=figure, size=size)
                figure = _set_cam(figure, "cam2")
                mlab.savefig("{}-Oif-T{}_cam2.png".format(file_name, labels[i] if labels is not None else i), figure=figure, size=size)
            else:
                figure = _set_cam(figure, "cam1")
                mlab.savefig("{}-Oif-{}.png".format(file_name, labels[i] if labels is not None else i), figure=figure, size=size)
                figure = _set_cam(figure, "cam2")
                mlab.savefig("{}-Oif-{}_cam2.png".format(file_name, labels[i] if labels is not None else i), figure=figure, size=size)
        O_data.remove()
        Op.remove()
        On.remove()
    mlab.close()

def viz_dip(data, elements_3D_coords, atom_pairs, atoms_Z, et_sym, file_name=None, labels=None, size=(width,height)):
    u"""Visualizes the electric transition dipole moment.

    ** Parameters **
      data : dictionnary of vectors positions tuples ([end],[origin]) like (array([ 0.0869, -0.0094,  0.0415]), array([0, 0, 0]))
    Column-major matrix containing the dipole moment.
      j_data : dict
    Data on the molecule, as deserialized from the scanlog format.
      file_name : str, optional
    Base name of the files in which to save the images.
      labels : list(str), optional
    Labels to append to `file_name` for each datum in `data`. If None, the position of the datum is appended.
      size : tuple(int, int), optional
    The size of the image to save.
    """

    figure = _init_scene(elements_3D_coords, atom_pairs, atoms_Z) 
    for dip_type in data :
        D=data[dip_type]
        Mu = mlab.quiver3d(D[1][0], D[1][1], D[1][2], D[0][0] - D[1][0], D[0][1] - D[1][1], D[0][2] - D[1][2] , figure=figure, mode='arrow', scale_factor=scale[dip_type], color=colors[dip_type])
        MuText = mlab.text3d(D[0][0]*scale[dip_type], D[0][1]*scale[dip_type], D[0][2]*scale[dip_type], dip_type, color=colors[dip_type], scale=0.1, figure=figure)
        
    if file_name is not None:
        if 'Singlet' in et_sym:
            figure = _set_cam(figure, "cam1")
            mlab.savefig("{}-DIP-S{}.png".format(file_name, labels[0]), figure=figure, size=size)
            figure = _set_cam(figure, "cam2")
            mlab.savefig("{}-DIP-S{}_cam2.png".format(file_name, labels[0]), figure=figure, size=size)
        elif 'Triplet' in et_sym:
            figure = _set_cam(figure, "cam1")
            mlab.savefig("{}-DIP-T{}.png".format(file_name, labels[0]), figure=figure, size=size)
            figure = _set_cam(figure, "cam2")
            mlab.savefig("{}-DIP-T{}_cam2.png".format(file_name, labels[0]), figure=figure, size=size)
        else:
            figure = _set_cam(figure, "cam1")
            mlab.savefig("{}-DIP-{}.png".format(file_name, labels[0]), figure=figure, size=size)
            figure = _set_cam(figure, "cam2")
            mlab.savefig("{}-DIP-{}_cam2.png".format(file_name, labels[0]), figure=figure, size=size)

    mlab.clf(figure=figure)
            
    mlab.close()
        
def viz_Potential(r_data, V_data, X, Y, Z, elements_3D_coords, atom_pairs, atoms_Z, file_name=None, size=(width,height)):
    u"""Visualizes the electrostatic potential difference of the molecule.

    ** Parameters **
      r_data, V_data : numpy.ndarray
    Voxels of the electron density and the potential difference of the molecule, respectively.
      X, Y, Z : numpy.ndarray
    Meshgrids as generated by numpy.mgrid, for positioning the voxels.
      j_data : dict
    Data on the molecule, as deserialized from the scanlog format.
      file_name : str, optional
    Base name of the file in which to save the image.
      size : tuple(int, int), optional
    The size of the image to save.

    ** Returns **
      figure : mayavi.core.scene.Scene
    The MayaVi scene containing the visualization.
    """
    
    # Skip if missing data
    if (
      r_data is None
      or V_data is None
      or X is None
      or Y is None
      or Z is None
      or elements_3D_coords is None
      or atom_pairs is None
      or atoms_Z is None
    ):
      return

    # For a neutral compound we should have in the potential some negative small values (delta- between 0 to -0.1) 
    # and bigger range in the postive values (delta+ between 0 to several hundreds)
    # To a better comparaison between compounds we will produce two figures : one with fixed limits and one with a relative scale to emphasize the density differences
    # Jet scale is centred around the green - cyan color. To keep the neutral charges around the yellowish to cyan, better to use asymetric limit values (1,125/1).
    # For delta_plus a value of  0.08 a.u. seems reasonable. The corresponding delta_minus is around 0.07 a.u. ! limits below have the oppposite sign due to jet colormap. 
    # Unfortunately mlab.colorbar(orientation='horizontal') does not work. TODO !
    
    #print("Median, Min and Max values in potential:", np.median(V_data), np.min(V_data), np.max(V_data))
    # With jet colormap, we need to show the opposite of V to have delta+ in blue and delta- in red
    V = -V_data

    turbo_lut = [[0.18995,0.07176,0.23217],[0.19483,0.08339,0.26149],[0.19956,0.09498,0.29024],[0.20415,0.10652,0.31844],[0.20860,0.11802,0.34607],[0.21291,0.12947,0.37314],[0.21708,0.14087,0.39964],[0.22111,0.15223,0.42558],[0.22500,0.16354,0.45096],[0.22875,0.17481,0.47578],[0.23236,0.18603,0.50004],[0.23582,0.19720,0.52373],[0.23915,0.20833,0.54686],[0.24234,0.21941,0.56942],[0.24539,0.23044,0.59142],[0.24830,0.24143,0.61286],[0.25107,0.25237,0.63374],[0.25369,0.26327,0.65406],[0.25618,0.27412,0.67381],[0.25853,0.28492,0.69300],[0.26074,0.29568,0.71162],[0.26280,0.30639,0.72968],[0.26473,0.31706,0.74718],[0.26652,0.32768,0.76412],[0.26816,0.33825,0.78050],[0.26967,0.34878,0.79631],[0.27103,0.35926,0.81156],[0.27226,0.36970,0.82624],[0.27334,0.38008,0.84037],[0.27429,0.39043,0.85393],[0.27509,0.40072,0.86692],[0.27576,0.41097,0.87936],[0.27628,0.42118,0.89123],[0.27667,0.43134,0.90254],[0.27691,0.44145,0.91328],[0.27701,0.45152,0.92347],[0.27698,0.46153,0.93309],[0.27680,0.47151,0.94214],[0.27648,0.48144,0.95064],[0.27603,0.49132,0.95857],[0.27543,0.50115,0.96594],[0.27469,0.51094,0.97275],[0.27381,0.52069,0.97899],[0.27273,0.53040,0.98461],[0.27106,0.54015,0.98930],[0.26878,0.54995,0.99303],[0.26592,0.55979,0.99583],[0.26252,0.56967,0.99773],[0.25862,0.57958,0.99876],[0.25425,0.58950,0.99896],[0.24946,0.59943,0.99835],[0.24427,0.60937,0.99697],[0.23874,0.61931,0.99485],[0.23288,0.62923,0.99202],[0.22676,0.63913,0.98851],[0.22039,0.64901,0.98436],[0.21382,0.65886,0.97959],[0.20708,0.66866,0.97423],[0.20021,0.67842,0.96833],[0.19326,0.68812,0.96190],[0.18625,0.69775,0.95498],[0.17923,0.70732,0.94761],[0.17223,0.71680,0.93981],[0.16529,0.72620,0.93161],[0.15844,0.73551,0.92305],[0.15173,0.74472,0.91416],[0.14519,0.75381,0.90496],[0.13886,0.76279,0.89550],[0.13278,0.77165,0.88580],[0.12698,0.78037,0.87590],[0.12151,0.78896,0.86581],[0.11639,0.79740,0.85559],[0.11167,0.80569,0.84525],[0.10738,0.81381,0.83484],[0.10357,0.82177,0.82437],[0.10026,0.82955,0.81389],[0.09750,0.83714,0.80342],[0.09532,0.84455,0.79299],[0.09377,0.85175,0.78264],[0.09287,0.85875,0.77240],[0.09267,0.86554,0.76230],[0.09320,0.87211,0.75237],[0.09451,0.87844,0.74265],[0.09662,0.88454,0.73316],[0.09958,0.89040,0.72393],[0.10342,0.89600,0.71500],[0.10815,0.90142,0.70599],[0.11374,0.90673,0.69651],[0.12014,0.91193,0.68660],[0.12733,0.91701,0.67627],[0.13526,0.92197,0.66556],[0.14391,0.92680,0.65448],[0.15323,0.93151,0.64308],[0.16319,0.93609,0.63137],[0.17377,0.94053,0.61938],[0.18491,0.94484,0.60713],[0.19659,0.94901,0.59466],[0.20877,0.95304,0.58199],[0.22142,0.95692,0.56914],[0.23449,0.96065,0.55614],[0.24797,0.96423,0.54303],[0.26180,0.96765,0.52981],[0.27597,0.97092,0.51653],[0.29042,0.97403,0.50321],[0.30513,0.97697,0.48987],[0.32006,0.97974,0.47654],[0.33517,0.98234,0.46325],[0.35043,0.98477,0.45002],[0.36581,0.98702,0.43688],[0.38127,0.98909,0.42386],[0.39678,0.99098,0.41098],[0.41229,0.99268,0.39826],[0.42778,0.99419,0.38575],[0.44321,0.99551,0.37345],[0.45854,0.99663,0.36140],[0.47375,0.99755,0.34963],[0.48879,0.99828,0.33816],[0.50362,0.99879,0.32701],[0.51822,0.99910,0.31622],[0.53255,0.99919,0.30581],[0.54658,0.99907,0.29581],[0.56026,0.99873,0.28623],[0.57357,0.99817,0.27712],[0.58646,0.99739,0.26849],[0.59891,0.99638,0.26038],[0.61088,0.99514,0.25280],[0.62233,0.99366,0.24579],[0.63323,0.99195,0.23937],[0.64362,0.98999,0.23356],[0.65394,0.98775,0.22835],[0.66428,0.98524,0.22370],[0.67462,0.98246,0.21960],[0.68494,0.97941,0.21602],[0.69525,0.97610,0.21294],[0.70553,0.97255,0.21032],[0.71577,0.96875,0.20815],[0.72596,0.96470,0.20640],[0.73610,0.96043,0.20504],[0.74617,0.95593,0.20406],[0.75617,0.95121,0.20343],[0.76608,0.94627,0.20311],[0.77591,0.94113,0.20310],[0.78563,0.93579,0.20336],[0.79524,0.93025,0.20386],[0.80473,0.92452,0.20459],[0.81410,0.91861,0.20552],[0.82333,0.91253,0.20663],[0.83241,0.90627,0.20788],[0.84133,0.89986,0.20926],[0.85010,0.89328,0.21074],[0.85868,0.88655,0.21230],[0.86709,0.87968,0.21391],[0.87530,0.87267,0.21555],[0.88331,0.86553,0.21719],[0.89112,0.85826,0.21880],[0.89870,0.85087,0.22038],[0.90605,0.84337,0.22188],[0.91317,0.83576,0.22328],[0.92004,0.82806,0.22456],[0.92666,0.82025,0.22570],[0.93301,0.81236,0.22667],[0.93909,0.80439,0.22744],[0.94489,0.79634,0.22800],[0.95039,0.78823,0.22831],[0.95560,0.78005,0.22836],[0.96049,0.77181,0.22811],[0.96507,0.76352,0.22754],[0.96931,0.75519,0.22663],[0.97323,0.74682,0.22536],[0.97679,0.73842,0.22369],[0.98000,0.73000,0.22161],[0.98289,0.72140,0.21918],[0.98549,0.71250,0.21650],[0.98781,0.70330,0.21358],[0.98986,0.69382,0.21043],[0.99163,0.68408,0.20706],[0.99314,0.67408,0.20348],[0.99438,0.66386,0.19971],[0.99535,0.65341,0.19577],[0.99607,0.64277,0.19165],[0.99654,0.63193,0.18738],[0.99675,0.62093,0.18297],[0.99672,0.60977,0.17842],[0.99644,0.59846,0.17376],[0.99593,0.58703,0.16899],[0.99517,0.57549,0.16412],[0.99419,0.56386,0.15918],[0.99297,0.55214,0.15417],[0.99153,0.54036,0.14910],[0.98987,0.52854,0.14398],[0.98799,0.51667,0.13883],[0.98590,0.50479,0.13367],[0.98360,0.49291,0.12849],[0.98108,0.48104,0.12332],[0.97837,0.46920,0.11817],[0.97545,0.45740,0.11305],[0.97234,0.44565,0.10797],[0.96904,0.43399,0.10294],[0.96555,0.42241,0.09798],[0.96187,0.41093,0.09310],[0.95801,0.39958,0.08831],[0.95398,0.38836,0.08362],[0.94977,0.37729,0.07905],[0.94538,0.36638,0.07461],[0.94084,0.35566,0.07031],[0.93612,0.34513,0.06616],[0.93125,0.33482,0.06218],[0.92623,0.32473,0.05837],[0.92105,0.31489,0.05475],[0.91572,0.30530,0.05134],[0.91024,0.29599,0.04814],[0.90463,0.28696,0.04516],[0.89888,0.27824,0.04243],[0.89298,0.26981,0.03993],[0.88691,0.26152,0.03753],[0.88066,0.25334,0.03521],[0.87422,0.24526,0.03297],[0.86760,0.23730,0.03082],[0.86079,0.22945,0.02875],[0.85380,0.22170,0.02677],[0.84662,0.21407,0.02487],[0.83926,0.20654,0.02305],[0.83172,0.19912,0.02131],[0.82399,0.19182,0.01966],[0.81608,0.18462,0.01809],[0.80799,0.17753,0.01660],[0.79971,0.17055,0.01520],[0.79125,0.16368,0.01387],[0.78260,0.15693,0.01264],[0.77377,0.15028,0.01148],[0.76476,0.14374,0.01041],[0.75556,0.13731,0.00942],[0.74617,0.13098,0.00851],[0.73661,0.12477,0.00769],[0.72686,0.11867,0.00695],[0.71692,0.11268,0.00629],[0.70680,0.10680,0.00571],[0.69650,0.10102,0.00522],[0.68602,0.09536,0.00481],[0.67535,0.08980,0.00449],[0.66449,0.08436,0.00424],[0.65345,0.07902,0.00408],[0.64223,0.07380,0.00401],[0.63082,0.06868,0.00401],[0.61923,0.06367,0.00410],[0.60746,0.05878,0.00427],[0.59550,0.05399,0.00453],[0.58336,0.04931,0.00486],[0.57103,0.04474,0.00529],[0.55852,0.04028,0.00579],[0.54583,0.03593,0.00638],[0.53295,0.03169,0.00705],[0.51989,0.02756,0.00780],[0.50664,0.02354,0.00863],[0.49321,0.01963,0.00955],[0.47960,0.01583,0.01055]]
    
    turbo_with_alpha = np.zeros((256, 4))
    turbo_with_alpha[:, :3] = turbo_lut
    turbo_with_alpha[:, -1] = 0.85 # hard coded opacity for potential

    # Case with relative scale
    figure = _init_scene(elements_3D_coords, atom_pairs, atoms_Z) 
    src = mlab.pipeline.scalar_field(X, Y, Z, r_data, figure=figure)
    ## Add potential as additional array
    src.image_data.point_data.add_array(V.T.ravel()) #/units.V_to_Kcal_mol)
    ## Name it
    src.image_data.point_data.get_array(1).name = "potential"
    ## Update object
    src.update()
    ## Select scalar attribute
    srcp = mlab.pipeline.set_active_attribute(src, figure=figure, point_scalars="scalar")
    ## Plot it
    cont = mlab.pipeline.contour(srcp, figure=figure)
    cont.filter.contours=[0.002]
    ## Select potential
    cont_V = mlab.pipeline.set_active_attribute(cont, figure=figure, point_scalars="potential")
    surf = mlab.pipeline.surface(cont_V, figure=figure, opacity=0.85)
    vtk_data = surf.mlab_source.dataset
    scalar_data = vtk_data.point_data.scalars
    min_val = np.min(scalar_data)
    max_val = np.max(scalar_data)
    print("Potential Minimum value:", min_val)
    print("Potential Maximum value:", max_val)
    surf.module_manager.scalar_lut_manager.lut.table = turbo_lut
    if file_name is not None:
        figure = _set_cam(figure, "cam1")
        mlab.savefig("{}-MEP.png".format(file_name), figure=figure, size=size)
        figure = _set_cam(figure, "cam2")
        mlab.savefig("{}-MEP_cam2.png".format(file_name), figure=figure, size=size)

    # Case with a fixed scale
    figure_fixed = _init_scene(elements_3D_coords, atom_pairs, atoms_Z) 
    src_fixed = mlab.pipeline.scalar_field(X, Y, Z, r_data, figure=figure_fixed)
    ## Add potential as additional array
    src_fixed.image_data.point_data.add_array(V.T.ravel()) #/units.V_to_Kcal_mol)
    ## Name it
    src_fixed.image_data.point_data.get_array(1).name = "potential"
    ## Update object
    src_fixed.update()
    ## Select scalar attribute
    srcp_fixed = mlab.pipeline.set_active_attribute(src_fixed, figure=figure_fixed, point_scalars="scalar")
    ## Plot it
    cont_fixed = mlab.pipeline.contour(srcp_fixed, figure=figure_fixed)
    cont_fixed.filter.contours=[0.002]
    ## Select potential
    cont_V_fixed = mlab.pipeline.set_active_attribute(cont_fixed, figure=figure_fixed, point_scalars="potential")
    # Jet color map is not really centred on the green. There is a bit too much blue. So vmin is set lower (jet blue = negative values)
    surf = mlab.pipeline.surface(cont_V_fixed, figure=figure_fixed, opacity=0.85, vmin=-0.01, vmax=+0.01)
    surf.module_manager.scalar_lut_manager.lut.table = turbo_lut
    if file_name is not None:
        figure_fixed = _set_cam(figure_fixed, "cam1")
        mlab.savefig("{}-MEP_fixed.png".format(file_name), figure=figure_fixed, size=size)
        figure_fixed = _set_cam(figure_fixed, "cam2")
        mlab.savefig("{}-MEP_fixed_cam2.png".format(file_name), figure=figure_fixed, size=size)

def viz_Fukui(data, X, Y, Z, elements_3D_coords, atom_pairs, atoms_Z, file_name=None, labels=None, size=(width,height)):
    u"""Visualizes the fukui density differences for the molecule.

    ** Parameters **
      data : list(numpy.ndarray)
    Voxels containing the scalar values of the electron density difference to plot.
      X, Y, Z : numpy.ndarray
    Meshgrids as generated by numpy.mgrid, for positioning the voxels.
      j_data : dict
    Data on the optimized state of the molecule, as deserialized from the scanlog format.
      file_name : str, optional
    Base name of the files in which to save the images.
      label : text
    Label to append to `file_name` for each series in `data`. "plus" for sp_plus - opt; "minus" for opt - sp_minus.
      size : tuple(int, int), optional
    The size of the image to save.

    ** Returns **
      figure : mayavi.core.scene.Scene
    The MayaVi scene containing the visualization.
    """

    figure = _init_scene(elements_3D_coords, atom_pairs, atoms_Z) 
    Cutoffp, Cutoffn = CalcCutOff(data,IsoContourPercent=30) 
    F_data = mlab.pipeline.scalar_field(X, Y, Z, data, figure=figure)
    Fp = mlab.pipeline.iso_surface(F_data, figure=figure, contours=[ Cutoffp ], color=(0.0, 0.5, 0.5), opacity=surf_opacity)
    Fn = mlab.pipeline.iso_surface(F_data, figure=figure, contours=[ Cutoffn ], color=(0.95, 0.95, 0.95), opacity=surf_opacity)
    
    if file_name is not None:
        figure = _set_cam(figure, "cam1")
        mlab.savefig("{}-fukui-{}.png".format(file_name, labels), figure=figure, size=size)
        figure = _set_cam(figure, "cam2")
        mlab.savefig("{}-fukui-{}_cam2.png".format(file_name, labels), figure=figure, size=size)
    Fp.remove()
    Fn.remove()


def viz_Fdual(data, X, Y, Z, elements_3D_coords, atom_pairs, atoms_Z, file_name=None, size=(width,height)):
    u"""Visualizes the fukui density differences for the molecule.

    ** Parameters **
      data : list(numpy.ndarray)
    Voxels containing the scalar values of the electron density difference to plot.
      X, Y, Z : numpy.ndarray
    Meshgrids as generated by numpy.mgrid, for positioning the voxels.
      j_data : dict
    Data on the optimized state of the molecule, as deserialized from the scanlog format.
      file_name : str, optional
    Base name of the files in which to save the images.
      size : tuple(int, int), optional
    The size of the image to save.

    ** Returns **
      figure : mayavi.core.scene.Scene
    The MayaVi scene containing the visualization.
    """
    figure = _init_scene(elements_3D_coords, atom_pairs, atoms_Z) 
    Cutoffp, Cutoffn = CalcCutOff(data,IsoContourPercent=30) 

    F_data = mlab.pipeline.scalar_field(X, Y, Z, data, figure=figure)
    #print("positive Cutoff :", Cutoffp)
    Fp = mlab.pipeline.iso_surface(F_data, figure=figure, contours=[ Cutoffp ], color=(0.0, 0.5, 0.5), opacity=surf_opacity)
    Fn = mlab.pipeline.iso_surface(F_data, figure=figure, contours=[ Cutoffn ], color=(0.95, 0.95, 0.95), opacity=surf_opacity)

    if file_name is not None:
        figure = _set_cam(figure, "cam1")
        mlab.savefig("{}-Fdual.png".format(file_name), figure=figure, size=size)
        figure = _set_cam(figure, "cam2")
        mlab.savefig("{}-Fdual_cam2.png".format(file_name), figure=figure, size=size)
        # np.save("{}-Fdual.npy".format(file_name), data)
    Fp.remove()
    Fn.remove()

