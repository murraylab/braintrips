# from os import path
# import numpy as np
# import nibabel as nib


# --------
# Data I/O
# --------


# def load_gifti(of):
#     return of.darrays[0].data
#
#
# def load_nifti(of):
#     return np.array(of.get_data()).squeeze()
#
#
# def load(f):
#     """
#     Load data from a NIFTI/GIFTI file.
#
#     Parameters
#     ----------
#     f : filename (str)
#         must have nii or gii extension
#
#     Returns
#     -------
#     np.ndarray
#
#     """
#     of = nib.load(f)
#     try:
#         return load_nifti(of)
#     except AttributeError:
#         return load_gifti(of)


def load_sc():
    """
    Load structural connectivity matrix.

    Returns
    -------
    (180,180) np.ndnarray
        SC between left-hemispheric cortical parcels
    """
    pass


def load_fc():
    """
    Load functional connectivity matrix.

    Returns
    -------
    (180,180) np.ndnarray
        FC between left-hemispheric cortical parcels

    """
    pass


def load_group_dgbc(gsr=True):
    """
    Load group-averaged delta GBC map.

    Parameters
    ----------
    gsr : bool, optional (default True)
        load map with/without GSR performed

    Returns
    -------
    (180,) np.ndarray
        vector of delta GBC values per left-hemispheric cortical parcel

    """
    if gsr:
        pass
    pass


def load_subject_dgbc():
    """
    Load subject-level delta GBC maps.

    Returns
    -------
    (24, 180) np.ndarray

    """
    pass
