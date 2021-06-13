from braintrips.config import balsa_dir, data_dir
import numpy as np
import nibabel as nib
import pandas as pd


def _load_nifti(file):
    return np.array(nib.load(file).dataobj).squeeze()


def load_sc():
    """
    Load structural connectivity matrix.

    Returns
    -------
    (180,180) np.ndnarray
        SC between left-hemispheric cortical parcels

    """
    return _load_nifti(balsa_dir.joinpath("SC.pconn.nii"))


def load_placebo_fc():
    """
    Load functional connectivity matrix.

    Returns
    -------
    (180,180) np.ndnarray
        FC between left-hemispheric cortical parcels

    """
    return _load_nifti(balsa_dir.joinpath("FC-Placebo.pconn.nii"))


def load_group_dgbc():
    """
    Load group-averaged delta GBC map.

    Returns
    -------
    (180,) np.ndarray
        vector of delta GBC values per left-hemispheric cortical parcel

    """
    return _load_nifti(
        balsa_dir.joinpath("empirical_group_delta_gbc.pscalar.nii"))


def load_subject_dgbc():
    """
    Load subject-level delta GBC maps.

    Returns
    -------
    (24, 180) np.ndarray

    """
    file = balsa_dir.joinpath("subject_delta_gbc.csv")
    return pd.read_csv(file, index_col=0)


def load_gene(gene="HTR2A"):
    file = balsa_dir.joinpath(gene+".pscalar.nii")
    if not file.exists():
        raise FileNotFoundError(f"{file} not found")
    return _load_nifti(file)


def load_distmat():
    return np.load(data_dir.joinpath("human_geodesic_distance_matrix.npy"))
