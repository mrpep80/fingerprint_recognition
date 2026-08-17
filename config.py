"""
config.py – v2.3
=================
Parametri centralizzati del sistema di riconoscimento.
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    # Ridimensionamento
    max_working_dim: int = 1500

    # Feature extraction
    max_features: int = 3000
    sift_contrast_threshold: float = 0.02
    sift_edge_threshold: int = 12
    sift_sigma: float = 1.6
    orb_scale_factor: float = 1.15
    orb_n_levels: int = 12
    orb_edge_threshold: int = 15
    akaze_threshold: float = 0.003

    # Matching
    sift_ratio_threshold: float = 0.75
    bin_ratio_threshold: float = 0.80
    flann_trees: int = 5
    flann_checks: int = 100

    # RANSAC
    ransac_reprojection_error: float = 5.0
    min_inliers: int = 5

    # Gabor
    gabor_n_orientations: int = 8
    gabor_kernel_size: int = 31
    gabor_sigma: float = 4.5
    gabor_lambda: float = 8.0
    gabor_gamma: float = 0.5

    # ROI segmentation
    roi_min_area_ratio: float = 0.003
    roi_max_area_ratio: float = 0.20
    roi_min_side: int = 60
    roi_padding: int = 15

    # Minutiae extraction
    minutiae_border_margin: int = 15
    minutiae_min_distance: int = 10
    minutiae_max_dim: int = 900
    minutiae_max_count: int = 120
    minutiae_quality_radius: int = 8

    # Minutiae matching
    minutiae_dist_tolerance: float = 10.0
    minutiae_nn_loose_ratio: float = 2.0
    minutiae_angle_tolerance: float = 30.0

    # Scoring
    score_coverage_weight: float = 0.50
    score_precision_weight: float = 0.30
    score_bonus_weight: float = 0.20
    score_bonus_ref: float = 80.0

    # Enhanced SIFT
    enhanced_sift_max_dim: int = 800
    dft_k: float = 0.3
    hong_block_size: int = 16
    hong_seg_thresh: float = 0.1
    hong_grad_sigma: int = 1
    hong_block_sigma: int = 7
    hong_orient_sigma: int = 7
    hong_freq_blksize: int = 38
    hong_freq_windsize: int = 5
    hong_min_wavelength: int = 5
    hong_max_wavelength: int = 15
    hong_angle_inc: float = 3.0
    hong_filter_thresh: int = -3

    # Robust scoring / geometry
    descriptor_score_target: int = 60
    minutiae_score_target: int = 35
    coverage_scale: float = 120.0
    geometry_area_reference: float = 25000.0
    geometry_inlier_target: int = 35
    homography_min_scale: float = 0.35
    homography_max_scale: float = 2.80
    homography_max_perspective: float = 0.0025

    # I/O
    supported_extensions: Tuple[str, ...] = (
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
        '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.TIF',
    )
