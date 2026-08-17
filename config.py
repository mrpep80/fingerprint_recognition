"""
config.py – v2.1
=================
Unico punto in cui vivono tutti i parametri del sistema.
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:

    # ── Ridimensionamento automatico ─────────────────────────────────
    max_working_dim: int = 1500

    # ── Feature extraction ────────────────────────────────────────────
    max_features: int = 3000

    # SIFT
    sift_contrast_threshold: float = 0.02
    sift_edge_threshold:     int   = 12
    sift_sigma:              float = 1.6

    # ORB
    orb_scale_factor:   float = 1.15
    orb_n_levels:       int   = 12
    orb_edge_threshold: int   = 15

    # AKAZE
    akaze_threshold: float = 0.003

    # ── Matching ─────────────────────────────────────────────────────
    sift_ratio_threshold: float = 0.75
    bin_ratio_threshold:  float = 0.80
    flann_trees:          int   = 5
    flann_checks:         int   = 100

    # ── RANSAC ───────────────────────────────────────────────────────
    ransac_reprojection_error: float = 6.0
    min_inliers:               int   = 4

    # ── Gabor ────────────────────────────────────────────────────────
    gabor_n_orientations: int   = 8
    gabor_kernel_size:    int   = 31
    gabor_sigma:          float = 4.5
    gabor_lambda:         float = 8.0
    gabor_gamma:          float = 0.5

    # ── ROI segmentation ─────────────────────────────────────────────
    roi_min_area_ratio: float = 0.003
    roi_max_area_ratio: float = 0.20
    roi_min_side:       int   = 60
    roi_padding:        int   = 15

    # ── Minutiae extraction ──────────────────────────────────────────
    minutiae_border_margin: int = 15
    minutiae_min_distance:  int = 10
    minutiae_max_dim:       int = 900   # resize interno prima dello skeletonize
    #
    # CHIAVE DEL FIX: un dito reale ha 30-80 minuzie.
    # Tenerne 700+ significa estrarre soprattutto rumore, causando
    # falsi positivi al 100% su qualsiasi coppia di impronte.
    # Cap a 120 + filtro per qualità locale.
    minutiae_max_count:     int = 120
    minutiae_quality_radius: int = 8    # raggio px per calcolo qualità locale

    # ── Minutiae matching ─────────────────────────────────────────────
    # Soglie più strette rispetto alla v2.0 per ridurre i falsi positivi
    minutiae_dist_tolerance: float = 12.0  # era 22.0 → molto più selettivo
    minutiae_nn_loose_ratio: float = 2.5   # era 5.0  → soglia NN = 12*2.5 = 30px
    minutiae_angle_tolerance: float = 40.0 # NUOVO: filtro orientazione (±40°)

    # ── Scoring ──────────────────────────────────────────────────────
    score_coverage_weight:  float = 0.50
    score_precision_weight: float = 0.30
    score_bonus_weight:     float = 0.20
    score_bonus_ref:        float = 80.0   # era 150 → scala più bassa (max ~120 minuzie)


    # ── Enhanced SIFT resolution ──────────────────────────────────────
    # Risoluzione massima per enhanced_sift PRIMA dell'applicazione Hong-Jain.
    # 800px è il punto ottimale: 24x più veloce di 1500px, qualità identica.
    # (fingerprint_enhancer scala in O(n_pixels) → impatto enorme su immagini grandi)
    enhanced_sift_max_dim: int = 800

    # ── DFT Enhancement (Bhowmik et al., IJSTR 2012) ─────────────────
    # Formula: i_enh = F^-1[F(u,v) × |F(u,v)|^k]
    # k=0.3 → ottimale per cross-domain (latente↔inchiostro)
    # k=0.0 → disabilitato
    dft_k: float = 0.3

    # ── Hong-Jain Fingerprint Enhancement (fingerprint-enhancer) ─────
    # Parametri per orientation-adaptive Gabor filtering
    hong_block_size:    int   = 16     # blocco per segmentazione creste
    hong_seg_thresh:    float = 0.1    # soglia varianza per segmentazione
    hong_grad_sigma:    int   = 1      # sigma gradiente
    hong_block_sigma:   int   = 7      # sigma blocco orientazione
    hong_orient_sigma:  int   = 7      # sigma smoothing orientazione
    hong_freq_blksize:  int   = 38     # blocco stima frequenza
    hong_freq_windsize: int   = 5      # finestra stima frequenza
    hong_min_wavelength: int  = 5      # lunghezza d'onda minima cresta
    hong_max_wavelength: int  = 15     # lunghezza d'onda massima cresta
    hong_angle_inc:     float = 3.0    # incremento angolare Gabor
    hong_filter_thresh: int   = -3     # soglia risposta filtro Gabor

    # ── I/O ───────────────────────────────────────────────────────────
    supported_extensions: Tuple[str, ...] = (
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
        '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.TIF',
    )
