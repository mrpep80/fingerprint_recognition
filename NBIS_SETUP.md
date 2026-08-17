# NBIS setup

The v5 fusion uses two independent NBIS programs when available:

- `mindtct` — extracts minutiae into `.xyt` templates.
- `bozorth3` — matches two `.xyt` templates and returns the native BOZORTH3 score.

NBIS is optional. If either executable is unavailable, `--method fusion` automatically disables NBIS and continues with the OpenCV engines.

## Check installation

```bash
which mindtct
which bozorth3
```

If they are not in `PATH`, set:

```bash
export NBIS_MINDTCT=/absolute/path/to/mindtct
export NBIS_BOZORTH3=/absolute/path/to/bozorth3
```

## Notes

The adapter invokes `mindtct -b -m1` and `bozorth3 -m1`, so both sides use the same M1 minutiae representation. For TIFF/PNG/BMP gallery images, the adapter creates a temporary high-quality baseline JPEG before calling MINDTCT, because that is a documented NBIS input format.

NBIS itself is maintained/distributed by NIST; see the official NIST NBIS page and the NBIS User Guide for installation/build instructions.
