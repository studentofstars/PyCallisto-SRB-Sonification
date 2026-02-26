#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
callisto_pipeline.py
---------------------
Generalised processing pipeline for e-CALLISTO solar radio spectrogram
FITS files using the pyCallisto library.

Provides a clean, reusable workflow for loading, joining, processing and
plotting observations from any e-CALLISTO station, supporting an arbitrary
number of input files and fully configurable time/frequency slicing.

Features
--------
  - Load one or more .fit / .fit.gz FITS files
  - Join N files along the time axis via join_fits()
  - Background subtraction (median-column method)
  - Time-axis slicing  – variable HH:MM:SS window
  - Frequency-axis slicing – variable MHz band
  - Spectrogram, light-curve, mean-spectrum and universal plots
  - save_spectrogram() helper for PNG output

Usage (standalone)
------------------
    python callisto_pipeline.py

Usage (as a module)
-------------------
    from callisto_pipeline import run_pipeline
    obs = run_pipeline(
        fits_paths=["file1.fit.gz", "file2.fit.gz"],
        time_slice=("11:10:00", "11:40:00"),
        freq_slice=(20, 80),
    )

Repository
----------
  https://github.com/studentofstars/PyCallisto-SRB-Sonification

Install
-------
  pip install pyCallisto
  pip install -r requirements.txt      # full environment

References
----------
  Pawase, R. & Raja, K. S. (2020). pyCallisto: A Python Library To Process
  The CALLISTO Spectrometer Data. arXiv:2006.16300 [astro-ph.IM]
  https://arxiv.org/abs/2006.16300

  pyCallisto on GitHub : https://github.com/ravipawase/pyCallisto
  pyCallisto on PyPI   : https://pypi.org/project/pyCallisto/
  e-CALLISTO network   : http://www.e-callisto.org/

Contributors
------------
  Mr. Ravindra Pawase
      Data Scientist, Cummins Inc., Pune 411 004, India
      ravi.pawase@gmail.com

  Dr. K. Sasikumar Raja
      Assistant Professor, Indian Institute of Astrophysics,
      II Block, Koramangala, Bengaluru 560 034, India
      sasikumarraja@gmail.com

  Mr. Mrutyunjaya Muduli
      B.E. Computer Science and Engineering,
      HKBK College of Engineering, Nagavara, Bengaluru 560 045, India
      mudulimrutyunjaya42@gmail.com

License
-------
  MIT — see LICENSE
"""

import matplotlib.pyplot as plt
import pyCallisto as pyc


# ─────────────────────────────────────────────────────────────
# CONFIGURATION  –  edit this section to suit your event
# ─────────────────────────────────────────────────────────────

# List of FITS files in chronological order (add as many as needed)
FITS_PATHS = [
    "GREENLAND_20170906_111014_62.fit.gz",
    "GREENLAND_20170906_112514_62.fit.gz",
    "GREENLAND_20170906_114014_62.fit.gz",
    "GREENLAND_20170906_115514_62.fit.gz",
]

# Time slice – set to None to skip slicing
# Format: "HH:MM:SS"  (must lie within the joined observation window)
TIME_SLICE = ("11:10:00", "11:30:00")   # or None

# Frequency slice – set to None to skip slicing
# Values in MHz
FREQ_SLICE = (20, 80)                   # or None

# Spectrogram display parameters
XTICK        = 5          # x-axis tick interval in minutes
BLEVEL       = 0          # background/colour floor level
VMAX         = 40         # colour ceiling
FIG_SIZE     = (10, 5)    # (width, height) in inches
SUBTRACT_BG  = True       # apply background subtraction before slicing


# ─────────────────────────────────────────────────────────────
# HELPER: join a list of FITS paths / pyCallisto objects
# ─────────────────────────────────────────────────────────────

def join_fits(fits_paths):
    """
    Load and concatenate N FITS files along the time axis.

    Parameters
    ----------
    fits_paths : list of str
        Chronologically ordered list of FITS file paths.

    Returns
    -------
    pyCallisto object
        A single pyCallisto object spanning all input files.

    Raises
    ------
    ValueError
        If fewer than one path is provided.
    """
    if len(fits_paths) < 1:
        raise ValueError("Provide at least one FITS file path.")

    print(f"Loading {fits_paths[0]} ...")
    joined = pyc.pyCallisto.fromFile(fits_paths[0])

    for path in fits_paths[1:]:
        print(f"Appending {path} ...")
        joined = joined.appendTimeAxis(path)

    print(f"Joined observation spans "
          f"{joined.imageHeader['TIME-OBS']} – "
          f"{joined.imageHeader['TIME-END']} "
          f"on {joined.imageHeader['DATE-OBS']}")
    return joined


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline(
    fits_paths,
    subtract_bg=SUBTRACT_BG,
    time_slice=TIME_SLICE,
    freq_slice=FREQ_SLICE,
    xtick=XTICK,
    blevel=BLEVEL,
    vmax=VMAX,
    fig_size=FIG_SIZE,
    show_intermediate=True,
):
    """
    Full processing pipeline for e-Callisto observations.

    Parameters
    ----------
    fits_paths : list of str
        Chronologically ordered list of .fit / .fit.gz file paths.
    subtract_bg : bool
        Subtract median background before slicing (default True).
    time_slice : tuple of str or None
        ("HH:MM:SS", "HH:MM:SS") window to keep, or None.
    freq_slice : tuple of int/float or None
        (freq_min_MHz, freq_max_MHz) band to keep, or None.
    xtick : int
        X-axis (time) tick interval in minutes.
    blevel : int or float
        Colour floor / background level for imshow.
    vmax : int or float
        Colour ceiling for imshow.
    fig_size : tuple
        Figure size (width, height) in inches.
    show_intermediate : bool
        If True, plot the raw joined spectrogram before processing.

    Returns
    -------
    pyCallisto
        The final processed pyCallisto object.
    """

    # ── Step 1: load & join ──────────────────────────────────
    obs = join_fits(fits_paths)

    if show_intermediate and len(fits_paths) > 1:
        print("\n[Plot] Raw joined spectrogram")
        plt_obj = obs.spectrogram(xtick=xtick, blevel=blevel, vmax=vmax,
                                  figSize=fig_size)
        plt_obj.title("Joined – raw")
        plt_obj.tight_layout()
        plt_obj.show()

    # ── Step 2: background subtraction ──────────────────────
    if subtract_bg:
        print("Subtracting background ...")
        obs = obs.subtractBackground()

        if show_intermediate:
            print("[Plot] After background subtraction")
            plt_obj = obs.spectrogram(xtick=xtick, blevel=blevel, vmax=vmax,
                                      figSize=fig_size)
            plt_obj.title("Background subtracted")
            plt_obj.tight_layout()
            plt_obj.show()

    # ── Step 3: time-axis slice ──────────────────────────────
    if time_slice is not None:
        t1, t2 = time_slice
        print(f"Slicing time axis: {t1} → {t2}")
        obs = obs.sliceTimeAxis(t1, t2)

    # ── Step 4: frequency-axis slice ────────────────────────
    if freq_slice is not None:
        f1, f2 = freq_slice
        print(f"Slicing frequency axis: {f1} – {f2} MHz")
        obs = obs.sliceFrequencyAxis(f1, f2)

    # ── Step 5: final spectrogram ────────────────────────────
    print("\n[Plot] Final processed spectrogram")
    plt_obj = obs.spectrogram(xtick=xtick, blevel=blevel, vmax=vmax,
                              figSize=fig_size)
    plt_obj.title("Processed spectrogram")
    plt_obj.tight_layout()
    plt_obj.show()

    return obs


# ─────────────────────────────────────────────────────────────
# ADDITIONAL ANALYSIS HELPERS
# ─────────────────────────────────────────────────────────────

def plot_light_curve(obs, fig_size=(10, 4)):
    """
    Plot the mean light curve (flux vs time) from a pyCallisto object.

    Parameters
    ----------
    obs : pyCallisto
        A loaded / processed pyCallisto object.
    fig_size : tuple
        Figure size (width, height) in inches.
    """
    obs.meanLightCurve(plot=True, figSize=fig_size)
    plt.tight_layout()
    plt.show()


def plot_mean_spectrum(obs, fig_size=(6, 6)):
    """
    Plot the mean spectrum (flux vs frequency) from a pyCallisto object.

    Parameters
    ----------
    obs : pyCallisto
        A loaded / processed pyCallisto object.
    fig_size : tuple
        Figure size (width, height) in inches.
    """
    obs.meanSpectrum(plot=True)
    plt.tight_layout()
    plt.show()


def plot_universal(obs, title="Solar Radio Burst", fig_size=(12, 8)):
    """
    Produce the universal plot: spectrogram + light curve + mean spectrum.

    Parameters
    ----------
    obs : pyCallisto
        A loaded / processed pyCallisto object.
    title : str
        Title printed above the combined figure.
    fig_size : tuple
        Figure size (width, height) in inches.
    """
    obs.universalPlot(title=title, figSize=fig_size)
    plt.show()


def save_spectrogram(obs, outfile="spectrogram.png",
                     xtick=XTICK, blevel=BLEVEL, vmax=VMAX,
                     fig_size=FIG_SIZE):
    """
    Save the final spectrogram to a PNG file without displaying it.

    Parameters
    ----------
    obs : pyCallisto
        A loaded / processed pyCallisto object.
    outfile : str
        Path / filename for the saved PNG.
    xtick, blevel, vmax, fig_size : see run_pipeline()
    """
    plt_obj = obs.spectrogram(xtick=xtick, blevel=blevel, vmax=vmax,
                              figSize=fig_size)
    plt_obj.tight_layout()
    plt_obj.savefig(outfile, dpi=150)
    plt_obj.close()
    print(f"Spectrogram saved → {outfile}")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run the full pipeline with the parameters defined at the top
    processed = run_pipeline(
        fits_paths=FITS_PATHS,
        subtract_bg=SUBTRACT_BG,
        time_slice=TIME_SLICE,
        freq_slice=FREQ_SLICE,
        xtick=XTICK,
        blevel=BLEVEL,
        vmax=VMAX,
        fig_size=FIG_SIZE,
        show_intermediate=True,
    )

    # Uncomment any of the lines below for additional analysis:

    # plot_light_curve(processed)
    # plot_mean_spectrum(processed)
    # plot_universal(processed, title="20170906 Type III Burst")
    # save_spectrogram(processed, outfile="plots/20170906_processed.png")
