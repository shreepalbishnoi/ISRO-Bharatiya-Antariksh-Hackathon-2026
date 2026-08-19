"""
data_acquisition.py
--------------------
Downloads TESS light curves from MAST using `lightkurve`.

Two modes are supported, matching the PS requirements:
  1. Single-target retrieval (by TIC ID / name) - for testing & known objects.
  2. Bulk sector retrieval - downloads (or lists) all targets observed in a
     given TESS sector at a given cadence, as suggested in the PS
     ("download a sector's high-cadence data ... ~20-30k light curves").

NOTE: Bulk-downloading an entire sector (20-30k targets) is multiple
terabytes via individual FITS files. In practice we recommend:
  - Using MAST's bulk-download curl scripts / TESS-CTL target lists, or
  - Using the TESS-SPOC / QLP HLSP bulk light-curve archives, or
  - Sub-sampling N targets per sector for prototyping (done here).

This module is written to run wherever MAST is reachable (it will not run
inside network-restricted sandboxes - tested here against synthetic data,
see demo_synthetic.py).
"""

import logging
from typing import List, Optional

import lightkurve as lk
import numpy as np
import pandas as pd

from config import DEFAULT_AUTHOR, DEFAULT_MISSION, DEFAULT_FLUX_COLUMN

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def search_target(tic_id: str, author: str = DEFAULT_AUTHOR) -> lk.SearchResult:
    """Search MAST for all available light curve products of a single TIC target."""
    query = f"TIC {tic_id}" if not str(tic_id).upper().startswith("TIC") else tic_id
    result = lk.search_lightcurve(query, mission=DEFAULT_MISSION, author=author)
    log.info("Found %d light-curve products for %s", len(result), query)
    return result


def download_target(tic_id: str, author: str = DEFAULT_AUTHOR,
                     sector: Optional[int] = None) -> lk.LightCurveCollection:
    """
    Download all (or one sector's) light curves for a single TIC target.
    Returns a LightCurveCollection; use .stitch() to merge multi-sector data.
    """
    result = search_target(tic_id, author=author)
    if sector is not None:
        result = result[result.table["sequence_number"] == sector]
    if len(result) == 0:
        raise ValueError(f"No light curves found for TIC {tic_id} (sector={sector}).")
    return result.download_all()


def list_sector_targets(sector: int, author: str = DEFAULT_AUTHOR,
                         cadence: str = "short", max_targets: Optional[int] = None
                         ) -> pd.DataFrame:
    """
    Query MAST for the catalog of targets observed in a given TESS sector.
    Returns a DataFrame with TIC IDs and basic metadata (no FITS download yet).
    This is the entry point for the "download a sector's high-cadence data"
    requirement -- list_sector_targets() enumerates targets, then
    bulk_download() pulls the actual light curves (optionally a random
    subsample for tractable prototyping).
    """
    exptime = "short" if cadence == "short" else "long"
    result = lk.search_lightcurve(f"sector={sector}", mission=DEFAULT_MISSION,
                                   author=author, exptime=exptime)
    table = result.table.to_pandas()
    if max_targets is not None and len(table) > max_targets:
        table = table.sample(n=max_targets, random_state=42).reset_index(drop=True)
    log.info("Sector %d: %d targets listed (cadence=%s)", sector, len(table), cadence)
    return table


def bulk_download(tic_ids: List[str], author: str = DEFAULT_AUTHOR,
                   flux_column: str = DEFAULT_FLUX_COLUMN
                   ) -> List[lk.LightCurve]:
    """
    Download and lightly clean a list of TIC IDs. Skips targets that fail
    (e.g. missing PDCSAP flux, transient MAST errors) and logs the failure.
    """
    light_curves = []
    for tic in tic_ids:
        try:
            coll = download_target(tic, author=author)
            lc = coll.stitch().remove_nans(column=flux_column)
            lc.meta["TIC_ID"] = tic
            light_curves.append(lc)
        except Exception as exc:  # noqa: BLE001 - log & continue, bulk job
            log.warning("Failed to download TIC %s: %s", tic, exc)
    log.info("Successfully downloaded %d / %d targets", len(light_curves), len(tic_ids))
    return light_curves


if __name__ == "__main__":
    # Example usage (requires internet access to MAST):
    # lc_collection = download_target("261136679")  # TOI-700, known multi-planet system
    # lc = lc_collection.stitch()
    # print(lc)
    print("Run this module from an environment with access to archive.stsci.edu / MAST.")
