# geometry
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional, Tuple, Union
import warnings
import pandas as pd
import numpy as np

try:
    import geopandas as gpd
    from shapely.geometry import LineString, MultiLineString
except ImportError as e:
    raise ImportError("Requires geopandas and shapely") from e


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Issue:
    severity: Severity
    code: str
    message: str
    count: Optional[int] = None
    sample_index: Optional[Union[int, List[int]]] = None


def _is_lines_like(geom) -> bool:
    return isinstance(geom, (LineString, MultiLineString))


def validate_lines_gdf(
    gdf: "gpd.GeoDataFrame",
    *,
    require_crs: Optional[str] = None,
    allowed_types: Tuple[str, ...] = ("LineString",),
    allow_z: bool = False,
    raise_on_error: bool = False,
    warn_on: Iterable[Severity] = (Severity.WARNING,),
) -> pd.DataFrame:
    """
    Validate a GeoDataFrame intended to contain (Multi)LineStrings.

    Returns a pandas DataFrame with columns: ['severity','code','message','count','sample_index'].
    Set raise_on_error=True to raise a ValueError if any errors are found.
    """
    issues: List[Issue] = []

    # --- Basic structure checks ---
    if not isinstance(gdf, gpd.GeoDataFrame):
        issues.append(Issue(Severity.ERROR, "NOT_GDF",
                            "Object is not a GeoDataFrame."))
        return _finalize(issues, raise_on_error, warn_on)

    if gdf.geometry is None or gdf._geometry_column_name not in gdf.columns:
        issues.append(Issue(Severity.ERROR, "NO_GEOMETRY_COLUMN",
                            "GeoDataFrame has no active geometry column named 'geometry'. "
                            "Use gdf.set_geometry('your_geom_col')."))
        return _finalize(issues, raise_on_error, warn_on)

    # Empty GeoDataFrame
    if len(gdf) == 0:
        issues.append(Issue(Severity.ERROR, "EMPTY_GDF",
                            "GeoDataFrame has 0 rows."))
        return _finalize(issues, raise_on_error, warn_on)

    # Index checks
    if not gdf.index.equals(pd.RangeIndex(start=0, stop=len(gdf))):
        issues.append(Issue(
            Severity.WARNING,
            "INDEX_NOT_RANGE_0_N1",
            "Index is not exactly 0..n-1 with step 1; "
            "consider gdf.reset_index(drop=True)."
        ))

    # null geometries (e.g. None)
    is_null = gdf.geometry.isna()
    if is_null.any():
        idx = gdf.index[is_null].tolist()[:10]
        issues.append(Issue(Severity.ERROR, "NULL_GEOMETRIES",
                            "Found null geometries.", count=int(is_null.sum()),
                            sample_index=idx))

    # empty geometries (e.g. Point())
    is_empty = gdf.geometry.is_empty
    if is_empty.any():
        idx = gdf.index[is_empty].tolist()[:10]
        issues.append(Issue(Severity.ERROR, "EMPTY_GEOMETRIES",
                            "Found empty geometries.", count=int(is_empty.sum()),
                            sample_index=idx))

    # --- Geometry types ---
    geom_types = gdf.geometry.geom_type
    bad_type = ~geom_types.isin(allowed_types)
    if bad_type.any():
        idx = gdf.index[bad_type].tolist()[:10]
        found = ", ".join(sorted(set(geom_types[bad_type])))
        issues.append(Issue(Severity.ERROR, "BAD_GEOM_TYPE",
                            f"Found non-line geometry types: {found}. "
                            f"Allowed: {', '.join(allowed_types)}.",
                            count=int(bad_type.sum()), sample_index=idx))

    # --- Validity ---
    # Note: validity for lines catches things like NaN coordinates, but self-intersections may be allowed in some domains.
    is_valid = gdf.geometry.is_valid
    if (~is_valid).any():
        idx = gdf.index[~is_valid].tolist()[:10]
        issues.append(Issue(Severity.ERROR, "INVALID_GEOM",
                            "Invalid geometries (shapely.is_valid == False).",
                            count=int((~is_valid).sum()), sample_index=idx))

    # --- LineStrings or Segments ---
    n_coordinates = gdf.count_coordinates()
    non_segmented = n_coordinates > 2
    if (non_segmented).any():
        idx = gdf.index[non_segmented].tolist()[:10]
        issues.append(Issue(Severity.INFO, "NON_SEGMENTED",
                            "Not all segments.", count=int(non_segmented.sum()),
                            sample_index=idx))
    
    # --- Zero length ---
    # (computed length is CRS-dependent; still useful as a sanity check)
    with np.errstate(all='ignore'):
        zero_len = gdf.geometry.length == 0
    if zero_len.any():
        idx = gdf.index[zero_len].tolist()[:10]
        issues.append(Issue(Severity.WARNING, "ZERO_LENGTH",
                            "Found zero-length geometries.", count=int(zero_len.sum()),
                            sample_index=idx))

    # --- Z dimension ---
    if not allow_z:
        hasz = gdf.geometry.has_z
        if hasz.any():
            idx = gdf.index[hasz].tolist()[:10]
            issues.append(Issue(Severity.WARNING,
                                "HAS_Z",
                                "Found geometries with Z coordinates.",
                                count=int(hasz.sum()),
                                sample_index=idx))

    # --- CRS checks ---
    if require_crs is not None:
        if gdf.crs is None:
            issues.append(Issue(Severity.ERROR, "NO_CRS",
                                f"Missing CRS; expected {require_crs}."))
        elif str(gdf.crs) != str(require_crs):
            issues.append(Issue(Severity.WARNING, "CRS_MISMATCH",
                                f"CRS is {gdf.crs}, expected {require_crs}."))
    issues.append(Issue(Severity.INFO, "CRS",
                        f"CRS is {gdf.crs}"))

    # --- Duplicate geometries (optional but handy) ---
    try:
        dup = gdf.geometry.to_wkb().duplicated(keep=False)
        if dup.any():
            idx = gdf.index[dup].tolist()[:10]
            issues.append(Issue(Severity.WARNING, "DUPLICATE_GEOMS",
                                "Duplicate geometries detected (by WKB).",
                                count=int(dup.sum()), sample_index=idx))
    except Exception:
        # Some geometries may not serialize cleanly; ignore this check if it fails
        pass

    # --- Summarize and finish ---
    return _finalize(issues, raise_on_error, warn_on)


def _finalize(issues: List[Issue], raise_on_error: bool, warn_on: Iterable[Severity]) -> pd.DataFrame:
    df = pd.DataFrame([i.__dict__ for i in issues]) if issues else pd.DataFrame(
        [{"severity": "info", "code": "OK", "message": "All checks passed.", "count": None, "sample_index": None}]
    )

    # Emit warnings for selected severities
    warn_levels = set(warn_on)
    for _, row in df.iterrows():
        if Severity(row["severity"]) in warn_levels:
            warnings.warn(f"[{row['code']}] {row['message']} (count={row['count']})", stacklevel=2)

    # Optionally raise on any error
    if raise_on_error and (df["severity"] == "error").any():
        # Include the first error message for clarity
        first = df[df["severity"] == "error"].iloc[0]
        raise ValueError(f"Validation errors detected. First error: [{first['code']}] {first['message']}")

    return df
