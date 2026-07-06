"""CARLA parking-lot vector map — drop-in for UniAD's VectorizedLocalMap (approach B).

Supplies UniAD's seg head with CARLA lot map GT WITHOUT a NuScenesMap: it returns
the same `vectors` format that VectorizedLocalMap.gen_vectorized_samples produces
(list of {'pts': Nx2 ego-local meters, 'pts_num': int, 'type': int}), built from
the precomputed lot geometry (scripts/build_lot_map_gt.py output, nuScenes global
frame). preprocess_map() then rasterizes these into the BEV seg masks unchanged.

Map classes (num_classes=3): 0=divider, 1=ped_crossing, 2=contour/boundary.
Parking lot mapping (per project decision — lot is all-drivable, no painted road
lanes):
  * parking-slot polygon outlines  -> type 2 (boundary)   [the semantic bays]
  * drivable-area outline          -> type 2 (boundary)   [lot extent]
  * aisle centrelines              -> type 0 (divider)     [optional hint]
  * ped_crossing                   -> none

The vectors are in the ego-local patch frame: translate by -ego, rotate by -yaw,
clip to the patch box, sample along boundaries at `sample_dist`.
"""
import json
import math
import pathlib

import numpy as np
from pyquaternion import Quaternion
from shapely import affinity
from shapely.geometry import LineString, Polygon, box


def _quaternion_yaw(q):
    q = Quaternion(q)
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
    return math.atan2(v[1], v[0])


class CarlaVectorMap:
    def __init__(self, lot_gt_json, patch_size=(102.4, 102.4),
                 canvas_size=(200, 200), sample_dist=1.0,
                 use_aisle_lines=True):
        gt = json.loads(pathlib.Path(lot_gt_json).read_text())
        self.patch_size = patch_size
        self.canvas_size = canvas_size
        self.sample_dist = sample_dist
        self.use_aisle_lines = use_aisle_lines
        # Global-frame shapely geometry (built once).
        self._slot_polys = [Polygon(s["polygon"]) for s in gt["parking_slots"]]
        self._drivable = Polygon(gt["drivable_area"])
        self._aisle = [LineString(l) for l in gt.get("aisle_lines", [])]

    def _to_local(self, geom, ego_xy, yaw):
        """Global geom -> ego-local patch frame (translate -ego, rotate -yaw)."""
        g = affinity.translate(geom, xoff=-ego_xy[0], yoff=-ego_xy[1])
        g = affinity.rotate(g, -math.degrees(yaw), origin=(0, 0))
        return g

    def _sample(self, line):
        n = max(2, int(line.length / self.sample_dist) + 1)
        ds = np.linspace(0, line.length, n)
        return np.array([list(line.interpolate(d).coords)[0] for d in ds])

    def _emit(self, geom_local, vtype, patch, out):
        """Clip a local line/boundary to the patch and append sampled vectors."""
        clipped = geom_local.intersection(patch)
        if clipped.is_empty:
            return
        parts = getattr(clipped, "geoms", [clipped])
        for part in parts:
            if part.length < 0.5:
                continue
            pts = self._sample(LineString(part.coords) if part.geom_type == "LineString"
                               else LineString(part.exterior.coords))
            if len(pts) >= 2:
                out.append({"pts": pts.astype(float), "pts_num": len(pts), "type": vtype})

    def gen_vectorized_samples(self, location, ego2global_translation, ego2global_rotation):
        ego_xy = ego2global_translation[:2]
        yaw = _quaternion_yaw(ego2global_rotation)
        hx, hy = self.patch_size[1] / 2, self.patch_size[0] / 2
        patch = box(-hx, -hy, hx, hy)

        vectors = []
        # parking-slot outlines -> boundary (type 2)
        for poly in self._slot_polys:
            gl = self._to_local(poly, ego_xy, yaw)
            self._emit(LineString(gl.exterior.coords), 2, patch, vectors)
        # drivable-area outline -> boundary (type 2)
        dl = self._to_local(self._drivable, ego_xy, yaw)
        self._emit(LineString(dl.exterior.coords), 2, patch, vectors)
        # aisle centrelines -> divider (type 0)
        if self.use_aisle_lines:
            for ln in self._aisle:
                self._emit(self._to_local(ln, ego_xy, yaw), 0, patch, vectors)
        return vectors
