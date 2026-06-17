"""
Build NuScenes DB tables (v1.0-carla) from the CARLA parking pkl,
then patch the pkl with missing lidar/camera fields.

Run from the project root:
  python scripts/build_carla_nusc_tables.py

Outputs:
  data/nuscenes/v1.0-carla/  (NuScenes DB tables for CARLA episodes)
  data_carla/processed/parking_infos_temporal.pkl  (patched in-place)
"""

import argparse
import json
import math
import pathlib
import pickle

import numpy as np
from pyquaternion import Quaternion

import paths

_ROOT = paths.PROJECT_ROOT

# ---------------------------------------------------------------------------
# Real nuScenes calibrations used as proxy for CARLA sensors.
# All tokens are stable across v1.0-mini samples.
# ---------------------------------------------------------------------------
NUSC_SENSORS = {
    "LIDAR_TOP": {
        "sensor_token": "dc8b396651c05aedbb9cdaae573bb567",
        "cs_token":     "a183049901c24361a6b0b11b8013137c",
        "translation":  [0.943713, 0.0, 1.84023],
        "rotation":     [0.7077955119163518, -0.006492242056004365,
                         0.010646214713995808, -0.7063073142877817],
        "modality":     "lidar",
        "camera_intrinsic": [],
    },
    "CAM_FRONT": {
        "sensor_token": "725903f5b62f56118f4094b46a4470d8",
        "cs_token":     "1d31c729b073425e8e0202c5c6e66ee1",
        "translation":  [1.70079118954, 0.0159456324149, 1.51095763913],
        "rotation":     [0.4998015430569128, -0.5030316162024876,
                         0.4997798114386805, -0.49737083824542755],
        "modality":     "camera",
    },
    "CAM_FRONT_LEFT": {
        "sensor_token": "ec4b5d41840a509984f7ec36419d4c09",
        "cs_token":     "75ad8e2a8a3f4594a13db2398430d097",
        "translation":  [1.52387798135, 0.494631336551, 1.50932822144],
        "rotation":     [0.6757265034669446, -0.6736266522251881,
                         0.21214015046209478, -0.21122827103904068],
        "modality":     "camera",
    },
    "CAM_FRONT_RIGHT": {
        "sensor_token": "2f7ad058f1ac5557bf321c7543758f43",
        "cs_token":     "f8d0aaa1a8234ba3aeed5867e0aa81aa",
        "translation":  [1.5508477543, -0.493404796419, 1.49574800619],
        "rotation":     [0.2060347966337182, -0.2026940577919598,
                         0.6824507824531167, -0.6713610884174485],
        "modality":     "camera",
    },
    "CAM_BACK_LEFT": {
        "sensor_token": "a89643a5de885c6486df2232dc954da2",
        "cs_token":     "3bc29be787ea4fc79144c4a46a3c91ca",
        "translation":  [1.03569100218, 0.484795032713, 1.59097014818],
        "rotation":     [0.6924185592174665, -0.7031619420114925,
                         -0.11648342771943819, 0.11203317912370753],
        "modality":     "camera",
    },
    "CAM_BACK": {
        "sensor_token": "ce89d4f3050b5892b33b3d328c5e82a3",
        "cs_token":     "4ff47c4950f04cb4be1876bc0b028326",
        "translation":  [0.0283260309358, 0.00345136761476, 1.57910346144],
        "rotation":     [0.5037872666382278, -0.49740249788611096,
                         -0.4941850223835201, 0.5045496097725578],
        "modality":     "camera",
    },
    "CAM_BACK_RIGHT": {
        "sensor_token": "ca7dba2ec9f95951bbe67246f7f2c3f7",
        "cs_token":     "3b00acc55ed941fa9f405e0c1fd2b639",
        "translation":  [1.0148780988, -0.480568219723, 1.56239545128],
        "rotation":     [0.12280980120078765, -0.132400842670559,
                         -0.7004305821388234, 0.690496031265798],
        "modality":     "camera",
    },
}

CAMERA_NAMES = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

# CARLA intrinsic (1600×900, FOV=70°)
_f = 1600 / (2 * math.tan(math.radians(70 / 2)))
CARLA_CAM_INTRINSIC = [[_f, 0.0, 800.0], [0.0, _f, 450.0], [0.0, 0.0, 1.0]]

MAP_LOCATION = "boston-seaport"  # valid nuScenes map location for map lookups


def build_tables(pkl_path: pathlib.Path, out_dir: pathlib.Path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    infos = data["infos"]

    # Group frames by scene (episode)
    scenes_map: dict[str, list] = {}
    for info in infos:
        st = info["scene_token"]
        scenes_map.setdefault(st, []).append(info)

    # --- Static tables ---
    sensor_table = []
    cs_table = []
    for ch, cal in NUSC_SENSORS.items():
        sensor_table.append({
            "token": cal["sensor_token"],
            "channel": ch,
            "modality": cal["modality"],
        })
        entry = {
            "token": cal["cs_token"],
            "sensor_token": cal["sensor_token"],
            "translation": cal["translation"],
            "rotation": cal["rotation"],
            "camera_intrinsic": CARLA_CAM_INTRINSIC if cal["modality"] == "camera" else [],
        }
        cs_table.append(entry)

    # --- Per-episode / per-frame tables ---
    scene_table = []
    log_table = []
    sample_table = []
    sample_data_table = []
    ego_pose_table = []

    for scene_token, scene_infos in scenes_map.items():
        scene_infos = sorted(scene_infos, key=lambda x: x["frame_idx"])
        n = len(scene_infos)
        log_token = f"log_{scene_token}"

        log_table.append({
            "token": log_token,
            "vehicle": "CARLA",
            "date_captured": "2024-01-01",
            "location": MAP_LOCATION,
            "logfile": "",
            "vehicle_name": "CARLA",
        })

        scene_table.append({
            "token": scene_token,
            "log_token": log_token,
            "nbr_samples": n,
            "first_sample_token": scene_infos[0]["token"],
            "last_sample_token": scene_infos[-1]["token"],
            "name": scene_token,
            "description": f"CARLA parking episode {scene_token}",
        })

        for idx, info in enumerate(scene_infos):
            sample_token = info["token"]
            ts = int(info["timestamp"])
            ep_token = f"ep_{sample_token}"
            lidar_sd_token = f"sd_lidar_{sample_token}"

            # ego_pose (shared by all sensors for this frame)
            ego_pose_table.append({
                "token": ep_token,
                "timestamp": ts,
                "rotation": info["ego2global_rotation"],
                "translation": [float(x) for x in info["ego2global_translation"]],
            })

            # lidar sample_data
            prev_lidar = f"sd_lidar_{scene_infos[idx-1]['token']}" if idx > 0 else ""
            next_lidar = f"sd_lidar_{scene_infos[idx+1]['token']}" if idx < n-1 else ""
            sample_data_table.append({
                "token": lidar_sd_token,
                "sample_token": sample_token,
                "ego_pose_token": ep_token,
                "calibrated_sensor_token": NUSC_SENSORS["LIDAR_TOP"]["cs_token"],
                "timestamp": ts,
                "fileformat": "pcd",
                "is_key_frame": True,
                "height": 0,
                "width": 0,
                "filename": "",
                "prev": prev_lidar,
                "next": next_lidar,
                "sensor_modality": "lidar",
                "channel": "LIDAR_TOP",
            })

            # camera sample_data
            cam_data_tokens = {"LIDAR_TOP": lidar_sd_token}
            for cam in CAMERA_NAMES:
                cam_sd_token = f"sd_{cam}_{sample_token}"
                prev_cam = f"sd_{cam}_{scene_infos[idx-1]['token']}" if idx > 0 else ""
                next_cam = f"sd_{cam}_{scene_infos[idx+1]['token']}" if idx < n-1 else ""
                img_filename = info["cams"][cam]["data_path"] if cam in info["cams"] else ""
                sample_data_table.append({
                    "token": cam_sd_token,
                    "sample_token": sample_token,
                    "ego_pose_token": ep_token,
                    "calibrated_sensor_token": NUSC_SENSORS[cam]["cs_token"],
                    "timestamp": ts,
                    "fileformat": "jpg",
                    "is_key_frame": True,
                    "height": 900,
                    "width": 1600,
                    "filename": img_filename,
                    "prev": prev_cam,
                    "next": next_cam,
                    "sensor_modality": "camera",
                    "channel": cam,
                })
                cam_data_tokens[cam] = cam_sd_token

            sample_table.append({
                "token": sample_token,
                "timestamp": ts,
                "prev": scene_infos[idx-1]["token"] if idx > 0 else "",
                "next": scene_infos[idx+1]["token"] if idx < n-1 else "",
                "scene_token": scene_token,
                "anns": [],
                "data": cam_data_tokens,
            })

    # Map table: one entry for boston-seaport referencing all CARLA log tokens.
    # NuScenes' __make_reverse_index__ requires at least one map record.
    all_log_tokens = [f"log_{st}" for st in scenes_map]
    map_table = [{
        "token": "carla_map_boston_seaport",
        "category": "semantic_prior",
        "filename": "maps/53992ee3023e5494b90c316c183be829.png",
        "log_tokens": all_log_tokens,
    }]

    # Empty tables (no agents, annotations, etc.)
    empty = {
        "sample_annotation": [],
        "instance": [],
        "category": [],
        "attribute": [],
        "visibility": [],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "sensor": sensor_table,
        "calibrated_sensor": cs_table,
        "scene": scene_table,
        "log": log_table,
        "sample": sample_table,
        "sample_data": sample_data_table,
        "ego_pose": ego_pose_table,
        "map": map_table,
        **empty,
    }
    for name, rows in tables.items():
        (out_dir / f"{name}.json").write_text(json.dumps(rows))
    (out_dir / "version.txt").write_text("v1.0-carla\n")
    print(f"Wrote {len(scene_table)} scenes, {len(sample_table)} samples, "
          f"{len(sample_data_table)} sample_data, {len(ego_pose_table)} ego_poses")
    print(f"→ {out_dir}")


def _abs_image_path(stored_path: str, raw_dir: pathlib.Path) -> str:
    """Resolve a stored camera path to an absolute path under raw_dir.

    Handles relative ('data/raw/...'), already-absolute, and any other prefix
    uniformly by keying off the 'raw/' segment, so image paths never break when
    the dataset moves. Derived from raw_dir (= repo root), so it is portable.
    """
    p = stored_path.replace("\\", "/")
    marker = "raw/"
    idx = p.rfind(marker)
    if idx == -1:
        return stored_path  # unexpected layout; leave untouched
    rel = p[idx + len(marker):]  # e.g. episode_0000/frames/frame_0000/CAM_FRONT.jpg
    return str(raw_dir / rel)


def patch_pkl(pkl_path: pathlib.Path, raw_dir: pathlib.Path):
    """Add missing lidar/camera fields to the pkl in-place."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Update metadata to v1.0-carla
    data["metadata"] = {"version": "v1.0-carla"}

    for info in data["infos"]:
        # Resolve every camera image path to absolute (kills the relative-path
        # FileNotFoundError that bites whenever the dataset is moved/regenerated).
        for cam_info in info.get("cams", {}).values():
            if "data_path" in cam_info:
                cam_info["data_path"] = _abs_image_path(cam_info["data_path"], raw_dir)

        # Lidar stubs (no physical lidar; treat ego frame as lidar frame)
        if "lidar_path" not in info:
            info["lidar_path"] = ""
        if "sweeps" not in info:
            info["sweeps"] = []
        if "lidar2ego_rotation" not in info:
            info["lidar2ego_rotation"] = [1.0, 0.0, 0.0, 0.0]  # identity quat
        if "lidar2ego_translation" not in info:
            info["lidar2ego_translation"] = np.array([0.0, 0.0, 0.0])
        if "num_lidar_pts" not in info:
            info["num_lidar_pts"] = np.zeros(0, dtype=np.int32)

        # Camera sensor2lidar (= sensor2ego since lidar=ego)
        for cam, cal in NUSC_SENSORS.items():
            if cam == "LIDAR_TOP" or cam not in info.get("cams", {}):
                continue
            cam_info = info["cams"][cam]
            if "sensor2lidar_rotation" not in cam_info:
                q = Quaternion(cal["rotation"])
                cam_info["sensor2lidar_rotation"] = q.rotation_matrix
            if "sensor2lidar_translation" not in cam_info:
                cam_info["sensor2lidar_translation"] = np.array(cal["translation"])

    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Patched {len(data['infos'])} records in {pkl_path}")


def main():
    ap = argparse.ArgumentParser(description="Build v1.0-carla NuScenes tables and patch the infos pkl.")
    ap.add_argument("--infos", type=pathlib.Path, default=paths.INFOS_PKL,
                    help="Input infos pkl (patched in-place).")
    ap.add_argument("--raw_dir", type=pathlib.Path, default=paths.RAW_DIR,
                    help="Raw episodes dir; camera paths are resolved as absolute under here.")
    ap.add_argument("--out_dir", type=pathlib.Path, default=paths.NUSC_DB,
                    help="Output NuScenes DB tables dir.")
    args = ap.parse_args()

    print("=== Building v1.0-carla NuScenes tables ===")
    build_tables(args.infos, args.out_dir)

    print("\n=== Patching CARLA pkl ===")
    patch_pkl(args.infos, args.raw_dir)

    print("\nDone. Update carla_parking.py data_root to use v1.0-carla (set via metadata).")


if __name__ == "__main__":
    main()
