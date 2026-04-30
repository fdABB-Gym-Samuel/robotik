"""Helpers for acquiring and converting the official Unitree Inspire hand."""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path
from xml.etree import ElementTree

UNITREE_ROS_REPO = "https://github.com/unitreerobotics/unitree_ros.git"
HAND_URDF_RELATIVE_PATH = Path("robots/g1_description/inspire_hand/FTP_right_hand.urdf")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_asset_checkout() -> Path:
    return project_root() / "runs" / "assets" / "unitree_ros"


def ensure_unitree_g1_assets(asset_root: Path | None = None) -> Path:
    """Ensure the official Unitree Inspire hand assets are available locally."""
    checkout_dir = Path(
        asset_root or os.environ.get("UNITREE_G1_ASSET_DIR") or default_asset_checkout()
    )
    hand_urdf_path = checkout_dir / HAND_URDF_RELATIVE_PATH
    runtime_model_path = (
        checkout_dir
        / "robots"
        / "g1_description"
        / "inspire_hand"
        / "FTP_right_hand_runtime.xml"
    )

    if hand_urdf_path.exists():
        return _prepare_hand_runtime_model(hand_urdf_path, runtime_model_path)

    checkout_dir.parent.mkdir(parents=True, exist_ok=True)

    if not (checkout_dir / ".git").exists():
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                UNITREE_ROS_REPO,
                str(checkout_dir),
            ]
        )
        _run(
            ["git", "sparse-checkout", "set", "robots/g1_description"], cwd=checkout_dir
        )
    else:
        _run(["git", "pull", "--ff-only"], cwd=checkout_dir)

    if not hand_urdf_path.exists():
        raise FileNotFoundError(
            "Downloaded Unitree assets, but could not find "
            f"'{HAND_URDF_RELATIVE_PATH.as_posix()}' in {checkout_dir}"
        )

    return _prepare_hand_runtime_model(hand_urdf_path, runtime_model_path)


def _prepare_hand_runtime_model(hand_urdf_path: Path, runtime_model_path: Path) -> Path:
    if (
        runtime_model_path.exists()
        and runtime_model_path.stat().st_mtime >= hand_urdf_path.stat().st_mtime
    ):
        return runtime_model_path

    robot_root = ElementTree.parse(hand_urdf_path).getroot()
    model_dir = hand_urdf_path.parent

    links = {link.get("name"): link for link in robot_root.findall("link")}
    joints = [joint for joint in robot_root.findall("joint")]
    children_by_parent: dict[str, list[ElementTree.Element]] = {}
    child_link_names: set[str] = set()
    for joint in joints:
        parent_name = joint.find("parent").get("link")
        child_name = joint.find("child").get("link")
        children_by_parent.setdefault(parent_name, []).append(joint)
        child_link_names.add(child_name)

    root_link_name = next(name for name in links if name not in child_link_names)

    mujoco_root = ElementTree.Element("mujoco", {"model": "unitree_inspire_right_hand"})
    ElementTree.SubElement(mujoco_root, "compiler", {"angle": "radian"})
    option = ElementTree.SubElement(
        mujoco_root, "option", {"gravity": "0 0 0", "timestep": "0.005"}
    )
    option.set("integrator", "implicitfast")

    asset = ElementTree.SubElement(mujoco_root, "asset")
    _add_mesh_assets(asset, links, model_dir)

    worldbody = ElementTree.SubElement(mujoco_root, "worldbody")
    ElementTree.SubElement(
        worldbody, "light", {"pos": "0 0 1.4", "dir": "0 0 -1", "directional": "true"}
    )
    base_body = ElementTree.SubElement(
        worldbody, "body", {"name": root_link_name, "pos": "0 0 0"}
    )
    ElementTree.SubElement(
        base_body,
        "freejoint",
        {"name": "hand_freejoint"},
    )
    _populate_link_body(base_body, links[root_link_name], mesh_prefix="mesh")
    _build_body_tree(base_body, root_link_name, links, children_by_parent)

    actuator = ElementTree.SubElement(mujoco_root, "actuator")
    for joint in joints:
        if joint.get("type") == "revolute":
            name = joint.get("name")
            ctrlrange = _ctrlrange(joint)
            ElementTree.SubElement(
                actuator,
                "position",
                {"name": name, "joint": name, "kp": "10", "ctrlrange": ctrlrange},
            )

    runtime_model_path.parent.mkdir(parents=True, exist_ok=True)
    ElementTree.ElementTree(mujoco_root).write(
        runtime_model_path, encoding="utf-8", xml_declaration=False
    )
    return runtime_model_path


def _add_mesh_assets(
    asset: ElementTree.Element, links: dict[str, ElementTree.Element], model_dir: Path
) -> None:
    trimesh = _import_trimesh()
    seen: set[str] = set()
    for link_name, link in links.items():
        for visual in link.findall("visual"):
            mesh_elem = visual.find("./geometry/mesh")
            if mesh_elem is None:
                continue
            filename = mesh_elem.get("filename")
            if not filename:
                continue
            mesh_name = f"mesh_{link_name}"
            if mesh_name in seen:
                continue
            seen.add(mesh_name)
            mesh_path = _resolve_mesh_path(model_dir, filename)
            mesh = trimesh.load_mesh(mesh_path, force="mesh")
            if hasattr(mesh, "dump") and not hasattr(mesh, "vertices"):
                mesh = mesh.dump(concatenate=True)
            vertices = getattr(mesh, "vertices", None)
            faces = getattr(mesh, "faces", None)
            if vertices is None or faces is None:
                raise RuntimeError(
                    f"Could not extract triangle mesh data from {mesh_path}"
                )
            ElementTree.SubElement(
                asset,
                "mesh",
                {
                    "name": mesh_name,
                    "vertex": _flatten_floats(vertices),
                    "face": _flatten_ints(faces),
                },
            )


def _build_body_tree(
    parent_body: ElementTree.Element,
    parent_link_name: str,
    links: dict[str, ElementTree.Element],
    children_by_parent: dict[str, list[ElementTree.Element]],
) -> None:
    for joint in children_by_parent.get(parent_link_name, []):
        child_link_name = joint.find("child").get("link")
        child_body = ElementTree.SubElement(
            parent_body,
            "body",
            {
                "name": child_link_name,
                "pos": _origin_xyz(joint),
                "quat": _origin_quat(joint),
            },
        )

        if joint.get("type") == "revolute":
            joint_attrs = {
                "name": joint.get("name"),
                "type": "hinge",
                "axis": _axis_xyz(joint),
                "range": _joint_range(joint),
            }
            ElementTree.SubElement(child_body, "joint", joint_attrs)

        _populate_link_body(child_body, links[child_link_name], mesh_prefix="mesh")
        _build_body_tree(child_body, child_link_name, links, children_by_parent)


def _populate_link_body(
    body: ElementTree.Element, link: ElementTree.Element, mesh_prefix: str
) -> None:
    for visual in link.findall("visual"):
        mesh_elem = visual.find("./geometry/mesh")
        if mesh_elem is None:
            continue
        origin = visual.find("origin")
        color = visual.find("./material/color")
        rgba = color.get("rgba") if color is not None else "0.9 0.9 0.9 1"
        ElementTree.SubElement(
            body,
            "geom",
            {
                "type": "mesh",
                "mesh": f"{mesh_prefix}_{link.get('name')}",
                "pos": _xyz_from_origin(origin),
                "quat": _quat_from_origin(origin),
                "rgba": rgba,
                "contype": "0",
                "conaffinity": "0",
            },
        )


def _origin_xyz(joint: ElementTree.Element) -> str:
    origin = joint.find("origin")
    return _xyz_from_origin(origin)


def _origin_quat(joint: ElementTree.Element) -> str:
    origin = joint.find("origin")
    return _quat_from_origin(origin)


def _xyz_from_origin(origin: ElementTree.Element | None) -> str:
    if origin is None:
        return "0 0 0"
    return origin.get("xyz", "0 0 0")


def _quat_from_origin(origin: ElementTree.Element | None) -> str:
    if origin is None:
        return "1 0 0 0"
    rpy = origin.get("rpy", "0 0 0").split()
    roll, pitch, yaw = (float(value) for value in rpy)
    return _rpy_to_quat(roll, pitch, yaw)


def _rpy_to_quat(roll: float, pitch: float, yaw: float) -> str:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return f"{w:.9g} {x:.9g} {y:.9g} {z:.9g}"


def _axis_xyz(joint: ElementTree.Element) -> str:
    axis = joint.find("axis")
    return axis.get("xyz", "0 0 1") if axis is not None else "0 0 1"


def _joint_range(joint: ElementTree.Element) -> str:
    limit = joint.find("limit")
    lower = limit.get("lower", "0") if limit is not None else "0"
    upper = limit.get("upper", "0") if limit is not None else "0"
    return f"{lower} {upper}"


def _ctrlrange(joint: ElementTree.Element) -> str:
    return _joint_range(joint)


def _flatten_floats(values) -> str:
    return " ".join(f"{float(value):.9g}" for row in values for value in row)


def _flatten_ints(values) -> str:
    return " ".join(str(int(value)) for row in values for value in row)


def _resolve_mesh_path(model_dir: Path, filename: str) -> Path:
    candidates = [
        (model_dir / filename).resolve(),
        (model_dir.parent / filename).resolve(),
        (model_dir.parent / "meshes" / Path(filename).name).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve Unitree hand mesh path for '{filename}'"
    )


def _import_trimesh():
    try:
        import trimesh
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The G1 asset preparation step requires the `trimesh` package, "
            "which should be available inside the Nix shell."
        ) from exc
    return trimesh


def _run(command: list[str], cwd: Path | None = None) -> None:
    try:
        subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Git is required to download the official Unitree assets. "
            "Run this inside your nix shell or install git locally."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Command failed while preparing Unitree assets: {' '.join(command)}"
        ) from exc
