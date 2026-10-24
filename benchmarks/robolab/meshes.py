"""Surface points of RoboLab's objects from their USD scene, in each object's own frame (the rigid body's, whose pose
the ground-truth state reports), for the planner's mirror: their convex hulls for collisions and their shapes for
grasps. Needs USD's Python bindings (pxr), which Isaac Sim provides; on a CPU, export them from the command line:

  python -m benchmarks.robolab.meshes assets/scenes/banana_bowl.usda banana bowl   # in RoboLab's environment
"""

import argparse
import json

import numpy as np

MAX_POINTS = 2000  # Per object; a convex hull and grasp search need no more


def rigid_body(prim):
    """The prim itself or its first descendant with a rigid body, whose pose the simulator reports."""
    from pxr import Usd, UsdPhysics

    for child in Usd.PrimRange(prim):
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            return child
    return prim


def object_points(usd_path, names, max_points=MAX_POINTS):
    """{name: (n, 3) array}: the mesh vertices of each named object of the scene (a prim of that name), in its rigid
    body's frame, in meters. Objects that the scene does not have are left out."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    meters = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    cache = UsdGeom.XformCache()
    prims = {prim.GetName(): prim for prim in stage.Traverse() if prim.GetName() in set(names)}
    points = {}
    for name, prim in prims.items():
        root = Gf.Transform(cache.GetLocalToWorldTransform(rigid_body(prim)))
        # The rigid body's pose excludes its scale, which stays in the points
        root_from_world = Gf.Matrix4d().SetRotateOnly(root.GetRotation()).SetTranslateOnly(root.GetTranslation())
        root_from_world = root_from_world.GetInverse()
        vertices = []
        for mesh in Usd.PrimRange(prim):
            if not mesh.IsA(UsdGeom.Mesh):
                continue
            local = UsdGeom.Mesh(mesh).GetPointsAttr().Get() or []
            to_root = cache.GetLocalToWorldTransform(mesh) * root_from_world
            vertices.extend(to_root.Transform(Gf.Vec3d(*point)) for point in local)
        if vertices:
            array = meters * np.array(vertices, dtype=float)
            stride = max(1, -(-len(array) // max_points))
            points[name] = array[::stride]
    return points


def main():
    parser = argparse.ArgumentParser(description='Print the surface points of objects of a RoboLab scene as JSON.')
    parser.add_argument('scene', help='A RoboLab scene (.usd or .usda)')
    parser.add_argument('names', nargs='+', help='Object names')
    args = parser.parse_args()
    points = object_points(args.scene, args.names)
    print(json.dumps({name: np.round(array, 5).tolist() for name, array in points.items()}))


if __name__ == '__main__':
    main()
