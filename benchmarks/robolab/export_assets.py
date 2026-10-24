"""Exports the objects of RoboLab scenes as the box approximations the planner uses (ObjectSpecs), from their USD
files' bounding boxes. Needs usd-core (pip install usd-core), which runs on a CPU.

Usage:
  python -m benchmarks.robolab.export_assets scene.usda --containers bowl basket --surfaces table --output scene.json
Objects not listed as containers or surfaces are movable.
"""

import argparse
import json

from pxr import Usd, UsdGeom


def export(path, containers=(), surfaces=(), names=None):
    stage = Usd.Stage.Open(path)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    specs = []
    for prim in stage.Traverse():
        name = prim.GetName()
        if not prim.IsA(UsdGeom.Xformable) or (names and name not in names):
            continue
        size = cache.ComputeWorldBound(prim).ComputeAlignedRange().GetSize()
        if min(size) <= 0:
            continue
        kind = 'container' if name in containers else 'surface' if name in surfaces else 'movable'
        specs.append({'name': name, 'size': [round(value, 4) for value in size], 'kind': kind})
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene', help='A USD scene file')
    parser.add_argument('--objects', nargs='*', default=None, help="The task's objects (its contact_object_list)")
    parser.add_argument('--containers', nargs='*', default=[])
    parser.add_argument('--surfaces', nargs='*', default=[])
    parser.add_argument('--output', default='scene.json')
    args = parser.parse_args()
    specs = export(args.scene, args.containers, args.surfaces, args.objects)
    with open(args.output, 'w') as file:
        json.dump(specs, file, indent=1)
    print(f'{len(specs)} objects written to {args.output}')


if __name__ == '__main__':
    main()
