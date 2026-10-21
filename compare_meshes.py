"""Save RGB | metadata-rendered ground truth | predicted mesh comparisons.

Ground truth uses its stored world coordinates and metadata camera, material,
light and floor. The predicted mesh is a centered, scaled shape preview with
a generic camera; it is not aligned to ground truth. --preserve-coordinates
keeps the prediction's stored coordinates. Requires PAZ with its JAX backend.
"""

import argparse
import csv
import json
import os
from pathlib import Path


def parse_args():
    root = Path(__file__).resolve().parent / 'data/test_split'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb-dir', type=Path, default=root / 'rgb')
    parser.add_argument('--original-dir', type=Path, default=root / 'original_meshes')
    parser.add_argument('--metadata-dir', type=Path, default=root / 'metadata')
    parser.add_argument('--y-fov', type=float, default=45.0,
                        help='Generator vertical field of view in degrees (not stored in metadata).')
    parser.add_argument('--no-floor', action='store_true')
    parser.add_argument('--no-shadows', action='store_true')
    parser.add_argument('--chunk-size', type=int, default=512)
    parser.add_argument('--face-chunk-size', type=int, default=64)
    parser.add_argument('--predicted-dir', type=Path, default=root / 'predicted_meshes')
    parser.add_argument('--output-dir', type=Path, default=root / 'comparisons')
    parser.add_argument('--max-images', type=int)
    parser.add_argument('--elevation', type=float, default=20)
    parser.add_argument('--azimuth', type=float, default=-60)
    parser.add_argument('--dpi', type=int, default=150)
    parser.add_argument('--preview-grid', type=int, default=64,
                        help='Cluster dense mesh vertices onto this grid for rendering; 0 keeps all faces.')
    parser.add_argument('--preserve-coordinates', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        parser.error('--max-images must be positive')
    if args.dpi < 1:
        parser.error('--dpi must be positive')
    if not 0 < args.y_fov < 180 or min(args.chunk_size, args.face_chunk_size) < 1:
        parser.error('Require 0 < --y-fov < 180 and positive chunk sizes')
    if args.preview_grid < 0 or args.preview_grid == 1:
        parser.error('--preview-grid must be 0 or at least 2')
    return args


def mesh_index(directory):
    extensions = {'.ply', '.obj', '.glb', '.gltf', '.stl', '.off'}
    result = {}
    # Prefer PLY when both PLY and GLB exports exist.
    for path in sorted(directory.iterdir(), key=lambda p: (p.suffix.lower() != '.ply', p.name)):
        if path.is_file() and path.suffix.lower() in extensions:
            result.setdefault(path.stem, path)
    return result


def load_mesh(path, normalize, preview_grid):
    import numpy as np
    import trimesh

    mesh = trimesh.load(str(path), force='mesh', process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f'No triangle mesh in {path}')
    vertices = np.asarray(mesh.vertices, dtype=float).copy()
    if not np.isfinite(vertices).all():
        raise ValueError(f'Non-finite vertices in {path}')
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    extent = (upper - lower).max()
    if extent <= 0:
        raise ValueError(f'Degenerate mesh in {path}')
    faces = np.asarray(mesh.faces)
    if preview_grid and len(faces) > 50000:
        # Vertex clustering preserves a complete surface preview rather than
        # dropping a random subset of triangles. Source files remain untouched.
        cells = np.round((vertices - lower) / extent * (preview_grid - 1)).astype(int)
        _, inverse = np.unique(cells, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        vertices = np.column_stack([
            np.bincount(inverse, weights=vertices[:, axis]) / counts
            for axis in range(3)])
        faces = inverse[faces]
        valid = ((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
                 & (faces[:, 0] != faces[:, 2]))
        faces = faces[valid]
        _, unique = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
        faces = faces[unique]
    if normalize:
        vertices = (vertices - (lower + upper) / 2) / extent
    return vertices, faces


def render_ground_truth(original_path, metadata, image_shape, args):
    """Render the world-space PLY without applying its object pose twice."""
    os.environ.setdefault('KERAS_BACKEND', 'jax')
    import jax.numpy as jp
    import numpy as np
    import paz
    import paz.graphics
    from paz.backend.lie import SE3

    points, faces = load_mesh(original_path, normalize=False, preview_grid=0)
    values = metadata['material']
    color = jp.asarray(values['color_rgb'])
    material = paz.graphics.Material(
        color, values['ambient'], values['diffuse'],
        values['specular'], values['shininess'])
    edges = np.unique(np.sort(np.concatenate([
        faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    nodes = [paz.graphics.Mesh(
        jp.asarray(points, dtype=jp.float32), jp.tile(color, (len(points), 1)),
        jp.eye(4), material, jp.asarray(faces, dtype=jp.int32),
        jp.asarray(edges, dtype=jp.int32))]
    if not args.no_floor:
        floor_material = paz.graphics.Material(
            jp.asarray([0.72, 0.72, 0.72]), 0.18, 0.72, 0.05, 30.0)
        nodes.insert(0, paz.graphics.Plane(material=floor_material))
    camera = metadata['camera']
    world_to_camera = SE3.view_transform(
        jp.asarray(camera['position_world_xyz']),
        jp.asarray(camera['target_world_xyz']), jp.asarray([0., 1., 0.]))
    light_camera = np.append(metadata['light']['position_camera_xyz'], 1.)
    light_world = np.linalg.inv(np.asarray(world_to_camera)) @ light_camera
    light = paz.graphics.PointLight(
        jp.full(3, metadata['light']['intensity']), jp.asarray(light_world[:3]))
    rgb, _ = paz.graphics.render(
        image_shape, np.deg2rad(args.y_fov), world_to_camera,
        paz.graphics.Scene(nodes), None, light, tiles=(1, 1),
        chunk_size=args.chunk_size, face_chunk_size=args.face_chunk_size,
        shadows=not args.no_shadows)
    return np.clip(np.asarray(rgb), 0., 1.)


def save_comparison(image_path, original_path, predicted_path, output_path, args):
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image, ImageOps

    with (args.metadata_dir / f'{image_path.stem}.json').open() as file:
        metadata = json.load(file)
    with Image.open(image_path) as image:
        original_rgb = np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
    ground_truth_rgb = render_ground_truth(original_path, metadata, original_rgb.shape[:2], args)
    meshes = [load_mesh(predicted_path, not args.preserve_coordinates, args.preview_grid)]
    vertices = np.concatenate([mesh[0] for mesh in meshes])
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    center = (lower + upper) / 2
    radius = max(float((upper - lower).max()) * 0.55, 1e-6)
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    try:
        rgb_axis = fig.add_subplot(131)
        rgb_axis.imshow(original_rgb)
        rgb_axis.set_title('Original RGB')
        rgb_axis.axis('off')
        gt_axis = fig.add_subplot(132)
        gt_axis.imshow(ground_truth_rgb)
        gt_axis.set_title('Ground truth (PLY + metadata)')
        gt_axis.axis('off')
        for index, ((points, faces), title) in enumerate(zip(
                meshes, ('Predicted mesh (shape preview)',)), start=3):
            axis = fig.add_subplot(1, 3, index, projection='3d')
            triangles = points[faces]
            normals = np.cross(triangles[:, 1] - triangles[:, 0],
                               triangles[:, 2] - triangles[:, 0])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            light = np.array([1.0, -1.0, 2.0])
            light /= np.linalg.norm(light)
            brightness = 0.35 + 0.65 * np.abs(normals @ light)
            colors = brightness[:, None] * np.array([0.45, 0.65, 0.85])
            axis.add_collection3d(Poly3DCollection(
                triangles, facecolors=colors, edgecolors='none', linewidths=0,
                rasterized=True))
            axis.set_xlim(center[0] - radius, center[0] + radius)
            axis.set_ylim(center[1] - radius, center[1] + radius)
            axis.set_zlim(center[2] - radius, center[2] + radius)
            axis.set_box_aspect((1, 1, 1))
            axis.set_proj_type('ortho')
            axis.view_init(elev=args.elevation, azim=args.azimuth)
            axis.set_title(title)
            axis.set_axis_off()
        mode = 'prediction: stored coordinates' if args.preserve_coordinates else 'prediction: centered and scaled'
        preview = f' | preview grid {args.preview_grid}' if args.preview_grid else ''
        fig.suptitle(f'{image_path.name} | {mode}{preview}', fontsize=10)
        fig.tight_layout()
        fig.savefig(output_path, dpi=args.dpi, facecolor='white')
    finally:
        plt.close(fig)


def main():
    args = parse_args()
    for directory in (args.rgb_dir, args.original_dir, args.predicted_dir, args.metadata_dir):
        if not directory.is_dir():
            raise SystemExit(f'Directory does not exist: {directory}')
    images = sorted(path for path in args.rgb_dir.iterdir() if path.is_file()
                    and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'})
    if args.max_images is not None:
        images = images[:args.max_images]
    if not images:
        raise SystemExit(f'No images found in {args.rgb_dir}')
    if len({path.stem for path in images}) != len(images):
        raise SystemExit('Duplicate RGB filename stems would overwrite comparison files.')
    import matplotlib
    matplotlib.use('Agg')

    originals = mesh_index(args.original_dir)
    predictions = mesh_index(args.predicted_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    with (args.output_dir / 'comparison_report.csv').open('w', newline='') as report:
        writer = csv.writer(report)
        writer.writerow(['image', 'original_mesh', 'predicted_mesh', 'status', 'error'])
        for index, image in enumerate(images, 1):
            original, predicted = originals.get(image.stem), predictions.get(image.stem)
            output = args.output_dir / f'{image.stem}_comparison.png'
            try:
                if original is None or predicted is None:
                    raise FileNotFoundError(f'Missing matching mesh: original={original}, predicted={predicted}')
                status = 'skipped'
                if args.overwrite or not output.is_file():
                    save_comparison(image, original, predicted, output, args)
                    status = 'saved'
                writer.writerow([image.name, original, predicted, status, ''])
                print(f'[{index}/{len(images)}] {status}: {output.name}', flush=True)
            except Exception as error:
                failures += 1
                writer.writerow([image.name, original, predicted, 'error', str(error)])
                print(f'[{index}/{len(images)}] Failed {image.name}: {error}', flush=True)
            report.flush()
    print(f'Comparisons: {args.output_dir}; failures: {failures}')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
