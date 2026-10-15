"""Generate one mesh per RGB image, loading the TRELLIS model only once.

Example:
    python example_batch.py
    python example_batch.py --glb --max-images 2
"""

import argparse
import csv
import os
from pathlib import Path
import time


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=root / 'data/test_split/rgb')
    parser.add_argument('--output-dir', type=Path, default=root / 'data/test_split/meshes')
    parser.add_argument('--model', default='microsoft/TRELLIS-image-large')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--max-images', type=int, help='Process only the first N images.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--glb', action='store_true', help='Also export textured GLB meshes.')
    parser.add_argument('--skip-preprocess', action='store_true',
                        help='Disable TRELLIS background removal and cropping.')
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        parser.error('--max-images must be positive')
    return args


def main():
    args = parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f'Input directory does not exist: {args.input_dir}')
    extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
    images = sorted(path for path in args.input_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in extensions)
    if args.max_images is not None:
        images = images[:args.max_images]
    if not images:
        raise SystemExit(f'No supported images found in {args.input_dir}')
    stems = [path.stem for path in images]
    if len(stems) != len(set(stems)):
        raise SystemExit('Input images have duplicate filename stems; rename them to avoid output collisions.')
    pending = [path for path in images if args.overwrite
               or not (args.output_dir / f'{path.stem}.ply').is_file()
               or (args.glb and not (args.output_dir / f'{path.stem}.glb').is_file())]
    if not pending:
        print('All requested meshes already exist. Use --overwrite to regenerate.')
        return

    # Set backend options before importing TRELLIS, as in example.py.
    os.environ.setdefault('SPCONV_ALGO', 'native')
    import torch
    import trimesh
    from PIL import Image
    from trellis.pipelines import TrellisImageTo3DPipeline

    if not torch.cuda.is_available():
        raise SystemExit('TRELLIS requires a CUDA GPU. Activate your TRELLIS environment with CUDA support.')
    if args.glb:
        from trellis.utils import postprocessing_utils

    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    timing_path = args.output_dir / 'timings.csv'
    with timing_path.open('a', newline='') as timing_file:
        writer = csv.writer(timing_file)
        if timing_file.tell() == 0:
            writer.writerow(['image', 'status', 'seconds', 'error'])
        for index, path in enumerate(pending, start=1):
            print(f'[{index}/{len(pending)}] {path.name}', flush=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = None
            try:
                with Image.open(path) as image, torch.inference_mode():
                    outputs = pipeline.run(
                        image, seed=args.seed,
                        formats=['mesh', 'gaussian'] if args.glb else ['mesh'],
                        preprocess_image=not args.skip_preprocess,
                    )
                mesh = outputs['mesh'][0]
                trimesh.Trimesh(
                    vertices=mesh.vertices.detach().cpu().numpy(),
                    faces=mesh.faces.detach().cpu().numpy(),
                    process=False,
                ).export(args.output_dir / f'{path.stem}.ply')
                if args.glb:
                    glb = postprocessing_utils.to_glb(
                        outputs['gaussian'][0], mesh,
                        simplify=0.95, texture_size=1024,
                    )
                    glb.export(args.output_dir / f'{path.stem}.glb')
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                writer.writerow([path.name, 'ok', f'{elapsed:.3f}', ''])
                print(f'  Saved mesh ({elapsed:.2f}s)', flush=True)
            except Exception as error:
                failures += 1
                writer.writerow([path.name, 'error', f'{time.perf_counter() - start:.3f}', str(error)])
                print(f'  Failed: {error}', flush=True)
            finally:
                outputs = None
                # Release per-image GPU tensors before starting the next image.
                if 'mesh' in locals():
                    del mesh
                torch.cuda.empty_cache()
                timing_file.flush()
    print(f'Saved {len(pending) - failures}/{len(pending)} meshes to {args.output_dir}')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
