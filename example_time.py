import os
import time
import torch
import imageio

from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


os.environ["SPCONV_ALGO"] = "native"


def start_timer():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def stop_timer(start_time, name):
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time
    print(f"[TIME] {name}: {elapsed:.4f} s")
    return elapsed


timings = {}


# ---------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------
t = start_timer()

pipeline = TrellisImageTo3DPipeline.from_pretrained(
    "microsoft/TRELLIS-image-large"
)

timings["model_loading"] = stop_timer(t, "Model loading")


# ---------------------------------------------------------
# 2. Move model to GPU
# ---------------------------------------------------------
t = start_timer()

pipeline.cuda()

timings["model_to_gpu"] = stop_timer(t, "Move model to GPU")


# ---------------------------------------------------------
# 3. Load input image
# ---------------------------------------------------------
t = start_timer()

image = Image.open("assets/example_image/sphere_0.png")

timings["image_loading"] = stop_timer(t, "Image loading")


# ---------------------------------------------------------
# 4. TRELLIS inference / 3D generation
# ---------------------------------------------------------
t = start_timer()

outputs = pipeline.run(
    image,
    seed=1,
)

timings["pipeline_generation"] = stop_timer(
    t,
    "TRELLIS pipeline generation"
)


# ---------------------------------------------------------
# 5. Gaussian rendering
# ---------------------------------------------------------
t = start_timer()

video = render_utils.render_video(
    outputs["gaussian"][0]
)["color"]

timings["gaussian_render"] = stop_timer(
    t,
    "Gaussian rendering"
)


# save separately so rendering and disk writing are not mixed
t = start_timer()

imageio.mimsave(
    "experiments_HPC/sphere_0_gs.mp4",
    video,
    fps=30
)

timings["gaussian_video_save"] = stop_timer(
    t,
    "Gaussian video saving"
)


# ---------------------------------------------------------
# 6. Radiance field rendering
# ---------------------------------------------------------
t = start_timer()

video = render_utils.render_video(
    outputs["radiance_field"][0]
)["color"]

timings["radiance_field_render"] = stop_timer(
    t,
    "Radiance-field rendering"
)


t = start_timer()

imageio.mimsave(
    "experiments_HPC/sphere_0_rf.mp4",
    video,
    fps=30
)

timings["radiance_field_video_save"] = stop_timer(
    t,
    "Radiance-field video saving"
)


# ---------------------------------------------------------
# 7. Mesh rendering
# ---------------------------------------------------------
t = start_timer()

video = render_utils.render_video(
    outputs["mesh"][0]
)["normal"]

timings["mesh_render"] = stop_timer(
    t,
    "Mesh rendering"
)


t = start_timer()

imageio.mimsave(
    "experiments_HPC/sphere_0_mesh.mp4",
    video,
    fps=30
)

timings["mesh_video_save"] = stop_timer(
    t,
    "Mesh video saving"
)


# ---------------------------------------------------------
# 8. GLB generation / mesh post-processing
# ---------------------------------------------------------
t = start_timer()

glb = postprocessing_utils.to_glb(
    outputs["gaussian"][0],
    outputs["mesh"][0],
    simplify=0.95,
    texture_size=1024,
)

timings["glb_generation"] = stop_timer(
    t,
    "GLB generation / mesh post-processing"
)


# ---------------------------------------------------------
# 9. GLB export
# ---------------------------------------------------------
t = start_timer()

glb.export("experiments_HPC/sphere_0.glb")

timings["glb_export"] = stop_timer(
    t,
    "GLB export"
)


# ---------------------------------------------------------
# 10. Save Gaussian PLY
# ---------------------------------------------------------
t = start_timer()

outputs["gaussian"][0].save_ply(
    "experiments_HPC/sphere_0.ply"
)

timings["ply_save"] = stop_timer(
    t,
    "PLY saving"
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
print("\n")
print("=" * 65)
print("TRELLIS TIMING SUMMARY")
print("=" * 65)

for name, elapsed in timings.items():
    print(f"{name:35s}: {elapsed:8.4f} s")

print("=" * 65)

generation_and_rendering = (
    timings["pipeline_generation"]
    + timings["gaussian_render"]
    + timings["radiance_field_render"]
    + timings["mesh_render"]
)

print(
    f"{'Generation + rendering':35s}: "
    f"{generation_and_rendering:8.4f} s"
)