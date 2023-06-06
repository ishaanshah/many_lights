import mitsuba as mi
mi.set_variant("cuda_rgb")
mi.set_log_level(mi.LogLevel.Info)

# Register all plugins
import os
import plugins as _
import matplotlib.pyplot as plt
from utils.render import render_multi_pass, linear_to_srgb
from utils.parser import add_common_args
from utils.fs import create_dir
from argparse import ArgumentParser

parser = ArgumentParser(conflict_handler="resolve")
add_common_args(parser)
parser.add_argument("--ltc_dir", default="data")
args = parser.parse_args()

scene_desc = mi.cornell_box()
# scene: mi.Scene = mi.load_dict(scene_desc)

out_path = os.path.join("renders", "compare")
create_dir(out_path, del_old=True)

ltc_textures = {
    "ltc_1": {
        "type": "bitmap",
        "filename": os.path.join(args.ltc_dir, "isotropic_ggx_1.exr"),
        "raw": True,
        "wrap_mode": "clamp"
    },
    "ltc_2": {
        "type": "bitmap",
        "filename": os.path.join(args.ltc_dir, "isotropic_ggx_2.exr"),
        "raw": True,
        "wrap_mode": "clamp"
    },
    "ltc_3": {
        "type": "bitmap",
        "filename": os.path.join(args.ltc_dir, "isotropic_ggx_3.exr"),
        "raw": True,
        "wrap_mode": "clamp"
    }
}

# Define integrators
integrators = {
    "ris": mi.load_dict({ "type": "ris", "num_proposals": 32 }),
    "mis": mi.load_dict({ "type": "direct" }),
    "ltc": mi.load_dict({"type": "ltc", **ltc_textures }),
    "ltc_mc": mi.load_dict({"type": "ltc_mc", **ltc_textures}),
    "ltc_ris": mi.load_dict({"type": "ltc_ris", "num_proposals": 32, "num_pdf_samples": 4, **ltc_textures}),
    "emitter": mi.load_dict({ "type": "direct", "bsdf_samples": 0 }),
    "bsdf": mi.load_dict({ "type": "direct", "emitter_samples": 0 })
}

for int_name, integrator in integrators.items():
    emitter_type = "ltc_area" if "ltc" in int_name else "area"
    scene: mi.Scene = mi.load_file("scenes/veach-mis/scene_rectangle.xml", emitter_type=emitter_type)

    render_func = lambda scene, seed, spp: mi.render(scene, spp=spp, integrator=integrator, seed=seed)
    res = render_multi_pass(render_func, args.resolution, args.resolution, scene, args.spp, os.path.join(out_path, f"{int_name}.png"))
    if args.show_render:
        plt.imshow(linear_to_srgb(res))
        plt.axis("off")
        plt.title(int_name)
        plt.show()