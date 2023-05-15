from typing import Tuple
import mitsuba as mi
mi.set_variant("cuda_rgb", "llvm_rgb")
mi.set_log_level(mi.LogLevel.Info)

import drjit as dr
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser
from utils.render import render_multi_pass, linear_to_srgb
from utils.parser import add_common_args
from utils.fs import create_dir
from utils.ltc import si_ltc

import ipdb

class LTCIntegrator(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties) -> None:
        super().__init__(props)

        self.hide_emiters = props.get("hide_emitters", False)

        self.ltc_1: mi.Texture = props["ltc_1"]
        self.ltc_2: mi.Texture = props["ltc_2"]
        self.ltc_3: mi.Texture = props["ltc_3"]
    
    def sample(self, scene: mi.Scene, sampler: mi.Sampler, ray: mi.RayDifferential3f, medium: mi.Medium = None, active: bool = True) -> Tuple[mi.Color3f, bool]:
        si: mi.SurfaceInteraction3f = scene.ray_intersect(ray, active)

        active &= si.is_valid()

        result = mi.Color3f(0)

        # Instantiate si fields with stuff needed for LTC integration
        si_ltc(si, sampler, self.ltc_1, self.ltc_2, self.ltc_3, active)

        # Get emitters in the scene
        num_emitters = len(scene.emitters_dr())
        emitters = scene.emitters_dr()

        i = mi.UInt(0)
        loop = mi.Loop("LTC Eval", lambda: (i, result))
        while loop(i < num_emitters):
        # while i[0] < num_emitters:
            emitter: mi.EmitterPtr = dr.gather(mi.EmitterPtr, emitters, i, active)

            # Only loop over lights which are of type LTC
            is_ltc_light = mi.has_flag(emitter.flags(), mi.EmitterFlags.Ltc)

            # Evaluate LTC
            result += emitter.eval(si, active=(active & is_ltc_light))

            i += 1

        return result, active, []


if __name__ == "__main__":
    mi.register_integrator("ltc", lambda props: LTCIntegrator(props))

    parser = ArgumentParser(conflict_handler="resolve")
    add_common_args(parser)
    args = parser.parse_args()

    # Define integrators
    integrators = {
        "ltc": None,
        "gt": mi.load_dict({ "type": "direct" }),
    }

    out_path = os.path.join("renders", "ltc")
    create_dir(out_path)

    for int_name, integrator in integrators.items():
        emitter_type = "ltc_area" if "ltc" in int_name else "area"
        scene: mi.Scene = mi.load_file("scenes/matpreview_area_light/scene_ltc_surface.xml", emitter_type=emitter_type)

        render_func = lambda scene, seed, spp: mi.render(scene, spp=spp, integrator=integrator, seed=seed)
        res = render_multi_pass(render_func, args.resolution, args.resolution, scene, args.spp, os.path.join(out_path, f"{int_name}.exr"))
        if args.show_render:
            plt.imshow(linear_to_srgb(res))
            plt.show()