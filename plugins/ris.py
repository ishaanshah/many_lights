from typing import Tuple
import mitsuba as mi
mi.set_variant("cuda_rgb", "llvm_rgb")
mi.set_log_level(mi.LogLevel.Info)

import drjit as dr
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from utils.render import render_multi_pass, linear_to_srgb
from utils.parser import add_common_args
from utils.reservoir import Reservoir

import ipdb

class ReSTIR(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties) -> None:
        super().__init__(props)

        self.m = props.get("num_proposals", 32)
    
    def sample(self, scene: mi.Scene, sampler, ray: mi.RayDifferential3f, medium: mi.Medium = None, active: bool = True) -> Tuple[mi.Color3f, bool]:
        si = scene.ray_intersect(ray, active)

        bsdf: mi.BSDF = si.bsdf()
        bsdf_flags: mi.BSDFFlags = bsdf.flags()
        ctx: mi.BSDFContext = mi.BSDFContext()

        active = si.is_valid() & mi.has_flag(bsdf_flags, mi.BSDFFlags.Smooth)

        result = mi.Color3f(0)

        # Show emmiters
        result += si.emitter(scene, active).eval(si, active)

        i = mi.UInt(0)
        reservoir = Reservoir(mi.DirectionSample3f)
        loop = mi.Loop("RIS", lambda: (i, reservoir.sample, reservoir.w_sum, sampler))
        while loop(i < self.m):
            ds, emitter_val = scene.sample_emitter_direction(si, sampler.next_2d(active), active=active, test_visibility=False)

            # Evaluate BSDF
            wo = si.to_local(ds.d)
            bsdf_val: mi.Color3f = bsdf.eval(ctx, si, wo, active)

            # Calculate RIS weight
            p_hat = dr.norm(bsdf_val * emitter_val)
            w = p_hat / dr.maximum(1e-6, ds.pdf)

            reservoir.update(w, ds, sampler.next_1d(active), active)

            i += 1

        active = ~scene.ray_test(si.spawn_ray_to(reservoir.sample.p), active)
        emmiter_val = scene.eval_emitter_direction(si, reservoir.sample, active)
        wo = si.to_local(reservoir.sample.d)
        bsdf_val = bsdf.eval(ctx, si, wo)

        p_hat = dr.norm(bsdf_val * emitter_val)
        W = dr.select(p_hat > 1e-6, dr.rcp(p_hat) * reservoir.w_sum * dr.rcp(self.m), 0)

        result += bsdf_val * emmiter_val * W

        return result, active, []


mi.register_integrator("restir", lambda props: ReSTIR(props))

parser = ArgumentParser(conflict_handler="resolve")
add_common_args(parser)
args = parser.parse_args()

scene_desc = mi.cornell_box()
scene: mi.Scene = mi.load_file("scenes/veach-mis/scene.xml")

# Define integrators
integrators = {
    "restir": mi.load_dict({ "type": "restir" }),
    "mis": mi.load_dict({ "type": "direct" }),
    "emitter": mi.load_dict({ "type": "direct", "bsdf_samples": 0 }),
    "bsdf": mi.load_dict({ "type": "direct", "emitter_samples": 0 })
}

for int_name, integrator in integrators.items():
    render_func = lambda scene, seed, spp: mi.render(scene, spp=spp, integrator=integrator, seed=seed)
    res = render_multi_pass(render_func, args.resolution, args.resolution, scene, args.spp, f"{int_name}.exr")
    if args.show_render:
        plt.imshow(linear_to_srgb(res))
        plt.show()