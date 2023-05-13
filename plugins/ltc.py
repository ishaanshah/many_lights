from typing import Tuple
import mitsuba as mi
mi.set_variant("cuda_rgb", "llvm_rgb")
mi.set_log_level(mi.LogLevel.Info)

import drjit as dr
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from utils.render import render_multi_pass, linear_to_srgb
from utils.parser import add_common_args

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

        # Hack to get roughness values of the surface.
        # We can get rid of this after https://github.com/mitsuba-renderer/mitsuba3/pull/589
        # or https://github.com/mitsuba-renderer/mitsuba3/pull/589 get merged
        bsdf: mi.BSDF = si.bsdf()
        ctx: mi.BSDFContext = mi.BSDFContext()
        bs, _ = bsdf.sample(ctx, si, sampler.next_1d(active), sampler.next_2d(active), active)

        result = mi.Color3f(0)

        # Show emmiters
        if not self.hide_emiters:
            result += si.emitter(scene, active).eval(si, active)

        # Construct local coordinate frame
        wi_local = si.to_local(si.wi)
        c1 = dr.normalize(mi.Vector3f(wi_local.x, wi_local.y, 0))
        c3 = mi.Vector3f(0, 0, 1)
        c2 = dr.normalize(dr.cross(c3, c1))
        si.coord_r1 = c1
        si.coord_r2 = c2
        si.coord_r3 = c3

        # Fetch LTC matrix
        si_dummy: mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)
        si_dummy.uv = mi.Point2f(
            dr.acos(wi_local.z) * 2 * dr.inv_pi,    # Incident direction
            dr.clamp(bs.param1, 0.01, 0.99)         # Roughness
        )

        r1 = self.ltc_1.eval(si_dummy, active)
        r2 = self.ltc_2.eval(si_dummy, active)
        r3 = self.ltc_3.eval(si_dummy, active)

        # Matrix Constructor expects columns as inputs,
        # transpose to make columns rows
        ltc_mat = dr.transpose(mi.Matrix3f(r1, r2, r3))
        ltc_mat_inv = dr.inverse(ltc_mat)

        si.ltc_r1 = r1
        si.ltc_r2 = r2
        si.ltc_r3 = r3

        si.ltc_inv_r1 = mi.Vector3f(ltc_mat_inv[0][0], ltc_mat_inv[1][0], ltc_mat_inv[2][0])
        si.ltc_inv_r2 = mi.Vector3f(ltc_mat_inv[0][1], ltc_mat_inv[1][1], ltc_mat_inv[2][1])
        si.ltc_inv_r3 = mi.Vector3f(ltc_mat_inv[0][2], ltc_mat_inv[1][2], ltc_mat_inv[2][2])

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

    scene_desc = mi.cornell_box()
    scene: mi.Scene = mi.load_file("scenes/matpreview_area_light/scene_ltc_surface.xml")
    # scene: mi.Scene = mi.load_dict(scene_desc)

    # Define integrators
    integrators = {
        "ltc": None,
        # "mis": mi.load_dict({ "type": "direct" }),
    }

    for int_name, integrator in integrators.items():
        render_func = lambda scene, seed, spp: mi.render(scene, spp=spp, integrator=integrator, seed=seed)
        res = render_multi_pass(render_func, args.resolution, args.resolution, scene, args.spp, f"{int_name}.exr")
        if args.show_render:
            plt.imshow(linear_to_srgb(res))
            plt.show()