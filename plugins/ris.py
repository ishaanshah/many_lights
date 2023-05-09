import mitsuba as mi
mi.set_variant("cuda_rgb", "llvm_rgb")
mi.set_log_level(mi.LogLevel.Info)

import drjit as dr
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from utils.render import render_manual
from utils.parser import add_common_args

import ipdb

def direct_nee(si: mi.SurfaceInteraction3f) -> mi.Color3f:
    bsdf: mi.BSDF = si.bsdf()
    bsdf_flags: mi.BSDFFlags = bsdf.flags()
    ctx: mi.BSDFContext = mi.BSDFContext()

    sensor: mi.Sensor = scene.sensors()[0]
    sampler: mi.Sampler = sensor.sampler()

    active = si.is_valid()

    res = mi.Color3f(0)

    # Show emmiters
    res += si.emitter(scene, active).eval(si, active)

    # NEE
    # Sample Light
    active_e = mi.has_flag(bsdf_flags, mi.BSDFFlags.Smooth)
    ds, emitter_val = scene.sample_emitter_direction(si, sampler.next_2d(active_e), active=active_e)
    active_e &= dr.neq(ds.pdf, 0.0)

    # Evaluate BSDF
    wo = si.to_local(ds.d)
    bsdf_val: mi.Color3f = bsdf.eval(ctx, si, wo, active_e)

    # Rendering equation
    res += dr.select(active_e, bsdf_val * emitter_val, 0)

    return res

parser = ArgumentParser(conflict_handler="resolve")
add_common_args(parser)
args = parser.parse_args()

scene_desc = mi.cornell_box()
scene: mi.Scene  = mi.load_dict(scene_desc)

res = render_manual(direct_nee, scene, args.spp, True)

res = mi.Bitmap(res)
res.write("out.exr")