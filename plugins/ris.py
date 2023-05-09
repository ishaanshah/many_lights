import mitsuba as mi
mi.set_variant("cuda_rgb", "llvm_rgb")
mi.set_log_level(mi.LogLevel.Info)

import drjit as dr
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from utils.render import render_manual
from utils.parser import add_common_args
from utils.reservoir import Reservoir

import ipdb

def direct_nee(si: mi.SurfaceInteraction3f, scene: mi.Scene) -> mi.Color3f:
    sensor: mi.Sensor = scene.sensors()[0]
    sampler: mi.Sampler = sensor.sampler()

    bsdf: mi.BSDF = si.bsdf()
    bsdf_flags: mi.BSDFFlags = bsdf.flags()
    ctx: mi.BSDFContext = mi.BSDFContext()

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

def restir(si: mi.SurfaceInteraction3f, scene: mi.Scene) -> mi.Color3f:
    sensor: mi.Sensor = scene.sensors()[0]
    sampler: mi.Sampler = sensor.sampler()

    res = mi.Color3f(0)

    bsdf: mi.BSDF = si.bsdf()
    bsdf_flags: mi.BSDFFlags = bsdf.flags()
    ctx: mi.BSDFContext = mi.BSDFContext()

    active = si.is_valid() & mi.has_flag(bsdf_flags, mi.BSDFFlags.Smooth)

    # Show emmiters
    res += si.emitter(scene, active).eval(si, active)

    m = 32
    i = mi.UInt(0)
    reservoir = Reservoir(mi.DirectionSample3f)
    # loop = mi.Loop("RIS", lambda: (i, reservoir.w_sum, reservoir.sample))
    # while loop(i < m):
    while i[0] < m:
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
    W = dr.select(p_hat > 1e-6, dr.rcp(p_hat) * reservoir.w_sum * dr.rcp(m), 0)

    res += bsdf_val * emmiter_val * W

    return res
    
parser = ArgumentParser(conflict_handler="resolve")
add_common_args(parser)
args = parser.parse_args()

scene_desc = mi.cornell_box()
scene: mi.Scene = mi.load_file("scenes/veach-mis/scene.xml")

res = render_manual(restir, scene, args.spp, True)
res = mi.Bitmap(res)
res.write("restir.exr")

res = render_manual(direct_nee, scene, args.spp, True)
res = mi.Bitmap(res)
res.write("direct.exr")