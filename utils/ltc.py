import mitsuba as mi
import drjit as dr

def si_ltc(
    si: mi.SurfaceInteraction3f,
    sampler: mi.Sampler,
    ltc_1: mi.Texture,
    ltc_2: mi.Texture,
    ltc_3: mi.Texture,
    active: bool=True
):
    """
    Instantiate the interaction with parameters needed for LTC evaluation
    """
    # Hack to get roughness values of the surface.
    # We can get rid of this after https://github.com/mitsuba-renderer/mitsuba3/pull/589
    # or https://github.com/mitsuba-renderer/mitsuba3/pull/589 get merged
    bsdf: mi.BSDF = si.bsdf()
    ctx: mi.BSDFContext = mi.BSDFContext()
    bs, _ = bsdf.sample(ctx, si, sampler.next_1d(active), sampler.next_2d(active), active)

    # Construct local coordinate frame
    c1 = dr.normalize(mi.Vector3f(si.wi.x, si.wi.y, 0))
    c3 = mi.Vector3f(0, 0, 1)
    c2 = dr.normalize(dr.cross(c3, c1))
    si.coord_r1 = c1
    si.coord_r2 = c2
    si.coord_r3 = c3

    # Fetch LTC matrix
    si_dummy: mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)
    si_dummy.uv = mi.Point2f(
        0.99 * dr.acos(si.wi.z) * 2 * dr.inv_pi,    # Incident direction
        dr.clamp(bs.param1, 0.01, 0.99)         # Roughness
    )

    r1 = ltc_1.eval(si_dummy, active)
    r2 = ltc_2.eval(si_dummy, active)
    r3 = ltc_3.eval(si_dummy, active)

    # Matrix Constructor expects columns as inputs,
    # transpose to make columns rows
    ltc_mat = mi.Matrix3f(r1, r2, r3)
    ltc_mat_inv = dr.inverse(ltc_mat)

    si.ltc_r1 = r1
    si.ltc_r2 = r2
    si.ltc_r3 = r3

    si.ltc_inv_r1 = mi.Vector3f(ltc_mat_inv[0][0], ltc_mat_inv[0][1], ltc_mat_inv[0][2])
    si.ltc_inv_r2 = mi.Vector3f(ltc_mat_inv[1][0], ltc_mat_inv[1][1], ltc_mat_inv[1][2])
    si.ltc_inv_r3 = mi.Vector3f(ltc_mat_inv[2][0], ltc_mat_inv[2][1], ltc_mat_inv[2][2])

    # DEBUG: Only diffuse
    # si.ltc_inv_r1 = mi.Vector3f(1, 0, 0)
    # si.ltc_inv_r2 = mi.Vector3f(0, 1, 0)
    # si.ltc_inv_r3 = mi.Vector3f(0, 0, 1)