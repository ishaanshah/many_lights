import mitsuba as mi
import drjit as dr

def pol2cart(pol: mi.Point2f) -> mi.Vector3f:
    """
    Convert from polar to cartesian
    """
    theta = pol.x
    phi = pol.y

    sin_theta, cos_theta = dr.sincos(theta)
    sin_phi, cos_phi = dr.sincos(phi)
    return mi.Vector3f(
        sin_theta*cos_phi,
        sin_theta*sin_phi,
        cos_theta
    )

def cart2pol(cart: mi.Vector3f, clip_upper: bool=False) -> mi.Point2f:
    """
    Convert from cartesian to polar
    """
    theta = dr.clip(dr.acos(cart.z), 0 if clip_upper else -dr.pi / 2, dr.pi / 2)
    phi = dr.clip(dr.atan2(cart.y, cart.x), -dr.pi, dr.pi)

    return mi.Point2f(theta, phi)