#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/mesh.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _emitter-area:

Area light (:monosp:`area`)
---------------------------

.. pluginparameters::

 * - radiance
   - |spectrum| or |texture|
   - Specifies the emitted radiance in units of power per unit area per unit steradian.
   - |exposed|, |differentiable|

This plugin implements an area light, i.e. a light source that emits
diffuse illumination from the exterior of an arbitrary shape.
Since the emission profile of an area light is completely diffuse, it
has the same apparent brightness regardless of the observer's viewing
direction. Furthermore, since it occupies a nonzero amount of space, an
area light generally causes scene objects to cast soft shadows.

To create an area light source, simply instantiate the desired
emitter shape and specify an :monosp:`area` instance as its child:

.. tabs::
    .. code-tab:: xml
        :name: sphere-light

        <shape type="sphere">
            <emitter type="area">
                <rgb name="radiance" value="1.0"/>
            </emitter>
        </shape>

    .. code-tab:: python

        'type': 'sphere',
        'emitter': {
            'type': 'area',
            'radiance': {
                'type': 'rgb',
                'value': 1.0,
            }
        }

 */

template <typename Float, typename Spectrum>
class AreaLightLTC final : public Emitter<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Emitter, m_flags, m_shape, m_medium)
    MI_IMPORT_TYPES(Scene, Shape, ShapePtr, Mesh, Texture)

    AreaLightLTC(const Properties &props) : Base(props) {
        if (props.has_property("to_world"))
            Throw("Found a 'to_world' transformation -- this is not allowed. "
                  "The area light inherits this transformation from its parent "
                  "shape.");

        m_radiance = props.texture_d65<Texture>("radiance", 1.f);

        m_flags = +EmitterFlags::Ltc;
        if (m_radiance->is_spatially_varying())
            m_flags |= +EmitterFlags::SpatiallyVarying;
        dr::set_attr(this, "flags", m_flags);
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("radiance", m_radiance.get(), +ParamFlags::Differentiable);
    }

    Float integrate_edge(Vector3f v1, Vector3f v2) const {
        Float x = dr::dot(v1, v2);
        Float y = dr::abs(x);

        Float a = 0.8543985f + (0.4965155f + 0.0145206f * y) * y;
        Float b = 3.4175940f + (4.1616724f + y) * y;
        Float v = a / b;

        Mask cond = x > Float(0);
        Float theta_sintheta = dr::select(cond, v, 0.5 * (dr::rsqrt(dr::maximum(1.0f - x * x, 1e-7))) - v);

        return (cross(v1, v2) * theta_sintheta)[2];
    }

    Mask is_below_horizon(Vector3f v1) const {
        return v1[2] < 0.f;
    }

    Vector3f intersect_horizon(Vector3f v1, Vector3f v2) const {
        Float t = v1[2] / (v2[2] - v1[2]);
        return dr::normalize(v1 - (v2 - v1) * t);
    }

    /* Integrate the polygon while taking care of clipping vertices */
    Spectrum integrate_triangle(Vector3f l1, Vector3f l2, Vector3f l3) const {
        Spectrum result(0.f);

        // Common Vars
        Vector3f i1, i2;
        Mask cond;

        // Which are below horizon?
        Mask l1_below = is_below_horizon(l1);
        Mask l2_below = is_below_horizon(l2);
        Mask l3_below = is_below_horizon(l3);

        // l1 above
        // i1 = intersect_horizon(l1, l2);
        // i2 = intersect_horizon(l1, l3);
        // cond = !l1_below && l2_below && l3_below;
        
        // result += dr::select(cond, integrate_edge(l1, i1), 0.f);
        // result += dr::select(cond, integrate_edge(i1, i2), 0.f);
        // result += dr::select(cond, integrate_edge(i2, l1), 0.f);
        
        // result = dr::select(cond, dr::abs(result), result);
        
        // // l2 above
        // i1 = intersect_horizon(l2, l3);
        // i2 = intersect_horizon(l2, l1);
        // cond = l1_below && !l2_below && l3_below;
        
        // result += dr::select(cond, integrate_edge(l2, i1), 0.f);
        // result += dr::select(cond, integrate_edge(i1, i2), 0.f);
        // result += dr::select(cond, integrate_edge(i2, l2), 0.f);
        
        // result = dr::select(cond, dr::abs(result), result);
        
        // // l3 above
        // i1 = intersect_horizon(l3, l1);
        // i2 = intersect_horizon(l3, l2);
        // cond = l1_below && l2_below && !l3_below;
        
        // result += dr::select(cond, integrate_edge(l3, i1), 0.f);
        // result += dr::select(cond, integrate_edge(i1, i2), 0.f);
        // result += dr::select(cond, integrate_edge(i2, l3), 0.f);
        
        // result = dr::select(cond, dr::abs(result), result);
        
        // // l1, l2 above
        // i1 = intersect_horizon(l1, l3);
        // i2 = intersect_horizon(l2, l3);
        // cond = !l1_below && !l2_below && l3_below;
        
        // result += dr::select(cond, integrate_edge(l1, i1), 0.f);
        // result += dr::select(cond, integrate_edge(i1, i2), 0.f);
        // result += dr::select(cond, integrate_edge(i2, l2), 0.f);
        // result += dr::select(cond, integrate_edge(l2, l1), 0.f);
        
        // result = dr::select(cond, dr::abs(result), result);
        
        // // l1, l3 above
        // i1 = intersect_horizon(l1, l2);
        // i2 = intersect_horizon(l3, l2);
        // cond = !l1_below && l2_below && !l3_below;
        
        // result += dr::select(cond, integrate_edge(l1, i1), 0.f);
        // result += dr::select(cond, integrate_edge(i1, i2), 0.f);
        // result += dr::select(cond, integrate_edge(i2, l3), 0.f);
        // result += dr::select(cond, integrate_edge(l3, l1), 0.f);
        
        // result = dr::select(cond, dr::abs(result), result);
        
        // // l2, l3 above
        // i1 = intersect_horizon(l2, l1);
        // i2 = intersect_horizon(l3, l1);
        // cond = l1_below && !l2_below && !l3_below;
        
        // result += dr::select(cond, integrate_edge(l2, i1), 0.f);
        // result += dr::select(cond, integrate_edge(i1, i2), 0.f);
        // result += dr::select(cond, integrate_edge(i2, l3), 0.f);
        // result += dr::select(cond, integrate_edge(l3, l2), 0.f);
        
        // result = dr::select(cond, dr::abs(result), result);
        
        // All above
        cond = !l1_below && !l2_below && !l3_below;

        result += dr::select(cond, integrate_edge(l1, l2), 0.f);
        result += dr::select(cond, integrate_edge(l2, l3), 0.f);
        result += dr::select(cond, integrate_edge(l3, l1), 0.f);
        
        result = dr::select(cond, dr::abs(result), result);

        return result;
    }

    Spectrum eval(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

        Spectrum result(0.f);
        
        const Shape* shape = dynamic_cast<const Shape*>(this->shape());
        const Mesh* mesh = dynamic_cast<const Mesh*>(shape);

        Vector3f v1, v2, v3;
        v1 = mesh->vertex_position(UInt32(0));
        v2 = mesh->vertex_position(UInt32(1));
        v3 = mesh->vertex_position(UInt32(2));

        // Translate with shading point as origin, and normalize
        v1 = dr::normalize(v1 - si.p);
        v2 = dr::normalize(v2 - si.p);
        v3 = dr::normalize(v3 - si.p);
        
        // Convert to local shading frame
        v1 = si.to_local(v1);
        v2 = si.to_local(v2);
        v3 = si.to_local(v3);

        // Multiply by coord frame matrix
        Vector3f v1_(dr::dot(si.coord_r1, v1),
                     dr::dot(si.coord_r2, v1),
                     dr::dot(si.coord_r3, v1));
        v1_ = dr::normalize(v1_);

        Vector3f v2_(dr::dot(si.coord_r1, v2),
                     dr::dot(si.coord_r2, v2),
                     dr::dot(si.coord_r3, v2));
        v2_ = dr::normalize(v2_);

        Vector3f v3_(dr::dot(si.coord_r1, v3),
                     dr::dot(si.coord_r2, v3),
                     dr::dot(si.coord_r3, v3));
        v3_ = dr::normalize(v3_);

        // Multiply by ltc matrix
        Vector3f l1(dr::dot(si.ltc_inv_r1, v1_),
                    dr::dot(si.ltc_inv_r2, v1_),
                    dr::dot(si.ltc_inv_r3, v1_));
        l1 = dr::normalize(l1);

        Vector3f l2(dr::dot(si.ltc_inv_r1, v2_),
                    dr::dot(si.ltc_inv_r2, v2_),
                    dr::dot(si.ltc_inv_r3, v2_));
        l2 = dr::normalize(l2);

        Vector3f l3(dr::dot(si.ltc_inv_r1, v3_),
                    dr::dot(si.ltc_inv_r2, v3_),
                    dr::dot(si.ltc_inv_r3, v3_));
        l3 = dr::normalize(l3);

        result += integrate_triangle(l1, l2, l3);

        return result;
    }

    ScalarBoundingBox3f bbox() const override { return m_shape->bbox(); }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "AreaLightLTC[" << std::endl
            << "  radiance = " << string::indent(m_radiance) << "," << std::endl
            << "  surface_area = ";
        if (m_shape) oss << m_shape->surface_area();
        else         oss << "  <no shape attached!>";
        oss << "," << std::endl;
        if (m_medium) oss << string::indent(m_medium);
        else         oss << "  <no medium attached!>";
        oss << std::endl << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    ref<Texture> m_radiance;
};

MI_IMPLEMENT_CLASS_VARIANT(AreaLightLTC, Emitter)
MI_EXPORT_PLUGIN(AreaLightLTC, "Area emitter")
NAMESPACE_END(mitsuba)
