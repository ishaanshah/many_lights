#include <mitsuba/render/integrator.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/core/properties.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _integrator-direct:

Direct illumination integrator (:monosp:`direct`)
-------------------------------------------------

.. pluginparameters::

 * - shading_samples
   - |int|
   - This convenience parameter can be used to set both :code:`emitter_samples` and
     :code:`bsdf_samples` at the same time.

 * - emitter_samples
   - |int|
   - Optional more fine-grained parameter: specifies the number of samples that should be generated
     using the direct illumination strategies implemented by the scene's emitters.
     (Default: set to the value of :monosp:`shading_samples`)

 * - bsdf_samples
   - |int|
   - Optional more fine-grained parameter: specifies the number of samples that should be generated
     using the BSDF sampling strategies implemented by the scene's surfaces.
     (Default: set to the value of :monosp:`shading_samples`)

 * - hide_emitters
   - |bool|
   - Hide directly visible emitters.
     (Default: no, i.e. |false|)

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/integrator_direct_bsdf.jpg
   :caption: (**a**) BSDF sampling only
   :label: fig-direct-bsdf
.. subfigure:: ../../resources/data/docs/images/render/integrator_direct_lum.jpg
   :caption: (**b**) Emitter sampling only
   :label: fig-direct-lum
.. subfigure:: ../../resources/data/docs/images/render/integrator_direct_both.jpg
   :caption: (**c**) MIS between both sampling strategies
   :label: fig-direct-both
.. subfigend::
   :width: 0.32
   :label: fig-direct

This integrator implements a direct illumination technique that makes use
of *multiple importance sampling*: for each pixel sample, the
integrator generates a user-specifiable number of BSDF and emitter
samples and combines them using the power heuristic. Usually, the BSDF
sampling technique works very well on glossy objects but does badly
everywhere else (**a**), while the opposite is true for the emitter sampling
technique (**b**). By combining these approaches, one can obtain a rendering
technique that works well in both cases (**c**).

The number of samples spent on either technique is configurable, hence
it is also possible to turn this plugin into an emitter sampling-only
or BSDF sampling-only integrator.

.. note:: This integrator does not handle participating media or indirect illumination.

.. tabs::
    .. code-tab::  xml
        :name: direct-integrator

        <integrator type="direct"/>

    .. code-tab:: python

        'type': 'direct'

 */

template <typename Float, typename Spectrum>
class LTCSurfaceIntegrator : public SamplingIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(SamplingIntegrator, m_hide_emitters)
    MI_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, 
        EmitterPtr, BSDF, BSDFPtr, Texture, ShapePtr)

    LTCSurfaceIntegrator(const Properties &props) : Base(props) {
        if (props.has_property("shading_samples")
            && (props.has_property("emitter_samples") ||
                props.has_property("bsdf_samples"))) {
            Throw("Cannot specify both 'shading_samples' and"
                  " ('emitter_samples' and/or 'bsdf_samples').");
        }

        /// Number of shading samples -- this parameter is a shorthand notation
        /// to set both 'emitter_samples' and 'bsdf_samples' at the same time
        size_t shading_samples = props.get<size_t>("shading_samples", 1);

        /// Number of samples to take using the emitter sampling technique
        m_emitter_samples = props.get<size_t>("emitter_samples", shading_samples);

        /// Number of samples to take using the BSDF sampling technique
        m_bsdf_samples = props.get<size_t>("bsdf_samples", shading_samples);

        if (m_emitter_samples + m_bsdf_samples == 0)
            Throw("Must have at least 1 BSDF or emitter sample!");

        size_t sum    = m_emitter_samples + m_bsdf_samples;
        m_weight_bsdf = 1.f / (ScalarFloat) m_bsdf_samples;
        m_weight_lum  = 1.f / (ScalarFloat) m_emitter_samples;
        m_frac_bsdf   = m_bsdf_samples / (ScalarFloat) sum;
        m_frac_lum    = m_emitter_samples / (ScalarFloat) sum;

        // LTC tables
        if (!props.has_property("ltc_1") || !props.has_property("ltc_2") || !props.has_property("ltc_3")) {
            Throw("LTC table not referenced! Please add 'ltc_1', 'ltc_2' & 'ltc_3' as textures.");
        }

        ltc_1 = props.texture<Texture>("ltc_1", 1.f);
        ltc_2 = props.texture<Texture>("ltc_2", 1.f);
        ltc_3 = props.texture<Texture>("ltc_3", 1.f);
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray,
                                     const Medium * /* medium */,
                                     Float * /* aovs */,
                                     Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        SurfaceInteraction3f si = scene->ray_intersect(
            ray, +RayFlags::All, /* coherent = */ true, active);
        Mask valid_ray = active && si.is_valid();

        Spectrum result(0.f);

        // ----------------------- Visible emitters -----------------------

        if (!m_hide_emitters) {
            EmitterPtr emitter_vis = si.emitter(scene, active);
            if (dr::any_or<true>(dr::neq(emitter_vis, nullptr)))
                result += emitter_vis->eval(si, active);
        }

        active &= si.is_valid();
        if (dr::none_or<false>(active))
            return { result, valid_ray };

        // Get the BSDF hyperparams (alphax, alphay etc.) in 'bs'
        BSDFContext ctx;
        BSDFPtr bsdf = si.bsdf(ray);
        auto [bs, bsdf_val] = bsdf->sample(ctx, si, sampler->next_1d(active),
            sampler->next_2d(active), active);

        // Construct local coord frame
        Vector3f wi_local = si.to_local(si.wi);
        Vector3f c1 = dr::normalize(Vector3f(wi_local[0], wi_local[1], Float(0.f)));
        Vector3f c3 = Vector3f(0.f, 0.f, 1.f);
        Vector3f c2 = dr::normalize(dr::cross(c3, c1));
        si.coord_r1 = Vector3f(c1[0], c1[1], c1[2]);
        si.coord_r2 = Vector3f(c2[0], c2[1], c2[2]);
        si.coord_r3 = Vector3f(c3[0], c3[1], c3[2]);
        
        // Fetch LTC matrix
        SurfaceInteraction3f si_dummy = dr::zeros<SurfaceInteraction3f>();
        si_dummy.uv = Point2f(dr::acos(wi_local[2]) * 2.f * dr::InvPi<Float>,
            dr::clamp(bs.param1, Float(0.01f), Float(0.99f)));\

        Vector3f r1 = (Vector3f) this->ltc_1->eval_3(si_dummy, active);
        Vector3f r2 = (Vector3f) this->ltc_2->eval_3(si_dummy, active);
        Vector3f r3 = (Vector3f) this->ltc_3->eval_3(si_dummy, active);
        dr::Matrix<Float, 3> ltc_mat(r1[0], r1[1], r1[2],
            r2[0], r2[1], r2[2],
            r3[0], r3[1], r3[2]);
        dr::Matrix<Float, 3> ltc_mat_inv = inverse(ltc_mat);

        si.ltc_r1 = r1;
        si.ltc_r2 = r2;
        si.ltc_r3 = r3;

        si.ltc_inv_r1 = Vector3f(ltc_mat_inv(0, 0), ltc_mat_inv(1, 0), ltc_mat_inv(2, 0));
        si.ltc_inv_r2 = Vector3f(ltc_mat_inv(0, 1), ltc_mat_inv(1, 1), ltc_mat_inv(2, 1));
        si.ltc_inv_r3 = Vector3f(ltc_mat_inv(0, 2), ltc_mat_inv(1, 2), ltc_mat_inv(2, 2));

        // si.ltc_inv_r1 = Vector3f(ltc_mat_inv(0, 0), ltc_mat_inv(0, 1), ltc_mat_inv(0, 2));
        // si.ltc_inv_r2 = Vector3f(ltc_mat_inv(1, 0), ltc_mat_inv(1, 1), ltc_mat_inv(1, 2));
        // si.ltc_inv_r3 = Vector3f(ltc_mat_inv(2, 0), ltc_mat_inv(2, 1), ltc_mat_inv(2, 2));
        
        // ----------------------- LTC Integration -----------------------
        uint32_t emitter_count = (uint32_t)scene->emitters().size();
        
        DynamicBuffer<EmitterPtr> emitter_dr = scene->emitters_dr();
        for(uint32_t i=0; i<emitter_count; i++) {
            EmitterPtr emitter = dr::gather<EmitterPtr>(emitter_dr, UInt32(i), active);
            
            Mask is_ltc_light = has_flag(emitter->flags(), EmitterFlags::Ltc);
            if (dr::any_or<true>(is_ltc_light)) {
                result += emitter->eval(si, active);
            }
        }

        return { result, valid_ray };
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "LTCSurfaceIntegrator[" << std::endl
            << "  emitter_samples = " << m_emitter_samples << "," << std::endl
            << "  bsdf_samples = " << m_bsdf_samples << std::endl
            << "]";
        return oss.str();
    }

    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        Float w = pdf_a / (pdf_a + pdf_b);
        return dr::select(dr::isfinite(w), w, 0.f);
    }

    MI_DECLARE_CLASS()
private:
    size_t m_emitter_samples;
    size_t m_bsdf_samples;
    ScalarFloat m_frac_bsdf, m_frac_lum;
    ScalarFloat m_weight_bsdf, m_weight_lum;

    // LTC tables as textures
    ref<Texture> ltc_1;
    ref<Texture> ltc_2;
    ref<Texture> ltc_3;
};

MI_IMPLEMENT_CLASS_VARIANT(LTCSurfaceIntegrator, SamplingIntegrator)
MI_EXPORT_PLUGIN(LTCSurfaceIntegrator, "Direct integrator");
NAMESPACE_END(mitsuba)
