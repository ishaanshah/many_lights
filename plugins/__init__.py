import mitsuba as mi

from plugins.ltc import LTCIntegrator
from plugins.ris import RISIntegrator
from plugins.ltc_mc import LTC_MCIntegragtor
from plugins.ltc_ris import LTC_RISIntegrator

# Register all plugins
mi.register_integrator("ltc", lambda props: LTCIntegrator(props))
mi.register_integrator("ris", lambda props: RISIntegrator(props))
mi.register_integrator("ltc_mc", lambda props: LTC_MCIntegragtor(props))
mi.register_integrator("ltc_ris", lambda props: LTC_RISIntegrator(props))