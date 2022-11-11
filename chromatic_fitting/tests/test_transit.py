import unittest

# import pytest
from chromatic_fitting.models import transit
from chromatic_fitting.parameters import *
from pymc3 import Uniform, Normal
from pymc3_ext import eval_in_model
from exoplanet import ImpactParameter, orbits
from chromatic import *
import chromatic
import math

print(chromatic.version())
pltdir = "chromatic_fitting/tests/test_plots"


class TestTransit(unittest.TestCase):
    def create_rainbow(self):
        # create transit rainbow:
        r = SimulatedRainbow(dt=1 * u.minute, R=50).inject_noise(signal_to_noise=500)

        # add transit (with depth varying with wavelength):
        r = r.inject_transit(
            planet_radius=np.linspace(0.2, 0.15, r.nwave),
            method="trapezoid",
            P=1.0,
            t0=0.0,
            baseline=1.0
            # planet_params={
            #     "P": 1.0,
            #     "t0": 0.0,
            #     "baseline": 1.0
            # "a": 8.0,
            # "inc": 88,
            # },
        )

        # bin into 10 wavelength bins:
        nw = 5
        rb = r.bin(nwavelengths=int(r.nwave / nw), dt=5 * u.minute)
        return r, rb

    def setup_example_transit(self):
        t = transit.TransitModel()
        t.setup_parameters(
            period=1.0,
            epoch=Fitted(Uniform, lower=-0.05, upper=0.05),
            stellar_radius=Fitted(Uniform, lower=0.8, upper=1.2, testval=1),
            stellar_mass=Fitted(Uniform, lower=0.8, upper=1.2, testval=1),
            radius_ratio=WavelikeFitted(Normal, mu=0.1, sigma=0.05),
            impact_parameter=Fitted(ImpactParameter, ror=0.15, testval=0.2),
            limb_darkening=WavelikeFitted(Uniform, testval=[0.05, 0.35], shape=2),
            baseline=WavelikeFitted(Normal, mu=1.0, sigma=0.1),
        )
        return t

    def test_init(self):
        t = transit.TransitModel()
        assert t
        assert isinstance(t, transit.TransitModel)
        assert isinstance(t, transit.LightcurveModel)
        assert t.name == "transit"

    def test_setup_parameters(self):
        t = transit.TransitModel()
        t.setup_parameters()
        assert hasattr(t, "parameters")

        for k, v in t.defaults.items():
            assert t.parameters[f"{t.name+'_'}{k}"].value == v

        t = self.setup_example_transit()
        assert t.parameters["transit_period"].name == "transit_period"
        assert isinstance(t.parameters["transit_period"], Fixed)
        assert isinstance(t.parameters["transit_stellar_radius"], Fitted)
        assert isinstance(t.parameters["transit_radius_ratio"], Fitted)
        assert isinstance(t.parameters["transit_radius_ratio"], WavelikeFitted)

    def test_attach_data(self):
        t = self.setup_example_transit()
        ri, r = self.create_rainbow()
        t.attach_data(r)

        assert hasattr(t, "data")
        assert "flux" in t.data.fluxlike.keys()
        assert "uncertainty" in t.data.fluxlike.keys()
        assert "wavelength" in t.data.wavelike.keys()
        assert t.data.nwave == r.nwave
        assert t.data.ntime == r.ntime
        assert t.data.nwave == 5
        assert t.data.ntime == 61

        t.plot_lightcurves(filename=f"{pltdir}/lightcurves.png")

    def test_setup_orbit(self):
        t = self.setup_example_transit()
        ri, r = self.create_rainbow()
        t.attach_data(r)
        t.setup_orbit()

        assert hasattr(t, "orbit")
        assert type(t.orbit) == orbits.KeplerianOrbit
        assert t.orbit.a
        assert t._pymc3_model["transit_epoch"]

        with t._pymc3_model:
            assert eval_in_model(t.orbit.period) == 1.0
            # assert eval_in_model(t.orbit.a).round(8) == 8.75357171
            assert eval_in_model(t.orbit.r_star) == 1.0
            assert eval_in_model(t.orbit.m_star) == 1.0
            assert eval_in_model(t.orbit.b).round(8) == 0.2
            assert eval_in_model(t.orbit.t0) == 0.0

        t.plot_orbit(filename=f"{pltdir}/orbit_plot.png")

    def test_setup_lightcurve(self):
        t = self.setup_example_transit()
        ri, r = self.create_rainbow()
        t.attach_data(r)
        t.setup_lightcurves()

        assert hasattr(t, "orbit")
        assert "wavelength_0" in t.every_light_curve.keys()
        assert t.store_models is False
        assert t._pymc3_model["transit_a_R*"]
        assert t._pymc3_model["transit_radius_ratio"]

        with t._pymc3_model:
            assert (
                eval_in_model(t.every_light_curve["wavelength_0"].shape)[0]
                == t.data.nwave
            )
            assert (
                eval_in_model(t.every_light_curve["wavelength_0"].shape)[1]
                == t.data.ntime
            )

    def test_setup_likelihood(self):
        t = self.setup_example_transit()
        ri, r = self.create_rainbow()
        t.attach_data(r)
        t.setup_lightcurves()
        t.setup_likelihood()

        assert t._pymc3_model["data"]

        with t._pymc3_model:
            assert eval_in_model(t._pymc3_model["data"].shape)[0] == t.data.nwave
            assert eval_in_model(t._pymc3_model["data"].shape)[1] == t.data.ntime
            for i in range(t.data.nwave):
                assert np.all(
                    eval_in_model(t._pymc3_model["data"][i]) == t.data.flux[i, :]
                )

    def test_optimization_and_sampling(self):
        print(chromatic.version())
        t = self.setup_example_transit()
        ri, r = self.create_rainbow()
        t.attach_data(r)
        t.setup_lightcurves()
        t.setup_likelihood()

        # optimize, sample and summarize results
        opt = t.optimize(plot=False)
        t.sample(start=opt, tune=2000, draws=3000, chains=4, cores=4)
        t.summarize(round_to=7, hdi_prob=0.68, fmt="wide", print_table=False)

        # chech basic summary table properties
        assert hasattr(t, "summary")
        assert "mean" in t.summary
        assert "sd" in t.summary
        assert "hdi_16%" in t.summary
        assert "hdi_84%" in t.summary
        assert "r_hat" in t.summary
        assert "transit_radius_ratio[0]" in t.summary.index
        assert "transit_limb_darkening[0, 0]" in t.summary.index

        # check for convergence
        assert np.all(t.summary["r_hat"].values < 1.1)

        t.parameters["transit_a_R*"] = []
        results = t.get_results(uncertainty="sd")
        # true_params = r.metadata["injected_transit_parameters"]
        # true_t0 = true_params["t0"]
        # true_per = r.metadata["transit_parameters"]["P"]
        # true_aR = r.metadata["transit_parameters"]["a"]
        # true_incl = r.metadata["transit_parameters"]["inc"]
        # true_cosi = math.cos(true_incl * math.pi / 180)
        # true_b = true_aR * true_cosi

        twosigma = {
            "t0": 3 * results["transit_epoch_sd"].values,
            "aR": 3 * results["transit_a_R*_sd"].values,
        }
        print(results)
        # assert np.all(true_t0 < results["transit_epoch"].values + twosigma["t0"])
        # assert np.all(true_t0 > results["transit_epoch"].values - twosigma["t0"])
        # assert np.all(true_aR < results["transit_a_R*"].values + twosigma["aR"])
        # assert np.all(true_aR > results["transit_a_R*"].values - twosigma["aR*"])

        transmission_spectrum = t.make_transmission_spectrum_table(
            uncertainty=["hdi_16%", "hdi_84%"]
        )
        models = t.get_model()
        assert "w0" in models.keys()
        assert "w4" in models.keys()

        t.plot_lightcurves(filename=f"{pltdir}/lightcurves_with_models.png")
        t.plot_with_model_and_residuals(filename=f"{pltdir}/residuals.png")
        t.imshow_with_models(filename=f"{pltdir}/imshow_models.png")

        t.plot_transmission_spectrum(uncertainty=["hdi_16%", "hdi_84%"])
        plt.plot(
            ri.wavelength,
            r.metadata["transit_parameters"]["rp_unbinned"],
            label="True Rp/R*",
        )
        plt.legend()
        plt.savefig(f"{pltdir}/transmission_spectrum.png")


if __name__ == "__main__":
    unittest.main()
