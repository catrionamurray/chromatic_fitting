from chromatic_fitting import *
from tqdm import tqdm
import aesara
from parameters import *
from pymc3_ext import eval_in_model, optimize, sample
from pymc3 import Normal, Uniform, Model, HalfNormal, Deterministic, plot_trace, sample_prior_predictive, sample_posterior_predictive

import warnings


class LightcurveModel:
    required_parameters = []
    # define some default parameter values (all fixed!)

    def __init__(self, **kw):
        # self.parameters = {}
        self.defaults = dict()
        pass

    def setup_parameters(self, **kw):
        """
        Set the values of the model parameters.

        Parameters
        ----------
        kw : dict
            All keyword arguments will be treated as model
            parameters and stored in the `self.parameters`
            dictionary attribute. This input both sets
            the initial values for each parameter and
            indicates whether it should be fixed or fitted,
            and if it is fitted what its prior should be.
        """

        # set up a dictionary of unprocessed
        unprocessed_parameters = dict(self.defaults)
        unprocessed_parameters.update(**kw)

        # process all parameters into Parameter objects
        self.parameters = {}
        for k, v in unprocessed_parameters.items():
            if isinstance(v, Parameter):
                self.parameters[k] = v
                self.parameters[k].set_name(k)
            else:
                self.parameters[k] = Fixed(v)
                self.parameters[k].set_name(v)

        # check that all the necessary parameters are defined somehow
        for k in self.required_parameters:
            assert k in self.parameters

    def summarize_parameters(self):
        """
        Print a friendly summary of the parameters.
        """
        for k, v in self.parameters.items():
            print(f"{k} =\n  {v}\n")

    def initialize_empty_model(self):
        """
        Restart with an empty model.
        """
        self.model = pm.Model()

    def attach_data(self, rainbow):
        """
        Connect a `chromatic` Rainbow dataset to this object.
        """
        self.data = rainbow._create_copy()

    def setup_likelihood(self):
        """
        Connect the light curve model to the actual data it aims to explain.
        """
        with self.model:
            for i, w in enumerate(self.data.wavelength):
                k = f"wavelength_{i}"
                pm.Normal(
                    f"{k}_data",
                    mu=self.every_light_curve[k],
                    sd=self.data.uncertainty[i, :],
                    observed=self.data.flux[i, :],
                )

    def sample_prior(self, ndraws=5):
        """
        Draw samples from the prior distribution.
        """
        with self.model:
            return sample_prior_predictive(ndraws)

    def sample_posterior(self, trace, ndraws=5):
        """
        Draw samples from the posterior distribution.
        """
        with self.model:
            return sample_posterior_predictive(trace, ndraws)

class PolynomialModel(LightcurveModel):

    def __init__(self, **kw):
        super().__init__(**kw)

    def __repr__(self):
        return "<experimental chromatic polynomial model 🌈>"

    def setup_lightcurves(self):
        """
        Creaet a polynomial model, given the stored parameters.
        [This should be run after .attach_data()]
        """

        # find the number of polynomial degrees to fit based on the user input
        self.degree = self.parameters["p"].inputs['shape']-1

        with self.model:
            x = self.data.time.to_value()
            self.every_light_curve = {}
            for i, w in enumerate(self.data.wavelength):
                poly = []
                p = self.parameters["p"].get_prior(i)
                for d in range(self.degree + 1):
                    poly.append(p[d] * (x ** d))
                self.every_light_curve[f"wavelength_{i}"] = pm.math.sum(poly, axis=0)

class TransitModel(LightcurveModel):
    def __repr__(self):
        return "<experimental chromatic transit model 🌈>"

    required_parameters = [
        "stellar_radius",
        "stellar_mass",
        "radius_ratio",
        "period",
        "epoch",
        "baseline",
        "impact_parameter",
        "limb_darkening",
    ]

    def __init__(self, **kw):
        super().__init__(**kw)
        self.defaults = dict(
            stellar_radius=1.0,
            stellar_mass=1.0,
            radius_ratio=1.0,
            period=1.0,
            epoch=0.0,
            baseline=1.0,
            impact_parameter=0.5,
            limb_darkening=[0.2, 0.2],
        )

    def setup_orbit(self):
        """
        Create an `exoplanet` orbit model, given the stored parameters.
        [This should be run after .setup_parameters()]
        """

        with self.model:
            # Set up a Keplerian orbit for the planets
            self.orbit = xo.orbits.KeplerianOrbit(
                period=self.parameters["period"].get_prior(),
                t0=self.parameters["epoch"].get_prior(),
                b=self.parameters["impact_parameter"].get_prior(),
                r_star=self.parameters["stellar_radius"].get_prior(),
                m_star=self.parameters["stellar_mass"].get_prior(),
            )

    def setup_lightcurves(self):
        """
        Create an `exoplanet` light curve model, given the stored parameters.
        [This should be run after .setup_orbit() and .attach_data()]
        """
        with self.model:
            self.every_light_curve = {}
            for i, w in enumerate(self.data.wavelength):
                limb_darkening = self.parameters["limb_darkening"].get_prior(i)
                light_curves = xo.LimbDarkLightCurve(limb_darkening).get_light_curve(
                    orbit=self.orbit,
                    r=self.parameters["radius_ratio"].get_prior(i)
                      * self.parameters["stellar_radius"].get_prior(),
                    t=list(self.data.time.to_value("day")),
                )
                self.every_light_curve[f"wavelength_{i}"] = pm.math.sum(
                    light_curves, axis=-1
                ) + self.parameters["baseline"].get_prior(i)

            self.model_chromatic_flux = [
                self.every_light_curve[k] for k in tqdm(self.every_light_curve)
            ]

    def run_simultaneous_fit(self, r):
        """
            Run the entire simultaneous wavelength fit.
        """
        self.setup_orbit()
        self.attach_data(r)
        self.setup_lightcurves()
        self.setup_likelihood()
        with self.model:
            opt = optimize(start=self.model.test_point)
            opt = optimize(start=opt)
            trace = sample(start=opt)
            summary = az.summary(trace, round_to=7, fmt='wide')
            print(summary)

        return trace, summary

    def plot_orbit(self, timedata=None):
        """
        Plot the orbit model.
        """

        if not hasattr(self, 'data'):
            if timedata is None:
                warnings.warn(
                    "No data attached to this object and no time data provided. Plotting orbit will not work.")
                print("No data attached to this object and no time data provided. Plotting orbit will not work.")
                return
        else:
            timedata = self.data.time

        with self.model:
            x, y, z = [eval_in_model(bla, point=self.model.test_point) for bla in
                       self.orbit.get_planet_position(timedata)]  # {}
            plt.figure(figsize=(10, 3))
            theta = np.linspace(0, 2 * np.pi)
            plt.fill_between(np.cos(theta), np.sin(theta), color='gray')
            plt.scatter(x, y, c=timedata)
            plt.axis('scaled')
            plt.ylim(-1, 1)
            plt.show()
            plt.close()
