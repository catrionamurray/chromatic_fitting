from chromatic_fitting import *
from tqdm import tqdm
import aesara
from experimental_parameters import *


class TransitModel(chromatic_model):
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

            Examples:
            `period =1.2345`
                Setting a single scalar value means that
                the value will be fixed to that value through
                the fitting.
            `stellar_radius = Fitted('stellar_radius', lower=0.1, upper=2.0)`
                Setting a PyMC3 prior will indicate that the
                parameter should be inferred during the fit,
                with the given prior applied.
            `radius_ratio = WavelikeNormalPrior(mu=0.1, sigma=0.05)`
                Sett

        """
        # define some default parameter values (all fixed!)
        defaults = dict(
            stellar_radius=1.0,
            stellar_mass=1.0,
            radius_ratio=1.0,
            period=1.0,
            epoch=0.0,
            baseline=1.0,
            impact_parameter=0.5,
            limb_darkening=[0.2, 0.2],
        )

        # set up a dictionary of unprocessed
        unprocessed_parameters = dict(defaults)
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
        for k, v in self.parameters.items():
            print(f"{k} =\n  {v}\n")

    def initialize_empty_model(self):
        self.model = pm.Model()

    def setup_orbit(self):
        """
        Create a PyMC3 transit model, given the stored parameters.
        (This should be run after the )
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

    def attach_data(self, rainbow):
        self.data = rainbow

    def setup_lightcurves(self):
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

    def setup_likelihood(self):
        with self.model:
            for i, w in enumerate(self.data.wavelength):
                k = f"wavelength_{i}"
                pm.Normal(
                    f"{k}_data",
                    mu=self.every_light_curve[k],
                    sd=self.data.uncertainty[i, :],
                    observed=self.data.flux[i, :],
                )
