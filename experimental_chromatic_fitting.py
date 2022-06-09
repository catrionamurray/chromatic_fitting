from chromatic_fitting import *
from tqdm import tqdm
import aesara
from parameters import *
from pymc3_ext import eval_in_model, optimize, sample
from pymc3 import Normal, Uniform, Model, HalfNormal, Deterministic, plot_trace, sample_prior_predictive, \
    sample_posterior_predictive
import warnings

class LightcurveModel:
    required_parameters = []

    def __init__(self, **kw):
        # define some default parameters (fixed):
        self.defaults = dict()
        pass

    # def __add__(self, model_to_add):
    #     # not working yet!
    #
    #     new_lcm = LightcurveModel()
    #     new_lcm.initialize_empty_model()
    #     new_lcm.defaults = self.defaults | model_to_add.defaults
    #
    #     # combine the parameters from the two separate models:
    #     new_parameters = self.parameters | model_to_add.parameters
    #
    #     new_lcm.setup_parameters(**dict(new_parameters))
    #
    #     return new_lcm

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
            elif isinstance(v, LightcurveModel):
                # if a LightcurveModel is passed as a prior for a variable then copy it, but
                # do not run the setup_lightcurves (weird pymc3 inheritance issues) - this will be called from within
                # the main setup_lightcurves!
                if hasattr(v, 'every_light_curve'):
                    new_v = v.__class__()
                    new_v.initialize_empty_model()
                    # can't just do new_v.setup_parameters(**dict(v.parameters)) - runs into weird pymc3 inheritance
                    # issues!
                    new_params = {}
                    for k2, v2 in v.parameters.items():
                        new_params[k2] = v2.__class__(v2.distribution, **v2.inputs)
                    new_v.setup_parameters(**new_params)
                    # if I've already attached data to this model then attach it to the new model:
                    if hasattr(v, 'data'):
                        new_v.attach_data(v.data)
                else:
                    new_v = v
                self.parameters[k] = new_v
                self.parameters[k].set_name(new_v)
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
                # mu = Deterministic(f'{k}_mu',self.every_light_curve[k])
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

class CombinedModel(LightcurveModel):
    def __repr__(self):
        if hasattr(self,'models_combined'):
            return f"<experimental chromatic combined model 🌈, models: {self.models_combined}>"
        else:
            return "<experimental chromatic combined model 🌈>"

    def attach_models(self, models):
        new_models = {}
        all_params = {}
        for name, model in models.items():
            if isinstance(model, LightcurveModel):
                # make a "copy" of each model:
                new_model = model.__class__()
                new_model.initialize_empty_model()
                # can't just do new_v.setup_parameters(**dict(v.parameters)) - runs into weird pymc3 inheritance
                # issues!
                model_params = {}
                for k, v in model.parameters.items():
                    if isinstance(v, Fixed):
                        model_params[k] = v.__class__(v.value)
                    elif isinstance(v, WavelikeFixed):
                        model_params[k] = v.__class__(v.values)
                    else:
                        model_params[k] = v.__class__(v.distribution, **v.inputs)
                new_model.setup_parameters(**model_params)
                new_models[name] = new_model
                all_params = all_params | model_params
            else:
                print("This class can only be used to combine LightcurveModels!")

        self.setup_parameters(**all_params)
        self.models_combined = new_models

    def apply_operation_to_constituent_models(self, operation, *args):
        for m in self.models_combined.values():
            try:
                operation(m, *args)
            except Exception as e:
                print(e)

    def attach_data(self, rainbow):
        self.data = rainbow._create_copy()
        self.apply_operation_to_constituent_models(LightcurveModel.attach_data,rainbow)

    def setup_orbits(self):
        transit_model_exists=False
        for mod in self.models_combined.values():
            if isinstance(mod,TransitModel):
                # if you set up the individual orbits then for whatever reason the parameters
                # aren't fitted in the combined model?
                # mod.setup_orbit()
                transit_model_exists = True
        if transit_model_exists:
            TransitModel.setup_orbit(self)

    def setup_lightcurves(self):
        for mod in self.models_combined.values():
            if isinstance(mod, TransitModel):
                TransitModel.setup_lightcurves(self)
            elif isinstance(mod, PolynomialModel):
                PolynomialModel.setup_lightcurves(self)

        # ALSO WORKS BUT IS LESS NICE:
        # with self.model:
        #     self.every_light_curve = {}
        #     for i, w in enumerate(self.data.wavelength):
        #         if f"wavelength_{i}" not in self.every_light_curve.keys():
        #             self.every_light_curve[f"wavelength_{i}"] = [0]*self.data.ntime
        #         for mod in self.models_combined.values():
        #             if isinstance(mod, TransitModel):
        #                 self.every_light_curve[f"wavelength_{i}"] += TransitModel.get_prior(self,i)
        #             elif isinstance(mod, PolynomialModel):
        #                 self.every_light_curve[f"wavelength_{i}"] += PolynomialModel.get_prior(self,i)


class PolynomialModel(LightcurveModel):

    def __init__(self, **kw):
        super().__init__(**kw)

    def __repr__(self):
        return "<experimental chromatic polynomial model 🌈>"

    def set_name(self, name):
        self.name = name

    def get_prior(self, i, *args, **kwargs):

        self.degree = self.parameters["p"].inputs['shape'] - 1
        x = self.data.time.to_value()
        poly = []

        p = self.parameters["p"].get_prior(i)

        for d in range(self.degree + 1):
            poly.append(p[d] * (x ** d))

        return pm.math.sum(poly, axis=0)

    def setup_lightcurves(self):
        """
        Create a polynomial model, given the stored parameters.
        [This should be run after .attach_data()]
        """

        # find the number of polynomial degrees to fit based on the user input
        self.degree = self.parameters["p"].inputs['shape'] - 1
        x = self.data.time.to_value()

        with self.model:
            if not hasattr(self, 'every_lightcurve'):
                self.every_light_curve = {}
            for i, w in enumerate(self.data.wavelength):
                # compute polynomial of user-defined degree
                poly = []
                p = self.parameters["p"].get_prior(i)
                for d in range(self.degree + 1):
                    poly.append(p[d] * (x ** d))

                # rd = RegexDict(dict(self.parameters))
                # models_to_add = rd.get_matching('add_.*')
                # for j, mod in enumerate(models_to_add):
                # models_to_add = pm.math.sum([mod.get_prior() for mod in models_to_add],axis=-1)
                if f"wavelength_{i}" not in self.every_light_curve.keys():
                    self.every_light_curve[f"wavelength_{i}"] = pm.math.sum(poly, axis=0) #+ models_to_add #+ pm.math.sum(models_to_add, axis=0)
                else:
                    self.every_light_curve[f"wavelength_{i}"] += pm.math.sum(poly, axis=0)

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

    def set_name(self, name):
        self.name = name

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

    def get_prior(self,i):
        try:
            self.orbit
        except AttributeError:
            self.setup_orbit()

        limb_darkening = self.parameters["limb_darkening"].get_prior(i)
        light_curves = xo.LimbDarkLightCurve(limb_darkening).get_light_curve(
                    orbit=self.orbit,
                    r=self.parameters["radius_ratio"].get_prior(i)
                      * self.parameters["stellar_radius"].get_prior(),
                    t=list(self.data.time.to_value("day")),
        )
        return pm.math.sum(light_curves, axis=-1) #+ (self.parameters["baseline"].get_prior(i))

    def setup_lightcurves(self):
        """
        Create an `exoplanet` light curve model, given the stored parameters.
        [This should be run after .setup_orbit() and .attach_data()]
        """
        with self.model:
            if not hasattr(self,'every_lightcurve'):
                self.every_light_curve = {}

            for i, w in enumerate(self.data.wavelength):
                limb_darkening = self.parameters["limb_darkening"].get_prior(i)
                light_curves = xo.LimbDarkLightCurve(limb_darkening).get_light_curve(
                    orbit=self.orbit,
                    r=self.parameters["radius_ratio"].get_prior(i)
                      * self.parameters["stellar_radius"].get_prior(),
                    t=list(self.data.time.to_value("day")),
                )

                if f"wavelength_{i}" not in self.every_light_curve.keys():
                    self.every_light_curve[f"wavelength_{i}"] = pm.math.sum(
                        light_curves, axis=-1
                    ) + (self.parameters["baseline"].get_prior(i))
                else:
                    self.every_light_curve[f"wavelength_{i}"] += pm.math.sum(
                        light_curves, axis=-1
                    ) + (self.parameters["baseline"].get_prior(i))

            self.model_chromatic_flux = [
                self.every_light_curve[k] for k in tqdm(self.every_light_curve)
            ]
            # pm.Deterministic("model_chromatic_flux", self.model_chromatic_flux)

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
