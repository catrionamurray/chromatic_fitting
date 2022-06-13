from chromatic_fitting import *
from tqdm import tqdm
import aesara
from parameters import *
from arviz import summary
from pymc3_ext import eval_in_model, optimize, sample
from pymc3 import Normal, Uniform, Model, HalfNormal, Deterministic, plot_trace, sample_prior_predictive, \
    sample_posterior_predictive
import warnings
from collections import Counter

class LightcurveModel:
    required_parameters = []

    def __init__(self, **kw):
        # define some default parameters (fixed):
        self.defaults = dict()
        self.optimization = "simultaneous"
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

    def white_light_curve(self):
        """
        Generate inverse-variance weighted white light curve by binning Rainbow to one bin
        """
        self.white_light = self.data.bin(nwavelengths=self.data.nwave)

    def choose_optimization_method(self,optimization_method='simultaneous'):
        """
        Choose the optimization method
        """
        possible_optimization_methods = ["simultaneous", "white_light", "separate"]
        if optimization_method in possible_optimization_methods:
            self.optimization = optimization_method
        else:
            print("Unrecognised optimization method, please select one of: " + str(", ".join(possible_optimization_methods)))
            self.optimization = "simultaneous"

    def get_data(self):
        """
        Extract the data to use for the optimization depending on the method chosen
        """
        if hasattr(self, 'optimization'):
            if self.optimization == "white_light":
                self.white_light_curve()
                return self.white_light
        return self.data

    def setup_likelihood(self):
        """
        Connect the light curve model to the actual data it aims to explain.
        """
        data = self.get_data()

        with self.model:
            for i, w in enumerate(data.wavelength):
                k = f"wavelength_{i}"
                # mu = Deterministic(f'{k}_mu',self.every_light_curve[k])
                pm.Normal(
                    f"{k}_data",
                    mu=self.every_light_curve[k],
                    sd=data.uncertainty[i, :],
                    observed=data.flux[i, :],
                )

    def sample_prior(self, ndraws=3):
        """
        Draw samples from the prior distribution.
        :parameter n
        Number of priors to sample
        """
        with self.model:
            return sample_prior_predictive(ndraws)

    def sample_posterior(self, trace, ndraws=3):
        """
        Draw samples from the posterior distribution.
        :parameter n
        Number of posteriors to sample
        """
        with self.model:
            return sample_posterior_predictive(trace, ndraws)

    def sample(self, **kw):
        """
        Wrapper for PyMC3_ext sample
        """
        with self.model:
            self.trace = sample(**kw)

    def summarize(self, **kw):
        """
        Wrapper for arviz summary
        """
        with self.model:
            self.summary = summary(self.trace, **kw)

    def plot_priors(self, n=3):
        """
        Plot n prior samples from the parameter distributions defined by the user
        :parameter n
        Number of priors to plot (default=3)
        """
        data = self.get_data()
        prior_predictive_trace = self.sample_prior(ndraws=n)
        for i in range(n):
            flux_for_this_sample = np.array([prior_predictive_trace[f'wavelength_{w}_data'][i] for w in range(data.nwave)])
            data.fluxlike[f'prior-predictive-{i}'] = flux_for_this_sample
        data.imshow_quantities()

    def plot_posteriors(self, trace, n=3):
        """
        Plot n posterior samples from the parameter distributions defined by the user
        :parameter trace
        PyMC3 trace object (need to run sample first)
        :parameter n
        Number of posteriors to plot (default=3)
        """
        data = self.get_data()
        posterior_predictive_trace = self.sample_posterior(trace, ndraws=n)
        for i in range(n):
            flux_for_this_sample = np.array(
                [posterior_predictive_trace[f'wavelength_{w}_data'][i] for w in range(data.nwave)])
            data.fluxlike[f'posterior-predictive-{i}'] = flux_for_this_sample
        data.imshow_quantities()

class CombinedModel(LightcurveModel):
    def __repr__(self):
        if hasattr(self,'models_combined'):
            return f"<experimental chromatic combined model 🌈, models: {self.models_combined}>"
        else:
            return "<experimental chromatic combined model 🌈>"

    def attach_models(self, models):
        """
        Attach multiple LightCurveModel in dictionary to the CombinedModel
        """
        new_models = {}
        all_params = {}

        for name, model in models.items():
            # check that the models passed to this function are LightcurveModels
            if isinstance(model, LightcurveModel):
                # make a "copy" of each model:
                new_model = model.__class__()
                new_model.initialize_empty_model()
                # can't just do new_v.setup_parameters(**dict(v.parameters)) - runs into weird pymc3 inheritance
                # issues!
                model_params = {}
                # for every parameter in the separate models redefine them in the separate and new models
                for k, v in model.parameters.items():
                    if isinstance(v, Fixed):
                        # parameter is Fixed
                        model_params[k] = v.__class__(v.value)
                    elif isinstance(v, WavelikeFixed):
                        # parameter is WavelikeFixed
                        model_params[k] = v.__class__(v.values)
                    else:
                        # parameter is Fitted or WavelikeFitted
                        model_params[k] = v.__class__(v.distribution, **v.inputs)
                # set up parameters in new models
                new_model.setup_parameters(**model_params)
                new_models[name] = new_model
                all_params = all_params | model_params # what happens if same keys?
            else:
                print("This class can only be used to combine LightcurveModels!")

        # set up parameters in new combined model
        self.setup_parameters(**all_params)
        self.models_combined = new_models

    def apply_operation_to_constituent_models(self, operation, *args):
        """
        Apply an operation to all models within a combined model
        """
        for m in self.models_combined.values():
            try:
                operation(m, *args)
            except Exception as e:
                print(e)

    def attach_data(self, r):
        """
        Connect a `chromatic` Rainbow dataset to this object and the constituent models.
        :parameter r:
        Rainbow object
        """
        self.data = r._create_copy()
        self.apply_operation_to_constituent_models(LightcurveModel.attach_data, r)

    def setup_orbit(self):
        """
        Create an `exoplanet` orbit model, given the stored parameters.
        """
        transit_model_exists = False
        for mod in self.models_combined.values():
            if isinstance(mod, TransitModel):
                # if you set up the individual orbits then for whatever reason the parameters
                # aren't fitted in the combined model?
                # mod.setup_orbit()
                transit_model_exists = True
        if transit_model_exists:
            TransitModel.setup_orbit(self)

    def setup_lightcurves(self):
        """
        Set-up lightcurves in combined model : for each consituent model set-up the lightcurves according to their type
        """

        # I want to get this to work, but it doesn't seem to save the self.every_light_curve for both :(
        # for mod in self.models_combined.values():
        #     if isinstance(mod, TransitModel):
        #         TransitModel.setup_lightcurves(self)
        #     elif isinstance(mod, PolynomialModel):
        #         PolynomialModel.setup_lightcurves(self)

        data = self.get_data()
        # WORKS BUT IS LESS NICE:
        with self.model:
            self.every_light_curve = {}
            for i, w in enumerate(data.wavelength):
                if f"wavelength_{i}" not in self.every_light_curve.keys():
                    self.every_light_curve[f"wavelength_{i}"] = [0]*data.ntime
                for mod in self.models_combined.values():
                    if isinstance(mod, TransitModel):
                        self.every_light_curve[f"wavelength_{i}"] += TransitModel.get_prior(self,i)
                    elif isinstance(mod, PolynomialModel):
                        self.every_light_curve[f"wavelength_{i}"] += PolynomialModel.get_prior(self,i)

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
        data = self.get_data()
        # find the number of polynomial degrees to fit based on the user input
        self.degree = self.parameters["p"].inputs['shape'] - 1
        x = data.time.to_value()

        with self.model:
            if not hasattr(self, 'every_lightcurve'):
                self.every_light_curve = {}
            for i, w in enumerate(data.wavelength):
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

        data = self.get_data()

        with self.model:
            if not hasattr(self,'every_lightcurve'):
                self.every_light_curve = {}

            for i, w in enumerate(data.wavelength):
                limb_darkening = self.parameters["limb_darkening"].get_prior(i)
                light_curves = xo.LimbDarkLightCurve(limb_darkening).get_light_curve(
                    orbit=self.orbit,
                    r=self.parameters["radius_ratio"].get_prior(i)
                      * self.parameters["stellar_radius"].get_prior(),
                    t=list(data.time.to_value("day")),
                )

                mu = pm.math.sum(
                        light_curves, axis=-1
                    ) + (self.parameters["baseline"].get_prior(i))

                # self.every_light_curve = dict(Counter(self.every_light_curve)+Counter({f"wavelength_{i}":mu}))
                if f"wavelength_{i}" not in self.every_light_curve.keys():
                    self.every_light_curve[f"wavelength_{i}"] = pm.math.sum(
                        light_curves, axis=-1
                    ) + (self.parameters["baseline"].get_prior(i))
                else:
                    self.every_light_curve[f"wavelength_{i}"] += pm.math.sum(
                        light_curves, axis=-1
                    ) + (self.parameters["baseline"].get_prior(i))

            self.model_chromatic_transit_flux = [
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
