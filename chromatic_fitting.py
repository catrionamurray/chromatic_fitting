# from src.archive.chromatic_fitting import *
from src.imports import *
from tqdm import tqdm
from parameters import *
from arviz import summary
from pymc3_ext import eval_in_model, optimize, sample
from pymc3 import sample_prior_predictive, sample_posterior_predictive
import warnings


def add_dicts(dict_1, dict_2):
    dict_3 = {**dict_1, **dict_2}
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            dict_3[key] = value + dict_1[key]
    return dict_3


def subtract_dicts(dict_1, dict_2):
    dict_3 = {**dict_1, **dict_2}
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            dict_3[key] = value - dict_1[key]
    return dict_3


def multiply_dicts(dict_1, dict_2):
    dict_3 = {**dict_1, **dict_2}
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            dict_3[key] = value * dict_1[key]
    return dict_3


def divide_dicts(dict_1, dict_2):
    dict_3 = {**dict_1, **dict_2}
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            dict_3[key] = value / dict_1[key]
    return dict_3


# define a dictionary mapping operations to functions
combination_options = {
    "+": add_dicts,
    "-": subtract_dicts,
    "*": multiply_dicts,
    "/": divide_dicts,
}


class LightcurveModel:
    required_parameters = []

    def __add__(self, other):
        """
        What should we return for `a + b` for two models `a` and `b`?
        """
        cm = CombinedModel()
        cm.initialize_empty_model()
        cm.attach_models({"left": self, "right": other}, how_to_combine="+")
        return cm

    def __sub__(self, other):
        """
        What should we return for `a - b` for two models `a` and `b`?
        """
        cm = CombinedModel()
        cm.initialize_empty_model()
        cm.attach_models({"left": self, "right": other}, how_to_combine="-")
        return cm

    def __mul__(self, other):
        """
        What should we return for `a * b` for two models `a` and `b`?
        """
        cm = CombinedModel()
        cm.initialize_empty_model()
        cm.attach_models({"left": self, "right": other}, how_to_combine="*")
        return cm

    def __truediv__(self, other):
        """
        What should we return for `a - b` for two models `a` and `b`?
        """
        cm = CombinedModel()
        cm.initialize_empty_model()
        cm.attach_models({"left": self, "right": other}, how_to_combine="/")
        return cm

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
                if hasattr(v, "every_light_curve"):
                    new_v = v.__class__()
                    new_v.initialize_empty_model()
                    # can't just do new_v.setup_parameters(**dict(v.parameters)) - runs into weird pymc3 inheritance
                    # issues!
                    new_params = {}
                    for k2, v2 in v.parameters.items():
                        new_params[k2] = v2.__class__(v2.distribution, **v2.inputs)
                    new_v.setup_parameters(**new_params)
                    # if I've already attached data to this model then attach it to the new model:
                    if hasattr(v, "data"):
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

    def reinitialize_parameters(self, exclude=[]):
        for k, v in self.parameters.items():
            if k not in exclude:
                if isinstance(v, Fitted):
                    v.clear_prior()

    def initialize_empty_model(self):
        """
        Restart with an empty model.
        """
        self.pymc3_model = pm.Model()

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

    def choose_optimization_method(self, optimization_method="simultaneous"):
        """
        Choose the optimization method
        [attach_data has to be run before!]
        """
        possible_optimization_methods = ["simultaneous", "white_light", "separate"]
        if optimization_method in possible_optimization_methods:
            self.optimization = optimization_method
            if self.optimization == "separate":
                try:
                    self.create_multiple_models()
                    self.change_all_priors_to_Wavelike()
                except:
                    pass
        else:
            print(
                "Unrecognised optimization method, please select one of: "
                + str(", ".join(possible_optimization_methods))
            )
            self.optimization = "simultaneous"

    def create_multiple_models(self):
        """
        Create a list of models to process wavelengths separately
        """
        self.pymc3_model = [pm.Model() for n in range(self.data.nwave)]
        # if the LightCurve model is a CombinedModel then update the constituent models too
        if isinstance(self, CombinedModel):
            for mod in self.chromatic_models.values():
                mod.pymc3_model = self.pymc3_model
                mod.optimization = self.optimization

    def change_all_priors_to_Wavelike(self):
        for k, v in self.parameters.items():
            if isinstance(v, Fitted) and not isinstance(v, WavelikeFitted):
                self.parameters[k] = WavelikeFitted(v.distribution, **v.inputs)
            if isinstance(v, Fixed) and not isinstance(v, WavelikeFixed):
                self.parameters[k] = WavelikeFixed([v.value] * self.data.nwave)

    def change_all_priors_to_notWavelike(self):
        for k, v in self.parameters.items():
            if isinstance(v, WavelikeFitted):
                self.parameters[k] = Fitted(v.distribution, **v.inputs)
            if isinstance(v, WavelikeFixed):
                self.parameters[k] = Fixed(v.values[0])

    def separate_wavelengths(self, i):
        data_copy = self.data._create_copy()
        for k, v in data_copy.fluxlike.items():
            data_copy.fluxlike[k] = np.array([data_copy.fluxlike[k][i, :]])
        for k, v in data_copy.wavelike.items():
            data_copy.wavelike[k] = [data_copy.wavelike[k][i]]
        return data_copy

    def get_data(self, i=0):
        """
        Extract the data to use for the optimization depending on the method chosen
        """
        if hasattr(self, "optimization"):
            if self.optimization == "white_light":
                self.white_light_curve()
                return self.white_light
            if self.optimization == "separate":
                return self.separate_wavelengths(i)
        return self.data

    def setup_likelihood(self):
        """
        Connect the light curve model to the actual data it aims to explain.
        """
        # data = self.get_data()

        if self.optimization == "separate":
            models = self.pymc3_model
            datas = [self.get_data(i) for i in range(self.data.nwave)]
        else:
            models = [self.pymc3_model]
            datas = [self.get_data()]

        for j, (mod, data) in enumerate(zip(models, datas)):
            with mod:
                for i, w in enumerate(data.wavelength):
                    k = f"wavelength_{j+i}"
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
        try:
            with self.pymc3_model:
                return sample_prior_predictive(ndraws)
        except:
            priors = []
            for mod in self.pymc3_model:
                with mod:
                    priors.append(sample_prior_predictive(ndraws))
            return priors

    def sample_posterior(self, ndraws=3):
        """
        Draw samples from the posterior distribution.
        :parameter n
        Number of posteriors to sample
        """
        if not hasattr(self, "trace"):
            print("Sampling has not been run yet! Running now with defaults...")
            self.sample()

        try:
            with self.pymc3_model:
                return sample_posterior_predictive(self.trace, ndraws)
        except:
            posteriors = []
            for mod, trace in zip(self.pymc3_model, self.trace):
                with mod:
                    posteriors.append(sample_posterior_predictive(trace, ndraws))
            return posteriors

    def optimize(self, **kw):
        """
        Wrapper for PyMC3_ext sample
        """
        with self.pymc3_model:
            return optimize(**kw)

    def sample(self, **kw):
        """
        Wrapper for PyMC3_ext sample
        """
        if self.optimization == "separate":
            self.trace = []
            for mod in self.pymc3_model:
                with mod:
                    self.trace.append(sample(**kw))
        else:
            with self.pymc3_model:
                self.trace = sample(**kw)

    def summarize(self, **kw):
        """
        Wrapper for arviz summary
        """
        if not hasattr(self, "trace"):
            print("Sampling has not been run yet! Running now with defaults...")
            self.sample()

        if self.optimization == "separate":
            self.summary = []
            for mod, trace in zip(self.pymc3_model, self.trace):
                with mod:
                    self.summary.append(summary(trace, **kw))
        else:
            with self.pymc3_model:
                self.summary = summary(self.trace, **kw)

    def plot_priors(self, n=3):
        """
        Plot n prior samples from the parameter distributions defined by the user
        :parameter n
        Number of priors to plot (default=3)
        """
        # setup the models, data and orbits in a format for looping
        if self.optimization == "separate":
            datas = [self.get_data(i) for i in range(self.data.nwave)]
            prior_predictive_traces = self.sample_prior(ndraws=n)
        else:
            datas = [self.get_data()]
            prior_predictive_traces = [self.sample_prior(ndraws=n)]

        for nm, (data, prior_predictive_trace) in enumerate(
            zip(datas, prior_predictive_traces)
        ):
            for i in range(n):
                flux_for_this_sample = np.array(
                    [
                        prior_predictive_trace[f"wavelength_{w+nm}_data"][i]
                        for w in range(data.nwave)
                    ]
                )
                data.fluxlike[f"prior-predictive-{i}"] = flux_for_this_sample
            data.imshow_quantities()

    def plot_posteriors(self, n=3):
        """
        Plot n posterior samples from the parameter distributions defined by the user
        :parameter trace
        PyMC3 trace object (need to run sample first)
        :parameter n
        Number of posteriors to plot (default=3)
        """
        if self.optimization == "separate":
            datas = [self.get_data(i) for i in range(self.data.nwave)]
            # traces = self.trace
            posterior_predictive_traces = self.sample_posterior(n)
        else:
            datas = [self.get_data()]
            # traces = [self.trace]
            posterior_predictive_traces = [self.sample_posterior(n)]

        for nm, (data, posterior_predictive_trace) in enumerate(
            zip(datas, posterior_predictive_traces)
        ):
            for i in range(n):
                flux_for_this_sample = np.array(
                    [
                        posterior_predictive_trace[f"wavelength_{w+nm}_data"][i]
                        for w in range(data.nwave)
                    ]
                )
                data.fluxlike[f"posterior-predictive-{i}"] = flux_for_this_sample
            data.imshow_quantities()

    def extract_mean_posteriors(self, summary, i):
        # there's definitely a sleeker way to do this
        posterior_means = summary["mean"]
        fv = {}
        for k, v in self.parameters.items():
            if k in posterior_means.index:
                fv[k] = posterior_means[k]
            elif f"{k}[0]" in posterior_means.index:
                n = 0
                fv[k] = []
                while f"{k}[{n}]" in posterior_means.index:
                    fv[k].append(posterior_means[f"{k}[{n}]"])
                    n += 1
            elif f"{k}_w{i}" in posterior_means.index:
                fv[k] = posterior_means[f"{k}_w{i}"]
            elif f"{k}_w{i}[0]" in posterior_means.index:
                n = 0
                fv[k] = []
                while f"{k}_w{i}[{n}]" in posterior_means.index:
                    fv[k].append(posterior_means[f"{k}_w{i}[{n}]"])
                    n += 1
            elif f"{k}_w{i}" in posterior_means.index:
                fv[k] = posterior_means[f"{k}_w{i}"]
            else:
                if isinstance(v, WavelikeFixed):
                    fv[k] = v.values[0]
                elif isinstance(v, Fixed):
                    fv[k] = v.value

        return fv


class CombinedModel(LightcurveModel):
    def __repr__(self):
        if hasattr(self, "chromatic_models"):
            return f"({self.chromatic_models['left']} {self.how_to_combine} {self.chromatic_models['right']})"
            # return f"<experimental chromatic combined model 🌈, models: {self.chromatic_models}>"
        else:
            return "<experimental chromatic combined model 🌈>"

    def attach_models(self, models, how_to_combine="+"):
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
                # new_model.initialize_empty_model()
                new_model.pymc3_model = self.pymc3_model
                # can't just do new_v.setup_parameters(**dict(v.parameters)) - runs into weird pymc3 inheritance
                # issues!
                model_params = {}
                # for every parameter in the separate models redefine them in the separate and new models
                for k, v in model.parameters.items():
                    if isinstance(v, WavelikeFixed):
                        # parameter is WavelikeFixed
                        model_params[k] = v.__class__(v.values)
                    elif isinstance(v, Fixed):
                        # parameter is Fixed
                        model_params[k] = v.__class__(v.value)
                    else:
                        # parameter is Fitted or WavelikeFitted
                        model_params[k] = v.__class__(v.distribution, **v.inputs)
                # set up parameters in new models
                new_model.setup_parameters(**model_params)
                new_models[name] = new_model
                # all_params = all_params | model_params # what happens if same keys?
            else:
                print("This class can only be used to combine LightcurveModels!")

        # set up parameters in new combined model
        # self.setup_parameters(**all_params)
        self.chromatic_models = new_models
        self.how_to_combine = how_to_combine

    def apply_operation_to_constituent_models(self, operation, *args):
        """
        Apply an operation to all models within a combined model
        """
        for m in self.chromatic_models.values():
            try:
                op = getattr(m, operation)
                op(*args)
            except Exception as e:
                print(m, operation)
                print(e)

    def attach_data(self, r):
        """
        Connect a `chromatic` Rainbow dataset to this object and the constituent models.
        :parameter r:
        Rainbow object
        """
        self.data = r._create_copy()
        self.apply_operation_to_constituent_models("attach_data", r)

    def choose_optimization_method(self, optimization_method="simultaneous"):
        LightcurveModel.choose_optimization_method(self, optimization_method)
        if optimization_method == "separate":
            self.apply_operation_to_constituent_models("change_all_priors_to_Wavelike")

    def setup_orbit(self):
        """
        Create an `exoplanet` orbit model, given the stored parameters.
        """
        self.apply_operation_to_constituent_models("setup_orbit")

    def setup_lightcurves(self):
        """
        Set-up lightcurves in combined model : for each consituent model set-up the lightcurves according to their type
        """
        self.every_light_curve = {}
        self.apply_operation_to_constituent_models("setup_lightcurves")
        for mod in self.chromatic_models.values():
            self.every_light_curve = combination_options[self.how_to_combine](
                self.every_light_curve, mod.every_light_curve
            )


class PolynomialModel(LightcurveModel):
    def __init__(self, **kw):
        super().__init__(**kw)

    def __repr__(self):
        return "<experimental chromatic polynomial model 🌈>"

    def set_name(self, name):
        self.name = name

    def get_prior(self, i, *args, **kwargs):
        data = self.get_data()
        self.degree = self.parameters["p"].inputs["shape"] - 1
        x = data.time.to_value()
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
        self.degree = self.parameters["p"].inputs["shape"] - 1

        if self.optimization == "separate":
            models = self.pymc3_model
            datas = [self.get_data(i) for i in range(self.data.nwave)]
        else:
            models = [self.pymc3_model]
            datas = [self.get_data()]

        if not hasattr(self, "every_lightcurve"):
            self.every_light_curve = {}

        for j, (mod, data) in enumerate(zip(models, datas)):
            with mod:
                x = data.time.to_value()

                for i, w in enumerate(data.wavelength):
                    # compute polynomial of user-defined degree
                    poly = []
                    p = self.parameters["p"].get_prior(i + j)
                    for d in range(self.degree + 1):
                        poly.append(p[d] * (x ** d))

                    # rd = RegexDict(dict(self.parameters))
                    # models_to_add = rd.get_matching('add_.*')
                    # for j, mod in enumerate(models_to_add):
                    # models_to_add = pm.math.sum([mod.get_prior() for mod in models_to_add],axis=-1)
                    if f"wavelength_{i+j}" not in self.every_light_curve.keys():
                        self.every_light_curve[f"wavelength_{i+j}"] = pm.math.sum(
                            poly, axis=0
                        )  # + models_to_add #+ pm.math.sum(models_to_add, axis=0)
                    else:
                        self.every_light_curve[f"wavelength_{i+j}"] += pm.math.sum(
                            poly, axis=0
                        )

    def polynomial_model(self, poly_params):
        data = self.get_data()
        poly = []
        for d in range(self.degree + 1):
            poly.append(poly_params["p"][d] * (data.time.to_value() ** d))
        return np.sum(poly, axis=0)


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

        if self.optimization == "separate":
            models = self.pymc3_model
        else:
            models = [self.pymc3_model]

        self.orbit = []
        for j, mod in enumerate(models):
            with mod:

                # Set up a Keplerian orbit for the planets
                orbit = xo.orbits.KeplerianOrbit(
                    period=self.parameters["period"].get_prior(j),
                    t0=self.parameters["epoch"].get_prior(j),
                    b=self.parameters["impact_parameter"].get_prior(j),
                    r_star=self.parameters["stellar_radius"].get_prior(j),
                    m_star=self.parameters["stellar_mass"].get_prior(j),
                )

                # store a separate orbit for each model if we are optimizing wavelengths separately
                if self.optimization == "separate":
                    self.orbit.append(orbit)
                else:
                    self.orbit = orbit

    def get_prior(self, i):
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
        return pm.math.sum(
            light_curves, axis=-1
        )  # + (self.parameters["baseline"].get_prior(i))

    def setup_lightcurves(self):
        """
        Create an `exoplanet` light curve model, given the stored parameters.
        [This should be run after .attach_data()]
        """

        # ensure that attach data has been run before setup_lightcurves
        if not hasattr(self, "data"):
            print("You need to attach some data to this chromatic model!")
            return

        # ensure that setup_orbit has been run before setup_lightcurves
        if not hasattr(self, "orbit"):
            self.setup_orbit()

        # setup the models, data and orbits in a format for looping
        if self.optimization == "separate":
            models = self.pymc3_model
            datas = [self.get_data(i) for i in range(self.data.nwave)]
            orbits = self.orbit
        else:
            models = [self.pymc3_model]
            datas = [self.get_data()]
            orbits = [self.orbit]

        if not hasattr(self, "every_lightcurve"):
            self.every_light_curve = {}

        for j, (mod, data, orbit) in enumerate(zip(models, datas, orbits)):
            with mod:
                for i, w in enumerate(data.wavelength):
                    # create quadratic limb-darkening lightcurves from Exoplanet
                    limb_darkening = self.parameters["limb_darkening"].get_prior(j + i)
                    planet_radius = self.parameters["radius_ratio"].get_prior(
                        j + i
                    ) * self.parameters["stellar_radius"].get_prior(j + i)
                    light_curves = xo.LimbDarkLightCurve(
                        limb_darkening
                    ).get_light_curve(
                        orbit=orbit,
                        r=planet_radius,
                        t=list(data.time.to_value("day")),
                    )

                    # calculate the transit + baseline model
                    mu = pm.math.sum(light_curves, axis=-1) + (
                        self.parameters["baseline"].get_prior(j + i)
                    )

                    # self.every_light_curve = dict(Counter(self.every_light_curve)+Counter({f"wavelength_{i}":mu}))
                    if f"wavelength_{j+i}" not in self.every_light_curve.keys():
                        self.every_light_curve[f"wavelength_{j+i}"] = pm.math.sum(
                            light_curves, axis=-1
                        ) + (self.parameters["baseline"].get_prior(j + i))
                    else:
                        self.every_light_curve[f"wavelength_{j+i}"] += pm.math.sum(
                            light_curves, axis=-1
                        ) + (self.parameters["baseline"].get_prior(j + i))

                self.model_chromatic_transit_flux = [
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
        with self.pymc3_model:
            opt = optimize(start=self.pymc3_model.test_point)
            opt = optimize(start=opt)
            trace = sample(start=opt)
            summary = az.summary(trace, round_to=7, fmt="wide")
            print(summary)

        return trace, summary

    def transit_model(self, transit_params):
        data = self.get_data()
        orbit = xo.orbits.KeplerianOrbit(
            period=transit_params["period"],
            t0=transit_params["epoch"],
            b=transit_params["impact_parameter"],
            r_star=transit_params["stellar_radius"],
            m_star=transit_params["stellar_mass"],
        )
        ldlc = (
            xo.LimbDarkLightCurve(transit_params["limb_darkening"])
            .get_light_curve(
                orbit=orbit,
                r=transit_params["radius_ratio"] * transit_params["stellar_radius"],
                t=list(data.time.to_value("day")),
            )
            .eval()
        )
        return ldlc.transpose()[0] + transit_params["baseline"]

    def plot_orbit(self, timedata=None):
        """
        Plot the orbit model.
        """

        if not hasattr(self, "data"):
            if timedata is None:
                warnings.warn(
                    "No data attached to this object and no time data provided. Plotting orbit will not work."
                )
                print(
                    "No data attached to this object and no time data provided. Plotting orbit will not work."
                )
                return
        else:
            timedata = self.data.time

        with self.pymc3_model:
            x, y, z = [
                eval_in_model(bla, point=self.pymc3_model.test_point)
                for bla in self.orbit.get_planet_position(timedata)
            ]  # {}
            plt.figure(figsize=(10, 3))
            theta = np.linspace(0, 2 * np.pi)
            plt.fill_between(np.cos(theta), np.sin(theta), color="gray")
            plt.scatter(x, y, c=timedata)
            plt.axis("scaled")
            plt.ylim(-1, 1)
            plt.show()
            plt.close()

    def plot_lightcurves(self, summary=None, trace=None, ax=None, **kw):
        if ax is None:
            ax = plt.subplot()
        plt.sca(ax)

        data = self.get_data()

        if summary is not None:
            params = [
                "period",
                "epoch",
                "impact_parameter",
                "stellar_radius",
                "stellar_mass",
                "stellar_mass",
                "limb_darkening",
                "radius_ratio",
            ]
            param_dict = {}
            posterior_means = summary["mean"]

            for p in params:
                if self.parameters[p] in posterior_means.index:
                    param_dict[p] = posterior_means[p]
                else:
                    param_dict[p] = self.parameters[p].value

            orbit = xo.orbits.KeplerianOrbit(
                period=param_dict["period"],
                t0=param_dict["epoch"],
                b=param_dict["impact_parameter"],
                r_star=param_dict["stellar_radius"],
                m_star=param_dict["stellar_mass"],
            )
            xo.LimbDarkLightCurve(param_dict["limb_darkening"]).get_light_curve(
                orbit=self.orbit,
                r=param_dict["radius_ratio"] * param_dict["stellar_radius"],
                t=list(data.time.to_value("day")),
            )

            if trace is not None:
                ndraws = 50
                posterior_traces = self.sample_posterior(trace, ndraws=ndraws)
                plt.plot(posterior_traces)
                for i in range(ndraws):
                    flux_for_this_sample = np.array(
                        [
                            posterior_traces[f"wavelength_{w}_data"][i]
                            for w in range(data.nwave)
                        ]
                    )
                    plt.plot(data.time, flux_for_this_sample)

        data.plot(ax=ax, **kw)
