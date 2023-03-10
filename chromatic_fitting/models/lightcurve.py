import warnings

from ..imports import *
from pymc3 import (
    sample_prior_predictive,
    sample_posterior_predictive,
    Deterministic,
    Normal,
    Uniform,
    TruncatedNormal,
    sample,
)
from pymc3_ext import eval_in_model, optimize
from pymc3_ext import sample as sample_ext

from arviz import summary
from ..parameters import *
from ..utils import *
from chromatic import *
import collections
from ..diagnostics import (
    chi_sq,
    generate_periodogram,
    check_rainbow,
    check_initial_guess,
)

allowed_types_of_models = ["planet", "systematic"]
import time
from functools import wraps
from tqdm import tqdm


def decorate_all_functions(function_decorator):
    def decorator(cls):
        for name, obj in vars(cls).items():
            if callable(obj):
                try:
                    obj = obj.__func__  # unwrap Python 2 unbound method
                except AttributeError:
                    pass  # not needed in Python 3
                setattr(cls, name, function_decorator(obj))
        return cls

    return decorator


def to_loop_for_separate_wavelength_fitting(func):
    @wraps(func)
    def wrapper(*args, **kw):
        if isinstance(args[0], LightcurveModels):
            for k, v in kw.items():
                setattr(args[0], k, v)
            return args[0].apply_operation_to_each_wavelength(func.__name__)(**kw)
        else:
            return func(*args, **kw)

    return wrapper


def my_timer(func):
    def _timer(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Execution time: {end - start}")
        return result

    return _timer


# @decorate_all_functions(print_on_call)
class LightcurveModel:
    """
    The base lightcurve model
    """

    required_parameters = []

    def __init__(self, name="lightcurve_model", **kw):
        """

        Parameters
        ----------
        name: the name of the model
        kw: additional keywords to pass ...
        """
        # define some default parameters (fixed):
        self.defaults = dict()
        # the default fitting method is simultaneous fitting
        self.optimization = "simultaneous"
        # by default do not store the lightcurve models
        self.store_models = False
        # assign the model a name (overwritten by the specific models)
        self.name = name
        # by default do not flag outliers
        self.outlier_flag = False

        self.initialize_empty_model()

    def __add__(self, other):
        """
        What should we return for `a + b` for two models `a` and `b`?
        """
        cm = CombinedModel()
        cm.initialize_empty_model()
        cm.combine(self, other, "+")

        return cm

    def __sub__(self, other):
        """
        What should we return for `a - b` for two models `a` and `b`?
        """
        cm = CombinedModel()
        cm.initialize_empty_model()
        cm.combine(self, other, "-")

        return cm

    def __mul__(self, other):
        """
        What should we return for `a * b` for two models `a` and `b`?
        """
        cm = CombinedModel()
        cm.initialize_empty_model()
        cm.combine(self, other, "*")

        return cm

    def __truediv__(self, other):
        """
        What should we return for `a / b` for two models `a` and `b`?
        """
        cm = CombinedModel()
        cm.initialize_empty_model()
        cm.combine(self, other, "/")
        return cm

    def set_name(self, name: str):
        """
        Set the name of the model.

        Parameters
        ----------
        name: name of the model
        """
        self.name = name

    def setup_parameters(self, **kw):
        """
        Set the values of the model parameters.

        Parameters
        ----------
        kw : object
            All keyword arguments will be treated as model
            parameters and stored in the `self.parameters`
            dictionary attribute. This input both sets
            the initial values for each parameter and
            indicates whether it should be fixed or fitted,
            and if it is fitted what its prior should be.
        """

        # set up a dictionary of unprocessed
        # print("RUNNING SETUP PARAMS")
        # print(self.defaults)
        unprocessed_parameters = dict(self.defaults)
        unprocessed_parameters.update(**kw)
        # print(unprocessed_parameters)

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
                self.parameters[k].set_name(k)

        # check that all the necessary parameters are defined somehow
        for k in self.required_parameters:
            assert k in self.parameters

        for k in self.parameters.keys():
            if self.name in k:
                warnings.warn(
                    f"{self.name} in the parameter name: {k}. Please avoid having the model name in the "
                    f"parameter name as it can get confusing!"
                )
                return
        self.parameters = add_string_before_each_dictionary_key(
            self.parameters, self.name
        )

        for v in self.parameters.values():
            v.set_name(f"{self.name}_{v.name}")

        self._original_parameters = self.parameters.copy()

    def get_parameter_shape(self, param):
        inputs = param.inputs
        for k in ["testval", "mu", "lower", "upper", "sigma"]:
            if k in inputs.keys():
                if (type(inputs[k]) == int) or (type(inputs[k]) == float):
                    inputs["shape"] = 1
                elif len(inputs[k]) == 1:
                    inputs["shape"] = 1
                else:
                    inputs["shape"] = len(inputs[k])
                break
        param.inputs = inputs

    def summarize_parameters(self):
        """
        Print a friendly summary of the parameters.
        """
        for k, v in self.parameters.items():
            print(f"{k} =\n  {v}\n")

    def reinitialize(self):
        self.__init__(name=self.name)
        self.reinitialize_parameters()

        for x in ["orbit", "every_light_curve"]:
            if hasattr(self, x):
                delattr(self, x)

    def reinitialize_parameters(self, exclude=[]):
        """
        Remove the pymc3 prior model from every parameter not in exclude
        """
        for k, v in self._original_parameters.copy().items():
            if k not in exclude:
                if isinstance(v, Fitted):
                    v.clear_prior()
                self.parameters[k] = v

    def extract_extra_class_inputs(self):
        """
        Extract any additional keywords passed to the LightcurveModel
        """
        class_inputs = {}
        varnames_to_remove = [
            "defaults",
            "optimization",
            "pymc3_model",
            "parameters",
            "required_parameters",
        ]
        for k, v in self.__dict__.items():
            if k not in varnames_to_remove:
                class_inputs[k] = v
        return class_inputs

    def initialize_empty_model(self):
        """
        Restart with an empty model.
        """
        self._pymc3_model = pm.Model()

    def copy(self):
        # make a "copy" of each model:
        class_inputs = self.extract_extra_class_inputs()
        new_model = self.__class__(**class_inputs)

        if isinstance(self, CombinedModel):
            if hasattr(self, "_chromatic_models"):
                new_model._chromatic_models = {}
                new_model.how_to_combine = self.how_to_combine
                for name, model in self._chromatic_models.items():
                    new_model._chromatic_models[name] = model.copy()

        # new_model._pymc3_model = self._pymc3_model
        model_params = {}

        # for every parameter in the separate models redefine them in the separate models within CombinedModel
        # and the new CombinedModel
        for k, v in self.parameters.items():
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
        new_model.defaults = add_string_before_each_dictionary_key(
            new_model.defaults, new_model.name
        )
        new_model.required_parameters = [
            f"{new_model.name}_{a}" for a in new_model.required_parameters
        ]
        new_model.setup_parameters(**model_params)

        return new_model

    def attach_data(self, rainbow):
        """
        Connect a `chromatic` Rainbow dataset to this object.
        """
        self.data = rainbow._create_copy()
        check_rainbow(self.data)

    def white_light_curve(self):
        """
        Generate inverse-variance weighted white light curve by binning Rainbow to one bin
        """
        self.white_light = self.data.bin(nwavelengths=self.data.nwave).trim()
        check_rainbow(self.white_light)

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
                    og_model = self.copy()
                    if hasattr(self, "data"):
                        og_model.data = self.data
                    self.__class__ = LightcurveModels
                    self.__init__(model=og_model)
                except Exception as e:
                    print(e)
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

        # self._pymc3_model = [pm.Model() for n in range(self.data.nwave)]
        # if the LightCurve model is a CombinedModel then update the constituent models too
        if isinstance(self, CombinedModel):
            for mod in self._chromatic_models.values():
                mod._pymc3_model = self._pymc3_model
                mod.optimization = self.optimization

    def change_all_priors_to_Wavelike(self):
        for k, v in self.parameters.items():
            if isinstance(v, Fitted) and not isinstance(v, WavelikeFitted):
                self.parameters[k] = WavelikeFitted(v.distribution, **v.inputs)
                self.parameters[k].set_name(k)
            if isinstance(v, Fixed) and not isinstance(v, WavelikeFixed):
                self.parameters[k] = WavelikeFixed([v.value] * self.data.nwave)
                self.parameters[k].set_name(k)

    def change_all_priors_to_notWavelike(self):
        for k, v in self.parameters.items():
            if isinstance(v, WavelikeFitted):
                self.parameters[k] = Fitted(v.distribution, **v.inputs)
                self.parameters[k].set_name(k)
            if isinstance(v, WavelikeFixed):
                self.parameters[k] = Fixed(v.values[0])
                self.parameters[k].set_name(k)

    def separate_wavelengths(self, i):
        """
        Extracts a single-wavelength Rainbow object from the attached Rainbow

        Parameters
        ----------
        i: the index of the wavelength to extract

        Returns
        -------
        Rainbow object for one wavelength of the original data

        """
        data_copy = self.data._create_copy()
        for k, v in data_copy.fluxlike.items():
            try:
                assert v.unit
                data_copy.fluxlike[k] = [data_copy.fluxlike[k][i, :]] * v.unit
            except AttributeError:
                data_copy.fluxlike[k] = np.array([data_copy.fluxlike[k][i, :]])
        for k, v in data_copy.wavelike.items():
            try:
                assert v.unit
                data_copy.wavelike[k] = [data_copy.wavelike[k][i]] * v.unit
            except AttributeError:
                data_copy.wavelike[k] = np.array([data_copy.wavelike[k][i]])
        return data_copy

    def get_data(self, i=None):
        """
        Extract the data to use for the optimization depending on the method chosen
        """
        if hasattr(self, "optimization"):
            if self.optimization == "white_light":
                if hasattr(self, "white_light"):
                    return self.white_light
                self.white_light_curve()
                return self.white_light
            else:
                if i is not None:
                    return self.separate_wavelengths(i)
                else:
                    return self.data

        return self.data

    def setup_priors(self, param_dict, **kw):
        for pname in param_dict.keys():
            param_dict[pname] = self.parameters[pname].get_prior_vector(**kw)
        return param_dict

    def extract_parameters_for_wavelength_i(self, param_dict, i):
        param_i = {}
        for pname, param in param_dict.items():
            if isinstance(self.parameters[pname], WavelikeFitted):
                param_i[pname] = param[i]
            else:
                param_i[pname] = param
        return param_i

    @to_loop_for_separate_wavelength_fitting
    def setup_likelihood(
        self,
        mask_outliers=False,
        mask_wavelength_outliers=False,
        sigma_wavelength=5,
        data_mask=None,
        inflate_uncertainties=False,
        setup_lightcurves_kw={},
        **kw,
    ):
        """
        Connect the light curve model to the actual data it aims to explain.
        """

        if not hasattr(self, "every_light_curve"):
            print(".setup_lightcurves() has not been run yet, running now...")
            self.setup_lightcurves(**setup_lightcurves_kw)

        data = self.get_data()
        mod = self._pymc3_model

        # if the data has outliers, then mask them out
        if mask_outliers:
            # if the user has specified a mask, then use that
            if data_mask is None:
                # sigma-clip in time
                data_mask = np.array(get_data_outlier_mask(data, **kw))
                if mask_wavelength_outliers:
                    # sigma-clip in wavelength
                    data_mask[
                        get_data_outlier_mask(
                            data, clip_axis="wavelength", sigma=sigma_wavelength
                        )
                        == True
                    ] = True
                # data_mask_wave =  get_data_outlier_mask(data, clip_axis='wavelength', sigma=4.5)
            self.outlier_mask = data_mask
            self.outlier_flag = True
            self.data_without_outliers = remove_data_outliers(data, data_mask)

        if inflate_uncertainties:
            self.parameters["nsigma"] = WavelikeFitted(
                Uniform, lower=1.0, upper=3.0, testval=1.01
            )
            self.parameters["nsigma"].set_name("nsigma")

        nsigma = []
        with mod:
            if inflate_uncertainties:
                nsigma.append(
                    self.parameters["nsigma"].get_prior_vector(shape=data.nwave)
                )
                uncertainty = [
                    np.array(data.uncertainty[i, :]) * nsigma[i]
                    for i in range(data.nwave)
                ]
                uncertainties = pm.math.stack(uncertainty)
            else:
                uncertainties = np.array(data.uncertainty)

            # if the user has passed mask_outliers=True then sigma clip and use the outlier mask
            if mask_outliers:
                flux = np.array(
                    [self.data_without_outliers.flux[i, :] for i in range(data.nwave)]
                )
            else:
                flux = np.array(data.flux)

            try:
                pm.Normal(
                    "data",
                    mu=self.every_light_curve,
                    sd=uncertainties,
                    observed=flux,
                )
            except Exception as e:
                print(e)

    @to_loop_for_separate_wavelength_fitting
    def sample_priors(self, ndraws=3):
        """
        Draw samples from the prior distribution.
        :parameter n
        Number of priors to sample
        """
        with self._pymc3_model:
            return sample_prior_predictive(ndraws)

    @to_loop_for_separate_wavelength_fitting
    def sample_posteriors(self, ndraws=3, var_names=None):
        """
        Draw samples from the posterior distribution.
        :parameter n
        Number of posteriors to sample
        """
        if not hasattr(self, "trace"):
            print("Sampling has not been run yet! Running now with defaults...")
            self.sample()

        if var_names is None:
            var_names = {**self.trace[0], **{"data": []}}.keys()

        with self._pymc3_model:
            return sample_posterior_predictive(self.trace, ndraws, var_names=var_names)

    @to_loop_for_separate_wavelength_fitting
    def optimize(self, plot=False, plotkw={}, **kw):
        """
        Wrapper for PyMC3_ext sample
        """
        check_initial_guess(self._pymc3_model)
        with self._pymc3_model:
            opt = optimize(**kw)
        if plot:
            self.plot_optimization(opt, **plotkw)
        self._map_soln = opt

        warnings.warn(
            "If you want to use this MAP-optimized solution as the start point in your sampling, you should run "
            "{self}.sample(use_optimized_start_point=True)!"
        )

        return opt

    def plot_optimization(self, map_soln, figsize=(12, 5), **kw):
        # datas, models, opts = self.choose_model_based_on_optimization_method(map_soln)
        data = self.get_data()
        model = self._pymc3_model

        # for i, (opt_sep, data, model) in enumerate(zip(opts, datas, models)):
        for j in range(data.nwave):
            # create a new plot for each wavelength
            plt.figure(figsize=figsize)
            plt.title(f"Wavelength: {data.wavelength[j]}")
            # plot the data and errors
            plt.plot(
                data.time,
                data.flux[j],
                "k.",
                alpha=0.3,
                ms=3,
                label="data",
            )
            plt.errorbar(
                data.time,
                data.flux[j],
                data.uncertainty[j],
                c="k",
                alpha=0.1,
                linestyle="None",
            )

            if hasattr(self, "initial_guess"):
                # plot the initial guess from priors
                # if f"wavelength_{i}" in self.initial_guess.keys():
                plt.plot(
                    self.data.time,
                    # pmx.eval_in_model(
                    self.initial_guess[j],
                    # )[j],
                    "C1--",
                    lw=1,
                    alpha=0.7,
                    label="Initial",
                )
            if hasattr(self, "every_light_curve"):
                # plot the final MAP-optimized solution (not sampled)
                # if f"wavelength_{i}" in self.every_light_curve.keys():
                plt.plot(
                    self.data.time,
                    pmx.eval_in_model(
                        self.every_light_curve[j],
                        map_soln,
                        model=model,
                    ),
                    "C1-",
                    label="MAP optimization",
                    lw=2,
                )
            plt.legend(fontsize=10, numpoints=5)
            plt.xlabel("time [days]", fontsize=24)
            plt.ylabel("relative flux", fontsize=24)

    def sample_individual(self, i, **kw):
        """
        Wrapper for PyMC3_ext sample - only for a single wavelength i
        """
        starts = []
        if "start" in kw:
            starts = kw["start"]
            kw.pop("start")

        with self._pymc3_model[i] as mod:
            if len(starts) > 0:
                start = starts[i]
            else:
                start = mod.test_point

            try:
                return sample(start=start, **kw)
            except Exception as e:
                print(f"Sampling failed for one of the models: {e}")
                return None

    @to_loop_for_separate_wavelength_fitting
    def sample(
        self,
        use_optimized_start_point=False,
        summarize_step_by_step=False,
        summarize_kw={"round_to": 7, "hdi_prob": 0.68, "fmt": "wide"},
        sampling_method=sample,
        sampling_kw={"init": "adapt_full"},
        **kw,
    ):
        """
        Wrapper for PyMC3_ext sample
        """
        print(f"Sampling model using the {sampling_method} method")
        check_initial_guess(self._pymc3_model)

        if use_optimized_start_point:
            if hasattr(self, "_map_soln"):
                kw["start"] = self._map_soln
                print("Using MAP-optimized start point...")
            else:
                warnings.warn(
                    "You have specified use_optimized_start_point=True, however, it seems {self}.optimize() "
                    "has not been run yet...\nRunning optimization step now..."
                )
                self.optimize()
                kw["start"] = self._map_soln

        with self._pymc3_model:
            self.trace = sampling_method(**sampling_kw, **kw)

        self.summarize(**summarize_kw)

    def summarize(self, hdi_prob=0.68, round_to=10, print_table=True, **kw):
        """
        Wrapper for arviz summary
        """
        if not hasattr(self, "trace"):
            print("Sampling has not been run yet! Running now with defaults...")
            self.sample()

        if hasattr(self, "summary"):
            print("Summarize has already been run")
            if print_table:
                print(self.summary)
            return

        with self._pymc3_model:
            self.summary = summary(
                self.trace, hdi_prob=hdi_prob, round_to=round_to, **kw
            )

        if print_table:
            print(self.summary)

        if isinstance(self, CombinedModel):
            for m in self._chromatic_models.values():
                m.summary = self.summary

    def get_results(self, i=None, as_df=True, uncertainty=["hdi_16%", "hdi_84%"]):
        """
        Extract mean results from summary
        """

        # if the user wants to have the same uncertainty for lower and upper error:
        if type(uncertainty) == str:
            uncertainty = [uncertainty, uncertainty]

        results = {}

        if self.optimization == "white_light":
            data = self.get_data()
        else:
            data = self.data

        if i is None:
            loops_i = range(data.nwave)
            loops_w = data.wavelength
        else:
            loops_i = [i]
            loops_w = data.wavelength

        for i, w in zip(loops_i, loops_w):
            params_mean = self.extract_from_posteriors(self.summary, i)
            params_lower_error = self.extract_from_posteriors(
                self.summary, i, op=uncertainty[0]
            )

            if params_lower_error is None:
                warnings.warn(f"{uncertainty[0]} is not in Summary table!")
                return

            params_lower_error = dict(
                (key + f"_{uncertainty[0]}", value)
                for (key, value) in params_lower_error.items()
            )
            params_upper_error = self.extract_from_posteriors(
                self.summary, i, op=uncertainty[1]
            )
            params_upper_error = dict(
                (key + f"_{uncertainty[1]}", value)
                for (key, value) in params_upper_error.items()
            )
            try:
                params = params_mean | params_lower_error | params_upper_error
            except TypeError:
                # for Python < 3.9 add dictionaries using a different method
                params = {**params_mean, **params_lower_error, **params_upper_error}
            ordered_params = collections.OrderedDict(sorted(params.items()))
            results[f"w{i}"] = ordered_params
            results[f"w{i}"]["wavelength"] = w

        if as_df:
            # if the user wants to return the results as a pandas DataFrame:
            return pd.DataFrame(results).transpose()
        else:
            # otherwise return a dictionary of dictionaries
            return results

    def run_simultaneous_fit(self, r, **kwargs):
        """
        Run the entire simultaneous wavelength fit.
        """
        self.attach_data(r)
        self.setup_lightcurves()
        self.setup_likelihood()
        opt = self.optimize(start=self._pymc3_model.test_point)
        opt = self.optimize(start=opt)
        self.sample(start=opt)
        self.summarize(round_to=7, fmt="wide")

    @to_loop_for_separate_wavelength_fitting
    def plot_priors(self, n=3, quantity="data", plot_all=True):
        """
        Plot n prior samples from the parameter distributions defined by the user
        :parameter n
        Number of priors to plot (default=3)
        """
        data = self.get_data()
        prior_predictive_trace = self.sample_priors(ndraws=n)

        for i in range(n):
            try:
                flux_for_this_sample = np.array(prior_predictive_trace[quantity][i])
            except:
                warnings.warn(f"Couldn't generate prior for {quantity}!")
                return

            if f"{self.name}_model" in prior_predictive_trace.keys():
                model_for_this_sample = np.array(
                    prior_predictive_trace[f"{self.name}_model"][i]
                )
            else:
                model_for_this_sample = []
                for w in range(data.nwave):
                    i_dict = {}
                    for k, v in prior_predictive_trace.items():
                        if np.shape(v) == 1:
                            i_dict = {**i_dict, **{k: v[i]}}
                        elif np.shape(v)[1] == 1:
                            i_dict = {**i_dict, **{k: v[i][0]}}
                        else:
                            i_dict = {**i_dict, **{k: v[i][w]}}
                    model_for_this_sample.append(self.model(params=i_dict, i=w))
                model_for_this_sample = np.array(model_for_this_sample)

            # add posterior model and draw from posterior distribution to the Rainbow quantities:
            data.fluxlike[f"prior-model-{i}"] = model_for_this_sample
            data.fluxlike[f"prior-predictive-{i}"] = flux_for_this_sample

        if plot_all:
            data.imshow_quantities()
        else:
            f, ax = plt.subplots(nrows=n, sharex=True, sharey=True, figsize=(6, 4 * n))
            for i in range(n):
                if f"{self.name}_model" in prior_predictive_trace.keys():
                    data.imshow(ax=ax[i], quantity=f"prior-model-{i}")
                else:
                    data.imshow(ax=ax[i], quantity=f"prior-predictive-{i}")

    @to_loop_for_separate_wavelength_fitting
    def plot_posteriors(self, n=3, quantity="data", plot_all=True):
        """
        Plot n posterior samples from the parameter distributions defined by the user
        :parameter trace
        PyMC3 trace object (need to run sample first)
        :parameter n
        Number of posteriors to plot (default=3)
        """
        data = self.get_data()
        posterior_predictive_trace = self.sample_posteriors(n)

        posterior_model = {}
        for i in range(n):
            # for every posterior sample extract the posterior model and distribution draw:
            try:
                flux_for_this_sample = np.array(posterior_predictive_trace[quantity][i])
            except:
                warnings.warn(f"Couldn't generate prior for {quantity}!")
                return

            if f"{self.name}_model" in posterior_predictive_trace.keys():
                model_for_this_sample = np.array(
                    posterior_predictive_trace[f"{self.name}_model"][i]
                )
            else:
                model_for_this_sample = []
                for w in range(data.nwave):
                    i_dict = {}
                    for k, v in posterior_predictive_trace.items():
                        if np.shape(v) == 1:
                            i_dict = {**i_dict, **{k: v[i]}}
                        elif np.shape(v)[1] == 1:
                            i_dict = {**i_dict, **{k: v[i][0]}}
                        else:
                            i_dict = {**i_dict, **{k: v[i][w]}}
                    model_for_this_sample.append(self.model(params=i_dict, i=w))
                model_for_this_sample = np.array(model_for_this_sample)

            # add posterior model and draw from posterior distribution to the Rainbow quantities:
            data.fluxlike[f"posterior-model-{i}"] = model_for_this_sample
            data.fluxlike[f"posterior-predictive-{i}"] = flux_for_this_sample

        if plot_all:
            data.imshow_quantities()
        else:
            f, ax = plt.subplots(nrows=n, sharex=True, sharey=True, figsize=(6, 4 * n))
            for i in range(n):
                if f"{self.name}_model" in posterior_model.keys():
                    data.imshow(ax=ax[i], quantity=f"posterior-model-{i}")
                else:
                    data.imshow(ax=ax[i], quantity=f"posterior-predictive-{i}")

    def extract_deterministic_model(self):
        """
        Extract the deterministic model from the summary statistics
        """
        data = self.get_data()

        model = {}
        summary = self.summary

        for w in range(data.nwave):
            # if self.optimization == "separate":
            #     summary = self.summary[w]
            # else:
            if f"w{w}" not in model.keys():
                model[f"w{w}"] = []

            if f"{self.name}_model[{w}, 0]" in summary["mean"].keys():
                for t in range(data.ntime):
                    model[f"w{w}"].append(
                        summary["mean"][f"{self.name}_model[{w}, {t}]"]
                    )
            elif f"{self.name}_model[0]" in summary["mean"].keys():
                for t in range(data.ntime):
                    model[f"w{w}"].append(summary["mean"][f"{self.name}_model[{t}]"])
        return model

    def get_stored_model(self, as_dict=True, as_array=False):
        """
        Return the 'best-fit' model from the summary table as a dictionary or as an array

        Parameters
        ----------
        as_dict: Boolean whether to return a dictionary (default=True)
        as_array: Boolean whether to return a dictionary (default=False)

        Returns
        -------
        model as either a dictionary or array

        """
        model = self.extract_deterministic_model()

        if as_array:
            return np.array(list(model.values()))
        elif as_dict:
            return model

    def plot_lightcurves(self, t_unit="day", ax=None, **kw):
        """
        Plot the 2D lightcurves for each wavelength separated on the same plot

        Parameters
        ----------
        t_unit: Unit for the time series (default="day")
        ax: Axis object to plot onto (default=None)
        kw: Any extra keywords to pass to chromatic.Rainbow.plot()

        Returns
        -------

        """

        data = self.get_data()

        if not hasattr(self, "summary"):
            print(
                "The summarize step has not been run yet. To include the 'best-fit' model please run "
                "{self}.sample() and {self}.summarize() before calling this step!"
            )
            add_model = False
        else:
            if isinstance(self, CombinedModel):
                model = self.get_model()
            elif isinstance(self, LightcurveModels):
                if isinstance(self._og_model, CombinedModel):
                    model = self.get_model()
                else:
                    model = {}
                    for i in range(data.nwave):
                        model[f"w{i}"] = {"total": self.get_model()[f"w{i}"]}
            else:
                model = {}
                for i in range(data.nwave):
                    model[f"w{i}"] = {"total": self.get_model()[f"w{i}"]}
            add_model = True

        # if the user has provided an axis object then use, otherwise create one here
        if ax is None:
            ax = plt.subplot()
        plt.sca(ax)

        # if we've flagged outliers then show these on the plot in red
        if self.outlier_flag:
            data.plot(ax=ax, cmap="Reds", **kw)
            self.data_without_outliers.plot(ax=ax, **kw)
        else:
            data.plot(ax=ax, **kw)

        # get the lightcurve spacing from chromatic
        spacing = ax._most_recent_chromatic_plot_spacing

        # if add_model=True then overplot the fitted model
        if add_model:
            for i in range(data.nwave):
                # print(i, model.keys())
                ax.plot(
                    data.time.to_value(t_unit),
                    np.array(model[f"w{i}"]["total"]) - (i * spacing),
                    color="k",
                )

        # if filename is provided then save the plot
        if "filename" not in kw.keys():
            plt.show()
        else:
            plt.savefig(kw["filename"])
        plt.close()

    def extract_from_posteriors(self, summary, i, op="mean"):
        # ensure that the specified operation is in the summary table!
        if op not in summary:
            print(
                f"Try a different operation or regenerate the summary table! "
                f"{op} is not currently in the summary table"
            )
            return None

        posterior_means = summary[op]
        fv = {}
        for k, v in self.parameters.items():
            more_than_one_input = False
            if isinstance(v, Fitted):
                if type(v.inputs["shape"]) == int:
                    if v.inputs["shape"] != 1:
                        more_than_one_input = True
                elif v.inputs["shape"] != (self.data.nwave, 1):
                    more_than_one_input = True

            if more_than_one_input:
                if f"{k}[{i}]" in posterior_means.index and "limb_darkening" not in k:
                    fv[k] = posterior_means[f"{k}[{i}]"]
                elif f"{k}[0]" in posterior_means.index:
                    n = 0
                    fv[k] = []
                    while f"{k}[{n}]" in posterior_means.index:
                        fv[k].append(posterior_means[f"{k}[{n}]"])
                        n += 1
                    if n == 1:
                        fv[k] = fv[k][0]
                elif f"{k}[{i}, 0]" in posterior_means.index:
                    n = 0
                    fv[k] = []
                    while f"{k}[{i}, {n}]" in posterior_means.index:
                        fv[k].append(posterior_means[f"{k}[{i}, {n}]"])
                        n += 1
                    if n == 1:
                        fv[k] = fv[k][0]
                elif f"{k}[0, 0]" in posterior_means.index:
                    n = 0
                    fv[k] = []
                    while f"{k}[0, {n}]" in posterior_means.index:
                        fv[k].append(posterior_means[f"{k}[0, {n}]"])
                        n += 1
                    if n == 1:
                        fv[k] = fv[k][0]
            else:
                if k in posterior_means.index:
                    fv[k] = posterior_means[k]
                elif f"{k}[{i}]" in posterior_means.index:
                    fv[k] = posterior_means[f"{k}[{i}]"]
                elif f"{k}[0]" in posterior_means.index:
                    fv[k] = posterior_means[f"{k}[0]"]
                elif f"{k}[{i}, 0]" in posterior_means.index:
                    fv[k] = posterior_means[f"{k}[{i}, 0]"]
                else:
                    if isinstance(v, WavelikeFixed):
                        fv[k] = v.values[0]
                    elif isinstance(v, Fixed):
                        fv[k] = v.value

        return fv

    def corner_plot(self, **kw):
        try:
            import corner
        except ImportError:
            warnings.warn(
                "corner is not installed, please install corner before trying this method!"
            )
            return

        if not hasattr(self, "trace"):
            print("Sampling has not been run yet! Running now with defaults...")
            self.sample()

        with self._pymc3_model:
            _ = corner.corner(self.trace, **kw)

    def check_and_fill_missing_parameters(self, params, i):
        if all([f"{self.name}_" in rp for rp in self.required_parameters]):
            name = ""
        else:
            name = f"{self.name}_"

        for rp in self.required_parameters:
            if f"{name}{rp}" not in params.keys():
                if isinstance(self.parameters[f"{name}{rp}"], WavelikeFixed):
                    params[f"{name}{rp}"] = self.parameters[f"{name}{rp}"].values[i]
                elif isinstance(self.parameters[f"{name}{rp}"], Fixed):
                    params[f"{name}{rp}"] = self.parameters[f"{name}{rp}"].value
                else:
                    warnings.warn(
                        f"{name}{rp} is missing from the parameter dictionary!"
                    )
        return params

    @to_loop_for_separate_wavelength_fitting
    def get_model(
        self,
        params_dict: dict = None,
        as_dict: bool = True,
        as_array: bool = False,
        store: bool = True,
    ):
        """
        Return the 'best-fit' model from the summary table as a dictionary or as an array

        Parameters
        ----------
        params_dict: [optional] dictionary of parameters with which to generate model
        as_dict: boolean whether to return the model as a dictionary (with keys indexing the wavelength)
        as_array: boolean whether to return the model as an array
        store: boolean whether to save the model

        Returns
        -------
        object: model for each wavelength (either a dict or array)
        """

        if self.store_models:
            # if we decided to store the LC model extract this now
            if store:
                self._fit_models = self.get_stored_model(
                    as_dict=as_dict, as_array=as_array
                )
            return self.get_stored_model(as_dict=as_dict, as_array=as_array)
        else:
            # if we decided not to store the LC model then generate the model
            model = {}
            # generate the transit model from the best fit parameters for each wavelength
            if self.optimization == "white_light":
                data = self.get_data()
            else:
                data = self.data

            for i, wave in enumerate(range(data.nwave)):
                if params_dict is None:
                    if hasattr(self, "summary"):
                        params = self.extract_from_posteriors(self.summary, i)
                    else:
                        warnings.warn(
                            "You haven't sampled/summarized the model yet!\nEither run .summarize() or pass "
                            "a dictionary of parameters to generate the model."
                        )
                        return
                else:
                    # if the user has given the same parameters for each wavelength
                    if f"w{i}" not in params_dict.keys():
                        params = params_dict
                    else:
                        params = params_dict[f"w{i}"]

                model_i = self.model(params=params, i=i)
                model[f"w{i}"] = model_i

            if store:
                self._fit_models = model
            if as_array:
                # return a 2D array (one row for each wavelength)
                return np.array(list(model.values()))
            elif as_dict:
                # return a dict (one key for each wavelength)
                return model

    def add_model_to_rainbow(self):
        """
        Add the model to the Rainbow object.
        """
        # if we decided to flag outliers then flag these in the final model
        if self.outlier_flag:
            data = self.data_without_outliers
        else:
            data = self.data

        # if optimization method is "white_light" then extract the white light curve
        if self.optimization == "white_light":
            data = self.white_light

        # extract model as an array
        model = self.get_model(as_array=True)

        # attach the model to the Rainbow (creating a Rainbow_with_model object)
        if isinstance(self, PolynomialModel):
            r_with_model = data.attach_model(model=model, systematics_model=model)
        elif isinstance(self, TransitModel):
            r_with_model = data.attach_model(model=model, planet_model=model)
        else:
            warnings.warn(
                f"This {self} model isn't a type that chromatic_fitting knows how to attach to a Rainbow!"
            )
        # save the Rainbow_with_model for later
        self.data_with_model = r_with_model

    def imshow_with_models(self, **kw):
        """
        Plot the lightcurves with the best-fit models (if they exist).
        [Wrapper for chromatic]
        """
        if not hasattr(self, "data_with_model"):
            print(
                "No model attached to data. Running `add_model_to_rainbow` now. You can access this data later"
                " using [self].data_with_model"
            )
            self.add_model_to_rainbow()
        self.data_with_model.imshow_with_models(**kw)

    def animate_with_models(self, **kw):
        """
        Animate the lightcurves with the best-fit models (if they exist).
        [Wrapper for chromatic]
        """
        if not hasattr(self, "data_with_model"):
            print(
                "No model attached to data. Running `add_model_to_rainbow` now. You can access this data later"
                " using [self].data_with_model"
            )
            self.add_model_to_rainbow()
        self.data_with_model.animate_with_models(**kw)

    def plot_with_model_and_residuals(self, **kw):
        """
        Plot the lightcurves with the best-fit models (if they exist) and residuals.
        [Wrapper for chromatic]

        Parameters
        ----------
        kw: Any additional keywords to pass to Rainbow.data_with_model.plot_with_model_and_residuals

        Returns
        -------

        """
        if not hasattr(self, "data_with_model"):
            print(
                "No model attached to data. Running `add_model_to_rainbow` now. You can access this data later"
                " using [self].data_with_model"
            )
            self.add_model_to_rainbow()
        self.data_with_model.plot_with_model_and_residuals(**kw)

    @to_loop_for_separate_wavelength_fitting
    def plot_model(self, normalize=True, plot_data=True, wavelength=None, **kw):
        """

        Parameters
        ----------
        normalize: Boolean to normalize all models to mean~1, some models will have default mean~0 (default=True)
        plot_data: Boolean to add real data on top of model plots (default=True)
        kw: Any additional keywords to pass to plot_one_model(), e.g. wavelength, ax

        Returns
        -------

        """
        model = self.get_model()

        def plot_models_for_each_w(ax, mod, name_label, normalize):
            if normalize:
                if np.nanmedian(mod) < 0.1:
                    ax.plot(self.data.time, mod + 1, label=f"{name_label}")
                else:
                    ax.plot(self.data.time, mod, label=f"{name_label}")
            else:
                ax.plot(self.data.time, mod, label=f"{name_label}")

        def plot_one_model(model, normalize, wavelength, name_label=None, **kw):
            wave_num = [int(k.replace("w", "")) for k in model.keys()]

            if wavelength is not None:
                try:
                    model_one_wavelength = {}
                    if type(wavelength) == int:
                        i = wavelength
                        if f"w{i}" in model.keys():
                            model_one_wavelength[f"w{i}"] = model[f"w{i}"]
                        else:
                            return
                    #                     else:
                    #                         model_one_wavelength[f"w{i}"] = model["w0"]
                    else:
                        for i in wavelength:
                            if f"w{i}" in model.keys():
                                model_one_wavelength[f"w{i}"] = model[f"w{i}"]
                        if len(model_one_wavelength.values()) == 0:
                            return
                    model = model_one_wavelength
                except:
                    return

            if "ax" in kw:
                ax = kw["ax"]
            else:
                if len(wave_num) == 1 or wavelength is not None:
                    figsize = (8, 4)
                else:
                    figsize = (8, 16)

                if wavelength is not None:
                    fi, ax = plt.subplots(
                        nrows=len(wave_num),
                        figsize=figsize,
                        constrained_layout=True,
                    )
                else:
                    # make sure ax is set up
                    fi, ax = plt.subplots(
                        nrows=len(wave_num),
                        figsize=(8, 4),
                        constrained_layout=True,
                    )
                if len(wave_num) == 1:
                    ax = [ax]
                kw["ax"] = ax

            if name_label is None:
                name_label = self.name

            for wnum, ax_i, mod in zip(wave_num, ax, model.values()):
                if isinstance(self, CombinedModel):
                    list_of_chrom_mods = list(self._chromatic_models.keys())
                    list_of_chrom_mods.append("total")
                    for chrom_mod in list_of_chrom_mods:
                        plot_models_for_each_w(
                            ax_i, mod[chrom_mod], chrom_mod, normalize
                        )
                else:
                    plot_models_for_each_w(ax_i, mod, name_label, normalize)

                if plot_data:
                    if wnum > 0 and self.data.nwave == 1:
                        flux = self.data.flux[0, :]
                        uncertainty = self.data.uncertainty[0, :]
                    else:
                        flux = self.data.flux[wnum, :]
                        uncertainty = self.data.uncertainty[wnum, :]

                    ax_i.plot(self.data.time, flux, "k.")
                    ax_i.errorbar(
                        self.data.time,
                        flux,
                        uncertainty,
                        color="k",
                        linestyle="None",
                        capsize=2,
                    )
                ax_i.set_title(f"Wavelength {wnum}")
                ax_i.set_ylabel("Relative Flux")
                ax_i.set_xlabel("Time [d]")
                ax_i.legend()

            return kw

        plot_one_model(model, normalize=normalize, wavelength=wavelength, **kw)

    def chi_squared(self, individual_wavelengths=False, **kw):
        """
        Calculate the chi-square and reduced chi-square of the model fits (a useful "goodness-of-fit" parameter)

        Parameters
        ----------
        individual_wavelengths: Boolean for whether to consider each wavelength individually or all at once
        kw: Any additional keywords to pass to statistics.chi_sq()

        Returns
        -------

        """
        if hasattr(self, "data_with_model"):
            if self.optimization != "white_light":
                if self.store_models:
                    summary = self.summary.iloc[
                        ~self.summary.index.str.contains(f"{self.name}_model")
                    ]
                else:
                    summary = self.summary
                fit_params = len(summary)
                degrees_of_freedom = (self.data.nwave * self.data.ntime) - fit_params
                print("\nFor Entire Simultaneous Fit:")
                print("Fitted Parameters:\n", ", ".join(summary.index))
                print(
                    f"\nDegrees of Freedom = n_waves ({self.data.nwave}) * n_times ({self.data.ntime}) - n_fitted_parameters ({fit_params}) = {degrees_of_freedom}"
                )
                chi_sq(
                    data=self.data_with_model.flux,
                    model=self.data_with_model.model,
                    uncertainties=self.data_with_model.uncertainty,
                    degrees_of_freedom=degrees_of_freedom,
                    **kw,
                )

                if individual_wavelengths:
                    count_wave_fit_params, count_nonwave_fit_params = 0, 0
                    wave_fit_params, nonwave_fit_params = [], []
                    for p in self.parameters.values():
                        try:
                            if type(p.inputs["shape"]) == int:
                                if p.inputs["shape"] > 1:
                                    count_wave_fit_params += 1
                                    wave_fit_params.append(p.name)
                                else:
                                    count_nonwave_fit_params += 1
                                    nonwave_fit_params.append(p.name)
                            else:
                                if p.inputs["shape"][0] > 1:
                                    count_wave_fit_params += p.inputs["shape"][1]
                                    for i in range(p.inputs["shape"][1]):
                                        wave_fit_params.append(p.name + f"_{i}")
                        except:
                            pass
                    fit_params = count_wave_fit_params + (
                        count_nonwave_fit_params / self.data.nwave
                    )
                    degrees_of_freedom = self.data.ntime - fit_params

                    for i in range(self.data.nwave):
                        print(f"\nFor Wavelength {i}:")
                        print(
                            "Wavelength Fitted Parameters:\n",
                            ", ".join(wave_fit_params),
                        )
                        print(
                            "Non-Wavelength Fitted Parameters:\n",
                            ", ".join(nonwave_fit_params),
                        )
                        print(
                            f"\nDegrees of Freedom = n_times ({self.data.ntime}) - n_fitted_parameters ({fit_params}) = {degrees_of_freedom}"
                        )
                        chi_sq(
                            data=self.data_with_model.flux[i],
                            model=self.data_with_model.model[i],
                            uncertainties=self.data_with_model.uncertainty[i],
                            degrees_of_freedom=degrees_of_freedom,
                            **kw,
                        )
            else:
                if self.store_models:
                    summary = self.summary.iloc[
                        ~self.summary.index.str.contains(f"{self.name}_model")
                    ]
                else:
                    summary = self.summary
                fit_params = len(summary)
                degrees_of_freedom = self.data.ntime - fit_params
                chi_sq(
                    data=self.data_with_model.flux,
                    model=self.data_with_model.model,
                    uncertainties=self.data_with_model.uncertainty,
                    degrees_of_freedom=degrees_of_freedom,
                    **kw,
                )

        else:
            warnings.warn(
                f"""Could not find .data_with_model in {self.name} model! 
            Please run [self].add_model_to_rainbow() or one of the plotting methods
            with models before rerunning this method."""
            )

    def plot_residuals_histogram(self, **kw):
        """
        Plot histogram of the residuals

        Parameters
        ----------
        kw: Any additional keywords to pass to matplotlib.pyplot.hist

        Returns
        -------

        """
        if hasattr(self, "data_with_model"):
            plt.figure(figsize=(8, 6))
            for i in range(self.data.nwave):
                plt.hist(
                    self.data_with_model.residuals[i],
                    alpha=0.5,
                    label=f"Wavelength {i}",
                    histtype="step",
                    **kw,
                )
            plt.hist(
                np.mean(self.data_with_model.residuals, axis=0),
                color="k",
                label=f"Mean Wavelength",
                histtype="step",
            )
            plt.legend()
            plt.show()
            plt.close()
        else:
            warnings.warn(
                f"""Could not find .data_with_model in {self.name} model! 
                Please run [self].add_model_to_rainbow() or one of the plotting methods
                with models before rerunning this method."""
            )

    def plot_residuals(self, **kw):
        """
        Plot the residuals from the model fits

        Parameters
        ----------
        kw: Any additional keywords to pass to matplotlib.pyplot.plot

        Returns
        -------

        """
        if hasattr(self, "data_with_model"):
            data = self.data_with_model
            plt.figure(figsize=(12, 6))
            for i in range(data.nwave):
                plt.plot(
                    data.time,
                    data.residuals[i],
                    alpha=0.5,
                    label=f"Wavelength {i}",
                    **kw,
                )
            plt.plot(
                data.time,
                np.mean(data.residuals, axis=0),
                color="k",
                label=f"Mean Wavelength",
            )
            plt.legend()
            plt.show()
            plt.close()
        else:
            warnings.warn(
                f"""Could not find .data_with_model in {self.name} model! 
                Please run [self].add_model_to_rainbow() or one of the plotting methods
                with models before rerunning this method."""
            )

    def residual_noise_calculator(self, individual_wavelengths=False, **kw):
        """
        Wrapper for utils.noise_calculator: Hannah Wakeford's code to calculate the noise parameters of the data by
        using the residuals of the fit

        Parameters
        ----------
        individual_wavelengths: Boolean for whether to consider each wavelength individually or just the overall noise
        kw: Any additional keywords to pass to the noise_calculator function (e.g. maxnbins, binstep, figname)

        Returns
        -------

        """
        if hasattr(self, "data_with_model"):
            data = self.data_with_model
            print("For the Wavelength-Averaged Residuals...")
            noise_calculator(np.mean(data.residuals, axis=0), **kw)
            if individual_wavelengths:
                for i in range(data.nwave):
                    print(f"\nFor wavelength {i}")
                    noise_calculator(data.residuals[i], **kw)
        else:
            warnings.warn(
                f"""Could not find .data_with_model in {self.name} model! 
                Please run [self].add_model_to_rainbow() or one of the plotting methods
                with models before rerunning this method."""
            )

    def plot_residuals_periodogram(self, **kw):
        """
        Plot a periodogram of the residuals using scipy.signal.periodogram

        Parameters
        ----------
        kw: any additional keywords to pass to scipy.signal.periodogram

        Returns
        -------

        """
        if hasattr(self, "data_with_model"):
            data = self.data_with_model
            plt.figure(figsize=(12, 6))
            plt.title("Periodogram of Wavelength-Averaged Residuals")
            generate_periodogram(
                x=np.mean(data.residuals, axis=0),
                fs=(1 / (data.time[1] - data.time[0])).to_value("1/d"),
                **kw,
            )
        else:
            warnings.warn(
                f"""Could not find .data_with_model in {self.name} model! 
                Please run [self].add_model_to_rainbow() or one of the plotting methods
                with models before rerunning this method."""
            )


# @decorate_all_functions(print_on_call)
class LightcurveModels(LightcurveModel):
    def __init__(self, model):
        self.models = {}
        self.optimization = "separate"
        self.name = model.name
        self._og_model = model
        self.outlier_flag = False
        self.parameters = model.parameters
        self.type_of_model = model.type_of_model
        self._list_of_methods = list(dir(self))
        self._list_of_og_methods = list(dir(model))

        if hasattr(model, "data"):
            self.attach_data(model.data)

    def __repr__(self):
        """
        Print the model
        """
        if hasattr(self, "nwave"):
            return f'<chromatic models ({self.nwave} separate wavelengths) "{self.name}" 🌈>'
        else:
            return f'<chromatic models (unknown number of separate wavelengths) "{self.name}" 🌈>'

    def __getattr__(self, item):
        if item not in self._list_of_methods:
            if item in self._list_of_og_methods:

                def make_wrap(**kw):
                    op = getattr(self._og_model.__class__, item)
                    return op(self, **kw)

                return make_wrap
                # return self.apply_operation_to_constituent_models(operation=item)
            else:
                # warnings.warn(f"{item} doesn't seem to exist for this model.")
                raise AttributeError
        else:
            return self.item

    def setup_separate_structure(self, wavelengths):
        self.wavelengths = wavelengths
        self.nwave = len(wavelengths)
        for w in range(self.nwave):
            self.models[f"w{w}"] = self._og_model.copy()
            if isinstance(self._og_model, CombinedModel):
                pm_model = pm.Model()
                for name, chromatic_model in self._og_model._chromatic_models.items():
                    self.models[f"w{w}"]._pymc3_model = pm_model
                    self.models[f"w{w}"]._chromatic_models[name]._pymc3_model = pm_model
            else:
                self.models[f"w{w}"]._pymc3_model = pm.Model()
            self.models[f"w{w}"].name = self.name
            self._pymc3_model = [m._pymc3_model for m in self.models.values()]

    def apply_operation_to_each_wavelength(self, operation: str) -> object:
        """
        Apply an operation to all models within LightcurveModels

        Parameters
        ----------
        operation: string name of the operation to carry out
        args: arguments to pass to the operation
        kwargs: keywords to pass to the operation

        Returns
        -------
        object
        """

        def wrapper_for_function(*args, **kwargs):
            results = []
            # for each constituent model apply the chosen operation
            for name, model in tqdm(self.models.items()):
                try:
                    op = getattr(model, operation)
                    result = op(*args, **kwargs)
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    print(f"Error applying {operation} to {model}: {e}")

            if len(results) == 0:
                return None
            else:
                return results

        return wrapper_for_function

    def separate_wavelengths(self, i):
        data_copy = self.data._create_copy()
        for k, v in data_copy.fluxlike.items():
            try:
                assert v.unit
                data_copy.fluxlike[k] = [data_copy.fluxlike[k][i, :]] * v.unit
            except AttributeError:
                data_copy.fluxlike[k] = np.array([data_copy.fluxlike[k][i, :]])
        for k, v in data_copy.wavelike.items():
            try:
                assert v.unit
                data_copy.wavelike[k] = [data_copy.wavelike[k][i]] * v.unit
            except AttributeError:
                data_copy.wavelike[k] = np.array([data_copy.wavelike[k][i]])
        return data_copy

    def attach_data(self, r: chromatic.Rainbow):
        """
        Connect a `chromatic` Rainbow dataset to this object and the constituent models.

        Parameters
        ----------
        r: Rainbow object with the light curve data

        """
        self.setup_separate_structure(r.wavelength)
        self.data = r._create_copy()

        for i in range(self.nwave):
            data_copy = self.separate_wavelengths(i)
            if isinstance(self.models[f"w{i}"], CombinedModel):
                self.models[f"w{i}"].attach_data(data_copy)
            else:
                self.models[f"w{i}"].data = data_copy

    def setup_lightcurves(self, **kw):
        """
        Set-up lightcurves
        """
        if not hasattr(self, "nwave"):
            self.setup_separate_structure(self.data.wavelength)
        # self.store_models = store_models
        self._og_model.__class__.setup_lightcurves(self, **kw)
        # self.apply_operation_to_constituent_models(operation="setup_lightcurves")(
        #     store_models=store_models, **kw
        # )

    # def plot_model(self, **kw):
    #     # self.get_model()
    #     self._og_model.__class__.plot_model(self, **kw)

    def get_model(self, **kw):
        if hasattr(self, "_fit_models") and "as_array" not in kw.keys():
            return self._fit_models
        else:
            models_list = self._og_model.__class__.get_model(self, **kw)
            # models_list = self.apply_operation_to_constituent_models("get_model")(**kw)
            if "as_array" in kw.keys():
                if kw["as_array"] == True:
                    models_array = np.array([m[0] for m in models_list])
                    self._fit_models = models_array
                    return models_array
            models_dict = {}
            for i, (name, model) in enumerate(self.models.items()):
                models_dict[name] = model._fit_models["w0"]
                model._fit_models = {name: model._fit_models["w0"]}
                # self.models[name]._fit_models = {name: model._fit_models["w0"]}
            self._fit_models = models_dict
            return models_dict

    def sample(self, **kw):
        """
        Sample parameters using NUTS MCMC
        """
        self._og_model.__class__.sample(self, **kw)
        # self.apply_operation_to_constituent_models("sample")(**kw)
        self.recombine_summaries()
        self.get_model()

    def recombine_one_line_dfs(self, attribute):
        for i, name, model in enumerate(self.models.items()):
            if name == "w0":
                attribute_to_combine = getattr(self, attribute).copy()
            else:
                one_line_df = getattr(self, attribute).copy()
                one_line_df.index = [
                    s.replace("[0", f"[{i}") for s in one_line_df.index.values
                ]
                one_line_df.index = [
                    s.replace("w0", f"w{i}") for s in one_line_df.index.values
                ]
                attribute_to_combine = pd.concat([attribute_to_combine, one_line_df])
        return attribute_to_combine

    # def recombine_spectrum_table(self, list_of_spec):
    #     for i, ((name, model), transspec) in enumerate(
    #         zip(self.models.items(), list_of_spec)
    #     ):
    #         if name == "w0":
    #             spec = transspec.copy()
    #         else:
    #             sp = transspec.copy()
    #             sp.index = [s.replace("0", f"{i}") for s in sp.index.values]
    #             spec = pd.concat([spec, sp])
    #     return spec

    def make_transmission_spectrum_table(self, **kw):
        if isinstance(self._og_model, TransitModel):
            return TransitModel.make_transmission_spectrum_table(self, **kw)
        elif isinstance(self._og_model, CombinedModel):
            for mod in self._og_model._chromatic_models.values():
                if isinstance(mod, TransitModel):
                    spec = TransitModel.make_transmission_spectrum_table(
                        self, name=mod.name, **kw
                    )
                    return spec
            warnings.warn(
                f"You cannot make a transmission spectrum table for a non-transit model!"
                f" Your model is: {self._og_model}"
            )
        else:
            warnings.warn(
                f"You cannot make a transmission spectrum table for a non-transit model!"
                f" Your model is: {self._og_model}"
            )

    def plot_transmission_spectrum(self, **kw):
        if hasattr(self, "transmission_spectrum"):
            transmission_table = self.transmission_spectrum
        else:
            transmission_table = self.make_transmission_spectrum_table()

        if isinstance(self._og_model, TransitModel):
            return TransitModel.plot_transmission_spectrum(
                self, table=transmission_table, **kw
            )
        elif isinstance(self._og_model, CombinedModel):
            for mod in self._og_model._chromatic_models.values():
                if isinstance(mod, TransitModel):
                    return TransitModel.plot_transmission_spectrum(
                        self, table=transmission_table, name=mod.name, **kw
                    )
        else:
            warnings.warn(
                f"You cannot plot a transmission spectrum table for a non-transit model!"
                f" Your model is: {self._og_model}"
            )

    def recombine_summaries(self, **kw):
        """ """
        for i, (name, model) in enumerate(self.models.items()):
            if name == "w0":
                summaries = model.summary.copy()
            else:
                summary = model.summary.copy()
                summary.index = [s.replace("[0", f"[{i}") for s in summary.index.values]
                summaries = pd.concat([summaries, summary])
        self.summary = summaries

    def get_results(self, **kw):
        """
        Get results from summaries
        """
        for name, model in self.models.items():
            if name == "w0":
                a = model.get_results()
            else:
                b = model.get_results()
                b.index = [name]
                a = pd.concat([a, b])
        return a

    def add_model_to_rainbow(self):
        # self.get_model()
        self._og_model.__class__.add_model_to_rainbow(self)
        # self.apply_operation_to_constituent_models("add_model_to_rainbow")()

        total_model, systematics, planet = [], [], []
        for name, model in self.models.items():
            if hasattr(model, "data_with_model"):
                total_model.append(model.data_with_model.fluxlike["model"][0])
                if "systematics_model" in model.data_with_model.fluxlike.keys():
                    systematics.append(
                        model.data_with_model.fluxlike["systematics_model"][0]
                    )
                if "planet_model" in model.data_with_model.fluxlike.keys():
                    planet.append(model.data_with_model.fluxlike["planet_model"][0])

        if len(total_model) > 0:
            if len(systematics) > 0 and len(planet) > 0:
                self.data_with_model = self.data.attach_model(
                    model=np.array(total_model),
                    systematics_model=np.array(systematics),
                    planet_model=np.array(planet),
                )
            elif len(systematics) > 0:
                self.data_with_model = self.data.attach_model(
                    model=np.array(total_model), systematics_model=np.array(systematics)
                )
            elif len(planet) > 0:
                self.data_with_model = self.data.attach_model(
                    model=np.array(total_model), planet_model=np.array(planet)
                )
            else:
                warnings.warn("WARNING: No models to attach!")


from .combined import *
