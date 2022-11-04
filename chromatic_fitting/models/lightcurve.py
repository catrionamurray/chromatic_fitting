import warnings

from ..imports import *
from pymc3 import (
    sample_prior_predictive,
    sample_posterior_predictive,
    Deterministic,
    Normal,
    TruncatedNormal,
)
from pymc3_ext import eval_in_model, optimize, sample
from arviz import summary
from ..parameters import *
from ..utils import *
from chromatic import *
import collections

#  - Q=sqrt(N)*depth/error (Winn 2010/ Carter 2008)
#  - chi-sq

# chi_sq = []
# nparams = 15
# for i in range(cmod.data.nwave-1):
#     chi_sq.append(np.sum((cmod.data_with_model.residuals[i,:]/cmod.data.uncertainty[i,:])**2)/(cmod.data.ntime-numparams-1))
#     if chi_sq[-1] > 2:
#         print(cmod.data.wavelength[i], chi_sq[-1])
# chi_sq

# nparams = 0
# for k, v in self._chromatic_models.items():
#     for pname, p in v.parameters.items():
#         if isinstance(p, Fixed) or isinstance(p, WavelikeFixed):
#             pass
#         elif isinstance(p, WavelikeFitted):
#             nparams += cmod.data.nwave
#
#             if "limb_darkening" in pname:
#                 nparams += 1
#         else:
#
#             if "limb_darkening" in pname:
#                 nparams += 1
#             nparams += 1


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
        kw : dict
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
                self.parameters[k].set_name(v)

        # check that all the necessary parameters are defined somehow
        for k in self.required_parameters:
            assert k in self.parameters

        for k in self.parameters.keys():
            if self.name in k:
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
        varnames_to_remove = ["defaults", "optimization", "pymc3_model", "parameters"]
        for k, v in self.__dict__.items():
            if k not in varnames_to_remove:
                class_inputs[k] = v
        return class_inputs

    def initialize_empty_model(self):
        """
        Restart with an empty model.
        """
        self._pymc3_model = pm.Model()

    def attach_data(self, rainbow):
        """
        Connect a `chromatic` Rainbow dataset to this object.
        """
        self.data = rainbow._create_copy()

    def white_light_curve(self):
        """
        Generate inverse-variance weighted white light curve by binning Rainbow to one bin
        """
        # if self.outlier_flag:
        #     data = self.data_without_outliers
        # else:
        #     data = self.data
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
        self._pymc3_model = [pm.Model() for n in range(self.data.nwave)]
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
        # if self.outlier_flag:
        #     data_copy = self.data_without_outliers._create_copy()
        # else:
        data_copy = self.data._create_copy()

        for k, v in data_copy.fluxlike.items():
            data_copy.fluxlike[k] = np.array([data_copy.fluxlike[k][i, :]])
        for k, v in data_copy.wavelike.items():
            data_copy.wavelike[k] = [data_copy.wavelike[k][i]]
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
            if self.optimization == "separate":
                if i is not None:
                    return self.separate_wavelengths(i)
                else:
                    return self.data
                    # return [
                    #     self.separate_wavelengths(i) for i in range(self.data.nwave)
                    # ]
        # if self.outlier_flag:
        #     return self.data_without_outliers
        # else:
        return self.data

    def setup_likelihood(
        self,
        mask_outliers=False,
        mask_wavelength_outliers=False,
        sigma_wavelength=5,
        data_mask=None,
        inflate_uncertainties=False,
        **kw,
    ):
        """
        Connect the light curve model to the actual data it aims to explain.
        """
        # data = self.get_data()
        self.bad_wavelengths = []

        if self.optimization == "separate":
            models = self._pymc3_model
            datas = [self.get_data(i) for i in range(self.data.nwave)]
            data = self.data
        else:
            models = [self._pymc3_model]
            data = self.get_data()
            datas = [data]

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
                TruncatedNormal, mu=1, sd=0.1, lower=1
            )
            self.parameters["nsigma"].set_name("nsigma")

        for j, (mod, data) in enumerate(zip(models, datas)):
            with mod:
                uncertainties, flux = [], []
                for i, w in enumerate(data.wavelength):
                    # k = f"wavelength_{j + i}"

                    if inflate_uncertainties:
                        uncertainties.append(
                            data.uncertainty[i, :]
                            * eval_in_model(self.parameters["nsigma"].get_prior(j + i))
                        )
                    else:
                        uncertainties.append(data.uncertainty[i, :])

                    # try:
                    # if the user has passed mask_outliers=True then sigma clip and use the outlier mask
                    if mask_outliers:
                        flux.append(self.data_without_outliers.flux[i + j, :])
                    else:
                        flux.append(data.flux[i, :])

                try:
                    # if self.optimization == "separate":
                    #     data_name = f"data_w{j}"
                    #     light_curve_name = f"wavelength_{j}"
                    # else:
                    data_name = f"data"  # _w{j}"
                    light_curve_name = f"wavelength_{j}"

                    pm.Normal(
                        data_name,
                        mu=self.every_light_curve[light_curve_name],
                        sd=np.array(uncertainties),
                        observed=np.array(flux),
                    )
                    # pm.Normal(f"{k}_data",
                    #         mu=self.every_light_curve[k],
                    #         sd=uncertainties,
                    #         observed=flux,
                    # )
                except Exception as e:
                    print(f"Setting up likelihood failed for wavelength {i}: {e}")
                    self.bad_wavelengths.append(i)

    def sample_prior(self, ndraws=3):
        """
        Draw samples from the prior distribution.
        :parameter n
        Number of priors to sample
        """
        try:
            with self._pymc3_model:
                return sample_prior_predictive(ndraws)
        except:
            priors = []
            for mod in self._pymc3_model:
                with mod:
                    priors.append(sample_prior_predictive(ndraws))
            return priors

    def sample_posterior(self, ndraws=3, var_names=None):
        """
        Draw samples from the posterior distribution.
        :parameter n
        Number of posteriors to sample
        """
        if not hasattr(self, "trace"):
            print("Sampling has not been run yet! Running now with defaults...")
            self.sample()

        if self.optimization != "separate":
            with self._pymc3_model:
                return sample_posterior_predictive(
                    self.trace, ndraws, var_names=var_names
                )
        else:
            posteriors = []
            for mod, trace in zip(self._pymc3_model, self.trace):
                with mod:
                    posteriors.append(
                        sample_posterior_predictive(trace, ndraws, var_names=var_names)
                    )
            return posteriors

    def optimize(self, plot=True, plotkw={}, **kw):
        """
        Wrapper for PyMC3_ext sample
        """
        if self.optimization == "separate":
            opts = []
            if "start" in kw:
                start = kw["start"]
                kw.pop("start")
                for mod, opt in zip(self._pymc3_model, start):
                    with mod:
                        opts.append(optimize(start=opt, **kw))
            else:
                for mod in self._pymc3_model:
                    with mod:
                        opts.append(optimize(**kw))
            if plot:
                self.plot_optimization(opts, **plotkw)
            return opts
        else:
            with self._pymc3_model:
                opt = optimize(**kw)
            if plot:
                self.plot_optimization(opt, **plotkw)
            return opt

    def plot_optimization(self, opt, offset=0.03, figsize=(6, 18)):
        if self.optimization == "separate":
            opts = opt
            datas = [self.get_data(i) for i in range(self.data.nwave)]
        else:
            opts = [opt]
            datas = [self.get_data()]

        plt.figure(figsize=figsize)
        for j, (opt_sep, data) in enumerate(zip(opts, datas)):
            for w in range(data.nwave):
                try:
                    if w == 0:
                        plt.plot(
                            data.time,
                            ((w + j) * offset) + opt_sep[f"{self.name}_model_w{w + j}"],
                            "k",
                            label=self.name,
                        )
                    else:
                        plt.plot(
                            data.time,
                            ((w + j) * offset) + opt_sep[f"{self.name}_model_w{w + j}"],
                            "k",
                        )

                    if isinstance(self, CombinedModel):
                        if w == 0:
                            for mod in self._chromatic_models.values():
                                plt.plot(
                                    data.time,
                                    ((w + j) * offset)
                                    + opt_sep[f"{mod.name}_model_w{w + j}"],
                                    label=mod.name,
                                )
                        else:
                            for mod in self._chromatic_models.values():
                                plt.plot(
                                    data.time,
                                    ((w + j) * offset)
                                    + opt_sep[f"{mod.name}_model_w{w + j}"],
                                )
                except:
                    pass
                plt.plot(data.time, ((w + j) * offset) + data.flux[w, :], "k.")
                plt.errorbar(
                    data.time,
                    ((w + j) * offset) + data.flux[w, :],
                    data.uncertainty[w, :],
                    color="k",
                    linestyle="None",
                    capsize=2,
                )
        plt.legend()
        plt.show()

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

    def sample(self, summarize_step_by_step=False, summarize_kw={}, **kw):
        """
        Wrapper for PyMC3_ext sample
        """
        if self.optimization == "separate":
            self.trace = []
            starts = []
            if "start" in kw:
                starts = kw["start"]
                kw.pop("start")

            for i, mod in enumerate(self._pymc3_model):
                if self.optimization == "separate":
                    print(f"\nSampling for Wavelength: {i}")
                with mod:
                    if len(starts) > 0:
                        start = starts[i]
                    else:
                        start = mod.test_point

                    try:
                        samp = sample(start=start, **kw)
                        if summarize_step_by_step:
                            self.summary.append(summary(samp, **summarize_kw))
                        else:
                            self.trace.append(samp)

                    except Exception as e:
                        print(f"Sampling failed for one of the models: {e}")
                        self.trace.append(None)

            if summarize_step_by_step:
                if isinstance(self, CombinedModel):
                    for m in self._chromatic_models.values():
                        m.summary = self.summary

            # for mod in self._pymc3_model:
            #     with mod:
            #         try:
            #             self.trace.append(sample(**kw))
            #         except Exception as e:
            #             print(f"Sampling failed for one of the models!: {e}")
            #             self.trace.append(None)
        else:
            with self._pymc3_model:
                self.trace = sample(**kw)

    def summarize(self, print_table=True, **kw):
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

        if self.optimization == "separate":
            self.summary = []
            for mod, trace in zip(self._pymc3_model, self.trace):
                with mod:
                    self.summary.append(summary(trace, **kw))
        else:
            with self._pymc3_model:
                self.summary = summary(self.trace, **kw)

        if print_table:
            print(self.summary)

        if isinstance(self, CombinedModel):
            for m in self._chromatic_models.values():
                m.summary = self.summary

    def get_results(self, as_df=True, uncertainty=["hdi_3%", "hdi_97%"]):
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

        for i, w in enumerate(data.wavelength):
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
            params = params_mean | params_lower_error | params_upper_error
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

    # def remove_data_outliers(self, **kw):
    #     """
    #     Remove outliers from the data.
    #     [Ideally to be replaced with some chromatic.flag_outliers() function]
    #     """
    #     data_outliers_removed = remove_data_outliers(self.data, **kw)
    #     self.data_outliers_removed = data_outliers_removed
    #     self.outlier_flag = True

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
                    [prior_predictive_trace[f"data"][i] for w in range(data.nwave)]
                )
                # model_for_this_sample = np.array(
                #     [
                #         prior_predictive_trace[f"{self.name}_model_w{w + nm}"][i]
                #         for w in range(data.nwave)
                #     ]
                # )
                # f"{name}model_w{i + j}"
                data.fluxlike[f"prior-predictive-{i}"] = flux_for_this_sample
                # data.fluxlike[f"prior-model-{i}"] = model_for_this_sample
            data.imshow_quantities()

    def plot_posteriors(self, n=3):
        """
        Plot n posterior samples from the parameter distributions defined by the user
        :parameter trace
        PyMC3 trace object (need to run sample first)
        :parameter n
        Number of posteriors to plot (default=3)
        """
        # if we have the separate wavelength optimization method chosen then repeat for every wavelength/model
        if self.optimization == "separate":
            datas = [self.get_data(i) for i in range(self.data.nwave)]
            posterior_predictive_traces = self.sample_posterior(n)
        else:
            datas = [self.get_data()]
            posterior_predictive_traces = [self.sample_posterior(n)]

        posterior_model = {}
        for nm, (data, posterior_predictive_trace) in enumerate(
            zip(datas, posterior_predictive_traces)
        ):

            for w in range(data.nwave):
                if "data" in posterior_predictive_trace.keys():
                    # generate a posterior model for every wavelength:
                    if (
                        f"{self.name}_model_w{w + nm}"
                        in posterior_predictive_trace.keys()
                    ):
                        posterior_model[f"{self.name}_model"] = self.sample_posterior(
                            n, var_names=[f"{self.name}_model"]
                        )[f"{self.name}_model"]

            for w in range(data.nwave):
                for i in range(n):
                    # for every posterior sample extract the posterior model and distribution draw for every wavelength:
                    flux_for_this_sample = np.array(
                        [
                            posterior_predictive_trace[f"data"][i]
                            for w in range(data.nwave)
                        ]
                    )

                    if f"{self.name}_model" in posterior_model.keys():
                        model_for_this_sample = np.array(
                            [
                                posterior_model[f"{self.name}_model"][i]
                                for w in range(data.nwave)
                            ]
                        )

                        # add posterior model and draw from posterior distribution to the Rainbow quantities:
                        data.fluxlike[f"posterior-model-{i}"] = model_for_this_sample
                        data.fluxlike[
                            f"posterior-predictive-{i}"
                        ] = flux_for_this_sample
            # plot the rainbow quantities:
            data.imshow_quantities()

    def extract_deterministic_model(self):
        """
        Extract the deterministic model from the summary statistics
        """
        if self.optimization == "separate":
            datas = [self.get_data(i) for i in range(self.data.nwave)]
        else:
            datas = [self.get_data()]

        model = {}
        for nm, data in enumerate(datas):
            for w in range(data.nwave):
                if self.optimization == "separate":
                    summary = self.summary[w + nm]
                else:
                    summary = self.summary

                if f"w{w + nm}" not in model.keys():
                    model[f"w{w + nm}"] = []
                for t in range(data.ntime):
                    model[f"w{w + nm}"].append(
                        summary["mean"][f"{self.name}_model[{w}, {t}]"]
                    )
        return model

    def get_stored_model(self, as_dict=True, as_array=False):
        """
        Return the 'best-fit' model from the summary table as a dictionary or as an array
        """
        model = self.extract_deterministic_model()

        if as_array:
            return np.array(list(model.values()))
        elif as_dict:
            return model

    def plot_lightcurves(self, detrend=False, t_unit="day", ax=None, **kw):
        if not hasattr(self, "summary"):
            print(
                "The summarize step has not been run yet. To include the 'best-fit' model please run "
                "{self}.sample() and {self}.summarize() before calling this step!"
            )
            add_model = False
        else:
            if isinstance(self, CombinedModel):
                model = self.get_model()
            else:
                model = {"total": self.get_model()}
            add_model = True

        if ax is None:
            ax = plt.subplot()
        plt.sca(ax)

        # if self.optimization == "separate":
        #     datas = [self.get_data(i) for i in range(self.data.nwave)]
        # else:
        #     datas = [self.get_data()]
        if self.optimization != "separate":
            data = self.get_data()
        else:
            data = self.data

        if self.outlier_flag:
            data.plot(ax=ax, cmap="Reds", **kw)
            self.data_without_outliers.plot(ax=ax, **kw)
        else:
            data.plot(ax=ax, **kw)

        # for data in datas:
        spacing = ax._most_recent_chromatic_plot_spacing

        if add_model:
            for i in range(data.nwave):
                ax.plot(
                    data.time.to_value(t_unit),
                    model["total"][f"w{i}"] - (i * spacing),
                    color="k",
                )
        plt.show()
        plt.close()

        # if add_model and detrend:

    def extract_from_posteriors(self, summary, i, op="mean"):
        # there's definitely a sleeker way to do this
        if self.optimization == "separate":
            summary = summary[i]

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
                    # n = 0
                    # fv[k] = []
                    # while f"{k}[{i}, {n}]" in posterior_means.index:
                    #     fv[k].append(posterior_means[f"{k}[{i}, {n}]"])
                    #     n += 1
                    # if n == 1:
                    #     fv[k] = fv[k][0]
                # elif f"{k}_w{i}" in posterior_means.index:
                #     fv[k] = posterior_means[f"{k}_w{i}"]
                # elif f"{k}_w{i}[0]" in posterior_means.index:
                #     n = 0
                #     fv[k] = []
                #     while f"{k}_w{i}[{n}]" in posterior_means.index:
                #         fv[k].append(posterior_means[f"{k}_w{i}[{n}]"])
                #         n += 1
                # elif f"{k}_w{i}" in posterior_means.index:
                #     fv[k] = posterior_means[f"{k}_w{i}"]
                else:
                    if isinstance(v, WavelikeFixed):
                        fv[k] = v.values[0]
                    elif isinstance(v, Fixed):
                        fv[k] = v.value

        return fv

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

        # if the optimization method is "separate" then loop over each wavelength's data
        # if self.optimization == "separate":
        #     datas = [self.get_data(i) for i in range(self.data.nwave)]
        # else:
        #     # data = [self.get_data()]
        #     data = self.get_data()
        # datas = [data[i, :] for i in range(data.nwave)]

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
            # for j, data in enumerate(datas):
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

                model_i = self.model(params, i)
                # if is instance(self, TransitModel):
                #     model_i = self.transit_model(params, i)
                # elif isinstance(self, PolynomialModel):
                #     model_i = self.polynomial_model(params, i)
                # else:
                #     warnings.warn(
                #         f"{self} doesn't have a defined model in lightcurve.py"
                #     )
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
        """
        if not hasattr(self, "data_with_model"):
            print(
                "No model attached to data. Running `add_model_to_rainbow` now. You can access this data later"
                " using [self].data_with_model"
            )
            self.add_model_to_rainbow()
        self.data_with_model.plot_with_model_and_residuals(**kw)

    def plot_model(self, normalize=True, plot_data=True, **kw):
        model = self.get_model()

        def plot_one_model(model, normalize, name_label=None, **kw):
            if "wavelength" in kw:
                wave_num = []
                model_one_wavelength = {}
                if type(kw["wavelength"]) == int:
                    i = kw["wavelength"]
                    wave_num.append(i)
                    model_one_wavelength[f"w{i}"] = model[f"w{i}"]
                else:
                    for i in kw["wavelength"]:
                        wave_num.append(i)
                        model_one_wavelength[f"w{i}"] = model[f"w{i}"]
                model = model_one_wavelength

            else:
                wave_num = range(self.data.nwave)

            if "ax" in kw:
                ax = kw["ax"]
            else:
                if "wavelength" not in kw:
                    fi, ax = plt.subplots(
                        nrows=len(wave_num),
                        figsize=(8, 16),
                        constrained_layout=True,
                    )
                else:
                    # make sure ax is set up
                    fi, ax = plt.subplots(
                        nrows=len(wave_num),
                        figsize=(8, 6),
                        constrained_layout=True,
                    )
                if len(wave_num) == 1:
                    ax = [ax]
                kw["ax"] = ax

            if name_label is None:
                name_label = self.name

            for wave_n, ax_i, mod in zip(wave_num, ax, model.values()):
                if normalize:
                    if np.nanmedian(mod) < 0.1:
                        ax_i.plot(self.data.time, mod + 1, label=f"{name_label}")
                    else:
                        ax_i.plot(self.data.time, mod, label=f"{name_label}")
                else:
                    ax_i.plot(self.data.time, mod, label=f"{name_label}")

                if plot_data:
                    ax_i.plot(self.data.time, self.data.flux[wave_n, :], "k.")
                    ax_i.errorbar(
                        self.data.time,
                        self.data.flux[wave_n, :],
                        self.data.uncertainty[wave_n, :],
                        color="k",
                        linestyle="None",
                        capsize=2,
                    )
                ax_i.set_title(f"Wavelength {wave_n}")
                ax_i.set_ylabel("Relative Flux")
                ax_i.set_xlabel("Time [d]")
                ax_i.legend()

            return kw

        if isinstance(self, CombinedModel):
            for chrom_mod in self._chromatic_models.keys():
                kw = plot_one_model(
                    model[chrom_mod], normalize=normalize, name_label=chrom_mod, **kw
                )
            plot_one_model(
                model["total"], normalize=normalize, name_label="total", **kw
            )
        else:
            plot_one_model(model, normalize=normalize, **kw)


from .combined import *
