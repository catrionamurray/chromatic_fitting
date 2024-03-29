
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


class LightcurveModel:
    """
    The base lightcurve model
    """

    def __init__(self, name="lightcurve_model", **kw):
        """

        Parameters
        ----------
        name: the name of the model (default='lightcurve_model')
        kw: additional keywords to pass
        """
        required_parameters = []
        self.defaults = dict()
        """The default fixed parameters"""
        self.optimization = "simultaneous"
        """The fitting method, default is simultaneous fitting"""
        self.store_models = False
        """Whether to store the lightcurve models, default is False"""
        self.name = name
        """The model name (overwritten by the specific models)"""
        self.outlier_flag = False
        """Whether to flag outliers, default is False (WARNING: OUTLIER FLAGGING STILL IN BETA TESTING)"""
        self.has_gp = False
        """Whether the model includes a Gaussian Process, by default False"""

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

        # set up a dictionary of unprocessed parameters
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
                # if type(v) == np.float64 or type(v) == float or type(v) == int:
                self.parameters[k] = Fixed(v)
                # else:
                #     self.parameters[k] = WavelikeFixed(v)
                self.parameters[k].set_name(k)

        # check that all the necessary parameters are defined somehow
        for k in self.required_parameters:
            assert k in self.parameters

        for k in self.parameters.keys():
            if self.name in k:
                # warnings.warn(
                #     f"{self.name} in the parameter name: {k}. Please avoid having the model name in the "
                #     f"parameter name as it can get confusing!"
                # )
                return
        self.parameters = add_string_before_each_dictionary_key(
            self.parameters, self.name
        )

        for v in self.parameters.values():
            v.set_name(f"{self.name}_{v.name}")

        self._original_parameters = self.parameters.copy()

    def get_parameter_shape(self, param):
        """
        Set-up the shape of a specified parameter

        Parameters
        ----------
        param: Parameter to set up the shape of

        Returns
        -------

        """
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

    def which_steps_have_been_run(self):
        """
        Print a friendly summary of which fitting stages have been run
        """
        parameters_setup = False
        data_attached = False
        setup_lightcurves_done = False
        setup_likelihood_done = False
        sampling_done = False

        if hasattr(self, "parameters"):
            parameters_setup = True
            print("✅ The chromatic model has parameters")
        else:
            print("❌ The chromatic model does not have parameters")
        if hasattr(self, "data"):
            data_attached = True
            print("✅ The chromatic model has data attached")
        else:
            print("❌ The chromatic model does not have data attached")
        if hasattr(self, "every_light_curve"):
            setup_lightcurves_done = True
            print("✅ The PyMC3 lightcurve model has been set up")
        else:
            print("❌ The PyMC3 model lightcurve has not been set up")
        if hasattr(self, "_pymc3_model"):
            if len(self._pymc3_model.observed_RVs) > 0:
                if self._pymc3_model.observed_RVs[0].name == "data":
                    print("✅ The likelihood function has been set up")
                else:
                    print("❌ The likelihood function has not been set up")
            else:
                print("❌ The likelihood function has not been set up")
        else:
            print("❌ The likelihood function has not been set up")
        if hasattr(self, "summary"):
            sampling_done = True
            print("✅ The PyMC3 model has been sampled")
        else:
            print("❌ The PyMC3 model has not been sampled")


    def reinitialize(self):
        """
        Reinitialize model (incl. parameters and pymc3 model)

        Returns
        -------

        """
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
        Restart with an empty PyMC3 model.
        """
        self._pymc3_model = pm.Model()

    def copy(self):
        """
         make a "copy" of each lightcurve model:

        :return:
        copy of lightcurve model
        """

        class_inputs = self.extract_extra_class_inputs()
        new_model = self.__class__(**class_inputs)

        if isinstance(self, CombinedModel):
            if hasattr(self, "_chromatic_models"):
                new_model._chromatic_models = {}
                new_model.how_to_combine = self.how_to_combine
                for name, model in self._chromatic_models.items():
                    new_model._chromatic_models[name] = model.copy()

                for name, mod in new_model._chromatic_models.items():
                    mod.__dict__ = self[name].__dict__.copy()

        new_model.__dict__ = self.__dict__.copy()

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
        # if self.outlier_flag:
        #     data = self.data_without_outliers
        # else:
        #     data = self.data
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
        """
        Change all parameter priors to Wavelike
        i.e. Change Fitted to WavelikeFitted and Fixed to WavelikeFixed
        """
        for k, v in self.parameters.items():
            if isinstance(v, Fitted) and not isinstance(v, WavelikeFitted):
                self.parameters[k] = WavelikeFitted(v.distribution, **v.inputs)
                self.parameters[k].set_name(k)
            if isinstance(v, Fixed) and not isinstance(v, WavelikeFixed):
                self.parameters[k] = WavelikeFixed([v.value] * self.data.nwave)
                self.parameters[k].set_name(k)

    def change_all_priors_to_notWavelike(self):
        """
        Change all Wavelike parameter priors to not Wavelike
        i.e. Change WavelikeFitted to Fitted and WavelikeFixed to Fixed
        """
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

    def choose_model_based_on_optimization_method(self, *extra_arrs):
        """
        Return the necessary format for the data, model and any extra arrays, according to the optimization method used.

        Parameters
        ----------
        extra_arrs: Any extra arrays needed to reformat

        Returns
        ----------
        datas: reformated data,
        models: reformated models,
        *extra_arrs: reformated arrays
        """
        extra_arrs_new = []
        if self.optimization == "separate":
            datas = [self.get_data(i) for i in range(self.data.nwave)]
            models = self._pymc3_model
            for arrs in extra_arrs:
                extra_arrs_new.append(arrs)
        else:
            datas = [self.get_data()]
            models = [self._pymc3_model]
            for arrs in extra_arrs:
                extra_arrs_new.append([arrs])
        return datas, models, *extra_arrs_new

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
        inflate_uncertainties_prior=WavelikeFitted(Uniform, lower=1.0, upper=3.0, testval=1.01),
        setup_lightcurves_kw={},
        **kw,
    ):
        """
        Connect the light curve model to the actual data it aims to explain.

        Parameters
        ----------
        mask_outliers: Boolean whether to mask outliers in time (WARNING: STILL IN BETA TESTING), \n
        mask_wavelength_outliers: Boolean whether to mask outliers in wavelength (WARNING: STILL IN BETA TESTING),
        sigma_wavelength: Number of sigmas for sigma-clipping in wavelength (WARNING: STILL IN BETA TESTING),
        data_mask: User-defined outlier mask (WARNING: STILL IN BETA TESTING),
        inflate_uncertainties: Boolean whether or not to inflate the uncertainties by a fitted factor `nsigma`,
        inflate_uncertainties_prior: Prior on `nsigma` (default = WavelikeFitted(Uniform, lower=1.0, upper=3.0, testval=1.01)),
        setup_lightcurves_kw: keywords to pass to self.setup_lightcurves()
        kw: other keywords to pass to data masking

        Returns
        -------
        """

        if hasattr(self, "every_light_curve"):
            if f"wavelength_0" not in self.every_light_curve.keys():
                print(".setup_lightcurves() has not been run yet, running now...")
                self.setup_lightcurves(**setup_lightcurves_kw)
        else:
            print(".setup_lightcurves() has not been run yet, running now...")
            self.setup_lightcurves(**setup_lightcurves_kw)

        datas, models = self.choose_model_based_on_optimization_method()

        # if the data has outliers, then mask them out
        if mask_outliers:
            # if the user has specified a mask, then use that
            if data_mask is None:
                # sigma-clip in time
                data_mask = np.array(get_data_outlier_mask(datas, **kw))
                if mask_wavelength_outliers:
                    # sigma-clip in wavelength
                    data_mask[
                        get_data_outlier_mask(
                            datas, clip_axis="wavelength", sigma=sigma_wavelength
                        )
                        == True
                    ] = True
                # data_mask_wave =  get_data_outlier_mask(data, clip_axis='wavelength', sigma=4.5)
            self.outlier_mask = data_mask
            self.outlier_flag = True
            self.data_without_outliers = remove_data_outliers(self.get_data(), data_mask)

        if inflate_uncertainties:
            self.parameters["nsigma"] = inflate_uncertainties_prior
            self.parameters["nsigma"].set_name("nsigma")

        nsigma = []
        if self.has_gp:
            self.gp = []
        for j, (mod, data) in enumerate(zip(models, datas)):
            with mod:
                if inflate_uncertainties:
                    nsigma.append(
                        self.parameters["nsigma"].get_prior_vector(
                            i=j, shape=datas[0].nwave
                        )
                    )
                    uncertainty = [
                        np.array(data.uncertainty[i, :]) * nsigma[j][i]
                        for i in range(data.nwave)
                    ]
                    uncertainties = pm.math.stack(uncertainty)
                else:
                    uncertainties = np.array(data.uncertainty)
                #                     uncertainties.append(data.uncertainty[i, :])

                # if the user has passed mask_outliers=True then sigma clip and use the outlier mask
                if mask_outliers:
                    flux = np.array(
                        [
                            self.data_without_outliers.flux[i + j, :]
                            for i in range(data.nwave)
                        ]
                    )
                else:
                    flux = np.array(data.flux)

                try:
                    data_name = f"data"
                    light_curve_name = f"wavelength_{j}"

                    if self.has_gp == False:
                        pm.Normal(
                            data_name,
                            mu=self.every_light_curve[light_curve_name],
                            sd=uncertainties,
                            observed=flux,
                        )
                    else:
                        if type(self.independant_variable) is not list:
                            x = data.get(self.independant_variable)
                            if self.independant_variable == "time":
                                x = x.to_value("day")
                        else:
                            warnings.warn("GP models use different independent variables - this does not work yet!!")

                        for i in range(data.nwave):
                            if len(np.shape(x)) > 1:
                                xi = x[i, :]
                            else:
                                xi = x
                            self.gp.append(
                                GaussianProcess(
                                    self.covar[light_curve_name][i],
                                    t=xi,
                                    diag=uncertainties[i] ** 2 + pm.math.exp(self.log_jitter.get_prior(i + j)),
                                    mean=self.mean.get_prior(i + j),
                                    quiet=True,
                                )
                            )

                            self.gp[-1].marginal(
                                f"gp_w{i + j}",
                                observed=flux[i] - self.every_light_curve[light_curve_name][i],
                            )


                except Exception as e:
                    print(e)

    #                 print(f"Setting up likelihood failed for wavelength {i}: {e}")
    #                 self.bad_wavelengths.append(i)

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

        if var_names is None:
            var_names = {**self.trace[0], **{"data": []}}.keys()

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

    def optimize(self, plot=False, plotkw={}, **kw):
        """
        Wrapper for PyMC3_ext sample
        """
        if self.optimization == "separate":
            opts = []
            if "start" in kw:
                start = kw["start"]
                kw.pop("start")
                for mod, opt in zip(self._pymc3_model, start):
                    check_initial_guess(mod)
                    with mod:
                        opts.append(optimize(start=opt, **kw))
            else:
                for mod in self._pymc3_model:
                    check_initial_guess(mod)
                    with mod:
                        opts.append(optimize(**kw))
            if plot:
                self.plot_optimization(opts, **plotkw)
            return opts
        else:
            check_initial_guess(self._pymc3_model)
            with self._pymc3_model:
                opt = optimize(**kw)
            if plot:
                self.plot_optimization(opt, **plotkw)
            return opt

    def plot_optimization(self, map_soln, figsize=(12, 5), **kw):
        datas, models, opts = self.choose_model_based_on_optimization_method(map_soln)

        for i, (opt_sep, data, model) in enumerate(zip(opts, datas, models)):
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
                    if f"wavelength_{i}" in self.initial_guess.keys():
                        plt.plot(
                            self.data.time,
                            self.initial_guess[f"wavelength_{i}"][j],
                            "C1--",
                            lw=1,
                            alpha=0.7,
                            label="Initial",
                        )

                if hasattr(self, "every_light_curve"):
                    # plot the final MAP-optimized solution (not sampled)
                    if f"wavelength_{i}" in self.every_light_curve.keys():
                        opt_y = pmx.eval_in_model(
                            self.every_light_curve[f"wavelength_{i}"],
                            opt_sep,
                            model=model,
                        )[j]

                        plt.plot(
                            self.data.time,
                            opt_y,
                            "C1-",
                            label="MAP optimization",
                            lw=2,
                            alpha=0.7
                        )

                        if hasattr(self, 'has_gp'):
                            if self.has_gp:
                                gp_model = self.generate_gp_model_from_params(params=opt_sep)
                                mu, variance = eval_in_model(gp_model.predict(data.flux[j] - opt_y,
                                                                              t=self.data.time,
                                                                              return_var=True), model=model)
                                opt_y = opt_y + mu

                                plt.plot(
                                    self.data.time,
                                    mu + 1,
                                    "C0-",
                                    label="GP_model (+1)",
                                    lw=2,
                                    alpha=0.5
                                )

                                plt.plot(
                                    self.data.time,
                                    opt_y,
                                    "C2-",
                                    label="MAP optimization (with GP)",
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

    def sample(
        self,
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
        if self.optimization == "separate":
            self.trace = []
            starts = []
            if "start" in kw:
                starts = kw["start"]
                kw.pop("start")

            for i, mod in enumerate(self._pymc3_model):
                if self.optimization == "separate":
                    check_initial_guess(mod)
                with mod:
                    if len(starts) > 0:
                        start = starts[i]
                    else:
                        start = mod.test_point

                    try:
                        print(f"Sampling Wavelength: {i}")
                        samp = sampling_method(start=start, **sampling_kw, **kw)
                        # if summarize_step_by_step:
                        #     self.summary.append(summary(samp, **summarize_kw))
                        # else:
                        self.trace.append(samp)

                    except EOFError:
                        print(
                            """An error (EOFError) with Starry has occurred related to multiprocessing. 
                            Sometimes it works if we try sampling using 'spawn' instead of default 'fork' (mp_ctx='spawn')..."""
                        )
                        sampling_kw['mp_ctx'] = "spawn"
                        samp = sampling_method(start=start, **sampling_kw, **kw)
                        self.trace.append(samp)

                    except Exception as e:
                        print(f"Sampling failed for one of the models: {e}")
                        self.trace.append(None)

            if summarize_step_by_step:
                if isinstance(self, CombinedModel):
                    for m in self._chromatic_models.values():
                        m.summary = self.summary

        else:
            check_initial_guess(self._pymc3_model)
            with self._pymc3_model:
                self.trace = sampling_method(**sampling_kw, **kw)

        self.summarize(**summarize_kw)

    def summarize(self, hdi_prob=0.68, round_to=10, fmt="wide", print_table=True, overwrite=False, **kw):
        """
        Wrapper for arviz summary
        """
        if not hasattr(self, "trace"):
            print("Sampling has not been run yet! Running now with defaults...")
            self.sample()

        if overwrite == False:
            if hasattr(self, "summary"):
                print("Summarize has already been run. If you want to overwrite the table include the `overwrite` kw: `{self}.summarize(..., overwrite=True)`")
                if print_table:
                    print(self.summary)
                return

        if self.optimization == "separate":
            self.summary = []
            for mod, trace in zip(self._pymc3_model, self.trace):
                with mod:
                    self.summary.append(
                        summary(trace, hdi_prob=hdi_prob, round_to=round_to, fmt=fmt, **kw)
                    )
        else:
            with self._pymc3_model:
                self.summary = summary(
                    self.trace, hdi_prob=hdi_prob, round_to=round_to, fmt=fmt, **kw
                )

        if print_table:
            print(self.summary)

        if isinstance(self, CombinedModel):
            for m in self._chromatic_models.values():
                m.summary = self.summary

    def get_results(self, as_df=True, uncertainty=["hdi_16%", "hdi_84%"]):
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
        # self.summarize(round_to=7, fmt="wide")

    def imshow_priors(self, n=3, quantity="data", plot_all=True):
        """
        Plot n prior samples from the parameter distributions defined by the user
        :parameter n
        Number of priors to plot (default=3)
        """
        # setup the models, data and orbits in a format for looping
        (
            datas,
            _,
            prior_predictive_traces,
        ) = self.choose_model_based_on_optimization_method(self.sample_prior(ndraws=n))

        for nm, (data, prior_predictive_trace) in enumerate(
            zip(datas, prior_predictive_traces)
        ):
            for i in range(n):
                # if data.nwave == 1:
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
                            if np.shape(v) == ():
                                i_dict = {**i_dict, **{k: v}}
                            elif np.shape(v)[1] == 1:
                                i_dict = {**i_dict, **{k: v[i][0]}}
                            elif np.shape(v)[1] == data.nwave:
                                i_dict = {**i_dict, **{k: v[i][w]}}
                            else:
                                i_dict = {**i_dict, **{k: v[i]}}
                        model_for_this_sample.append(self.model(params=i_dict))
                    model_for_this_sample = np.array(model_for_this_sample)

                # add posterior model and draw from posterior distribution to the Rainbow quantities:
                data.fluxlike[f"prior-model-{i}"] = model_for_this_sample
                data.fluxlike[f"prior-predictive-{i}"] = flux_for_this_sample

            if plot_all:
                data.imshow_quantities()
            else:
                f, ax = plt.subplots(
                    nrows=n, sharex=True, sharey=True, figsize=(6, 4 * n)
                )
                for i in range(n):
                    if f"{self.name}_model" in prior_predictive_trace.keys():
                        data.imshow(ax=ax[i], quantity=f"prior-model-{i}")
                    else:
                        data.imshow(ax=ax[i], quantity=f"prior-predictive-{i}")

    def imshow_posteriors(self, n=3, quantity="data", plot_all=True):
        """
        Plot n posterior samples from the parameter distributions defined by the user
        :parameter trace
        PyMC3 trace object (need to run sample first)
        :parameter n
        Number of posteriors to plot (default=3)
        """
        # if we have the separate wavelength optimization method chosen then repeat for every wavelength/model
        (
            datas,
            _,
            posterior_predictive_traces,
        ) = self.choose_model_based_on_optimization_method(self.sample_posterior(n))

        posterior_model = {}
        for nm, (data, posterior_predictive_trace) in enumerate(
            zip(datas, posterior_predictive_traces)
        ):
            for i in range(n):
                # for every posterior sample extract the posterior model and distribution draw:
                try:
                    flux_for_this_sample = np.array(
                        posterior_predictive_trace[quantity][i]
                    )
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
                            if np.shape(v) == ():
                                i_dict = {**i_dict, **{k: v}}
                            elif np.shape(v)[1] == 1:
                                i_dict = {**i_dict, **{k: v[i][0]}}
                            elif np.shape(v)[1] == data.nwave:
                                i_dict = {**i_dict, **{k: v[i][w]}}
                            else:
                                i_dict = {**i_dict, **{k: v[i]}}
                        model_for_this_sample.append(self.model(params=i_dict))
                    model_for_this_sample = np.array(model_for_this_sample)

                # add posterior model and draw from posterior distribution to the Rainbow quantities:
                data.fluxlike[f"posterior-model-{i}"] = model_for_this_sample
                data.fluxlike[f"posterior-predictive-{i}"] = flux_for_this_sample

            if plot_all:
                data.imshow_quantities()
            else:
                f, ax = plt.subplots(
                    nrows=n, sharex=True, sharey=True, figsize=(6, 4 * n)
                )
                for i in range(n):
                    if f"{self.name}_model" in posterior_model.keys():
                        data.imshow(ax=ax[i], quantity=f"posterior-model-{i}")
                    else:
                        data.imshow(ax=ax[i], quantity=f"posterior-predictive-{i}")

    def extract_deterministic_model(self):
        """
        Extract the deterministic model from the summary statistics
        """
        datas, _ = self.choose_model_based_on_optimization_method()

        model = {}
        for nm, data in enumerate(datas):
            for w in range(data.nwave):
                if self.optimization == "separate":
                    summary = self.summary[w + nm]
                else:
                    summary = self.summary

                if f"w{w + nm}" not in model.keys():
                    model[f"w{w + nm}"] = []

                if f"{self.name}_model[{w}, 0]" in summary["mean"].keys():
                    for t in range(data.ntime):
                        model[f"w{w + nm}"].append(
                            summary["mean"][f"{self.name}_model[{w}, {t}]"]
                        )
                elif f"{self.name}_model[0]" in summary["mean"].keys():
                    for t in range(data.ntime):
                        model[f"w{w + nm}"].append(
                            summary["mean"][f"{self.name}_model[{t}]"]
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
                    np.array(model["total"][f"w{i}"]) - (i * spacing),
                    color="k",
                )

        if "filename" not in kw.keys():
            plt.show()
        else:
            plt.savefig(kw["filename"])
        plt.close()

        return ax

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
                if f"{k}[{i}]" in posterior_means.index and\
                        "limb_darkening" not in k and\
                        "ecs" not in k and\
                        "planet_surface_map" not in k:
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
            return corner.corner(self.trace, **kw)

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

    def get_model(
        self,
        params_dict: dict = None,
        as_dict: bool = True,
        as_array: bool = False,
        store: bool = True,
        **kw
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
                model_i = self.model(params, i, **kw)
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

    def chi_squared(self, individual_wavelengths=False, **kw):
        if hasattr(self, "data_with_model"):
            if self.optimization == "simultaneous":
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

            elif self.optimization == "separate":
                for i in range(self.data.nwave):
                    if self.store_models:
                        summary = self.summary[i].iloc[
                            ~self.summary.index.str.contains(f"{self.name}_model")
                        ]
                    else:
                        summary = self.summary[i]
                    fit_params = len(summary)
                    degrees_of_freedom = self.data.ntime - fit_params
                    print(f"\nFor Wavelength {i}:")
                    chi_sq(
                        data=self.data_with_model.flux[i],
                        model=self.data_with_model.model[i],
                        uncertainties=self.data_with_model.uncertainty[i],
                        degrees_of_freedom=degrees_of_freedom,
                        **kw,
                    )
            elif self.optimization == "white_light":
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
        if hasattr(self, "data_with_model"):
            plt.figure(figsize=(12, 6))
            for i in range(self.data.nwave):
                plt.plot(
                    self.data_with_model.time,
                    self.data_with_model.residuals[i],
                    alpha=0.5,
                    label=f"Wavelength {i}",
                    **kw,
                )
            plt.plot(
                self.data_with_model.time,
                np.mean(self.data_with_model.residuals, axis=0),
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
        if hasattr(self, "data_with_model"):
            print("For the Wavelength-Averaged Residuals...")
            noise_calculator(np.mean(self.data_with_model.residuals, axis=0), **kw)
            if individual_wavelengths:
                for i in range(self.data.nwave):
                    print(f"\nFor wavelength {i}")
                    noise_calculator(self.data_with_model.residuals[i], **kw)
        else:
            warnings.warn(
                f"""Could not find .data_with_model in {self.name} model! 
                Please run [self].add_model_to_rainbow() or one of the plotting methods
                with models before rerunning this method."""
            )

    def plot_residuals_periodogram(self, **kw):
        if hasattr(self, "data_with_model"):
            plt.figure(figsize=(12, 6))
            plt.title("Periodogram of Wavelength-Averaged Residuals")
            generate_periodogram(
                x=np.mean(self.data_with_model.residuals, axis=0),
                fs=(
                    1 / (self.data_with_model.time[1] - self.data_with_model.time[0])
                ).to_value("1/d"),
                **kw,
            )
        else:
            warnings.warn(
                f"""Could not find .data_with_model in {self.name} model! 
                Please run [self].add_model_to_rainbow() or one of the plotting methods
                with models before rerunning this method."""
            )

    def write_chromatic_model(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    def make_spectrum_table(
        self, param="radius_ratio", uncertainty=["hdi_16%", "hdi_84%"], svname=None
    ):
        """
        Generate and return a spectrum table

        Parameters
        ----------
        uncertainty: [Optional] List of the names of parameters to use as the lower and upper errors. Options: "hdi_16%",
        "hdi_84%", "sd" etc. Default = ["hdi_16%", "hdi_84%"].
        svname: [Optional] String csv filename to save the transmission table. Default = None (do not save)

        Returns
        -------

        """
        results = self.get_results(uncertainty=uncertainty)[
            [
                "wavelength",
                f"{self.name}_{param}",
                f"{self.name}_{param}_{uncertainty[0]}",
                f"{self.name}_{param}_{uncertainty[1]}",
            ]
        ]
        trans_table = results[["wavelength", f"{self.name}_{param}"]]
        if "hdi" in uncertainty[0]:
            trans_table[f"{self.name}_{param}_neg_error"] = (
                results[f"{self.name}_{param}"]
                - results[f"{self.name}_{param}_{uncertainty[0]}"]
            )
            trans_table[f"{self.name}_{param}_pos_error"] = (
                results[f"{self.name}_{param}_{uncertainty[1]}"]
                - results[f"{self.name}_{param}"]
            )
        else:
            trans_table[f"{self.name}_{param}_neg_error"] = results[
                f"{self.name}_{param}_{uncertainty[0]}"
            ]
            trans_table[f"{self.name}_{param}_pos_error"] = results[
                f"{self.name}_{param}_{uncertainty[1]}"
            ]

        if svname is not None:
            assert isinstance(svname, object)
            trans_table.to_csv(svname)
        else:
            return trans_table

    def plot_spectrum(
        self, param="radius_ratio", name_of_spectrum="transmission", table=None, uncertainty=["hdi_16%", "hdi_84%"],
        ax=None, plotkw={}, **kw
    ):
        """
        Plot the transmission spectrum (specifically the planet size as a function of wavelength).

        Parameters
        ----------
        table: [Optional] Table to use as transmssion spectrum (otherwise the default is to use the MCMC sampling results.
        The table must have the following columns: "{self.name}_radius_ratio", "{self.name}_radius_ratio_neg_error",
        "{self.name}_radius_ratio_pos_error", "wavelength".
        uncertainty: [Optional] List of the names of parameters to use as the lower and upper errors. Options: "hdi_16%", "hdi_84%",
        "sd" etc. Default = ["hdi_16%", "hdi_84%"].
        ax: [Optional] Pass a preexisting matplotlib axis is to be used instead of creating a new one.Default = None.
        plotkw: [Optional] Dict of kw to pass to the transmission specrum plotting function.
        kw: [Optional] kw to pass to the TransitModel.make_transmission_spectrum_table

        Returns
        -------

        """
        if table is not None:
            transmission_spectrum = table
            try:
                # ensure the correct columns exist in the transmission spectrum table
                assert transmission_spectrum[f"{self.name}_{param}"]
                assert transmission_spectrum[f"{self.name}_{param}_neg_error"]
                assert transmission_spectrum[f"{self.name}_{param}_pos_error"]
                assert transmission_spectrum["wavelength"]
            except:
                print(
                    f"The given table doesn't have the correct columns 'wavelength', '{self.name}_{param}', "
                    f"{self.name}_{param}_pos_error' and '{self.name}_{param}_neg_error'"
                )
        else:
            kw["uncertainty"] = uncertainty
            transmission_spectrum = self.make_spectrum_table(param=param, **kw)
            transmission_spectrum["wavelength"] = [
                t.to_value("micron") for t in transmission_spectrum["wavelength"].values
            ]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        plt.sca(ax)
        plt.title(f"{name_of_spectrum} Spectrum")
        plt.plot(
            transmission_spectrum["wavelength"],
            transmission_spectrum[f"{self.name}_{param}"],
            "kx",
            **plotkw,
        )
        plt.errorbar(
            transmission_spectrum["wavelength"],
            transmission_spectrum[f"{self.name}_{param}"],
            yerr=[
                transmission_spectrum[f"{self.name}_{param}_neg_error"],
                transmission_spectrum[f"{self.name}_{param}_pos_error"],
            ],
            color="k",
            capsize=2,
            linestyle="None",
            **plotkw,
        )
        plt.xlabel("Wavelength (microns)")
        plt.ylabel(param)


from .combined import *
