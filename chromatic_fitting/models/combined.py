import matplotlib.axes

from ..imports import *

# from .lightcurve import *
from typing import Union
import operator

from .transit import *
from .polynomial import *
from .eclipse import *
from .phase_curve import *
from .gp import *

"""
Example of setting up a CombinedModel:

def create_new_combined_model(models):
    # e.g. if there was a transit model and 2 polynomial models:
    t = models['transit']
    p1 = models['p1']
    p2 = models['p2']
    
    # combine using standard arithmetic operators:
    cm = t * (p1 + p2)
    
    # attach a Rainbow object, r, to the model:
    cm.attach_data(r)
    
    # setup the lightcurves for the combined model:
    cm.setup_lightcurves()
    
    # relate the "actual" data to the model (using a Normal likelihood function)
    cm.setup_likelihood()    
    
    # MCMC (NUTS) sample the parameters:
    cm.sample(tune=2000, draws=2000, chains=4, cores=4)
    
    # summarize the results:
    cm.summarize(round_to=7, fmt='wide')
    
    return cm

"""


def add_dicts(dict_1: dict, dict_2: dict) -> dict:
    """
    Combine two dictionaries, if they have the same key then add the values of that key

    Parameters
    ----------
    dict_1: first dictionary to add
    dict_2: second dictionary to add

    Returns
    -------
    object

    """
    # combine the keys that are unique to dict_1 or dict_2 into new dict_3 (order doesn't matter for addition)
    dict_3 = {**dict_1, **dict_2}
    # for the keys that appear in both add the values
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            dict_3[key] = np.array(value) + np.array(dict_1[key])
    return dict_3


def subtract_dicts(dict_1: dict, dict_2: dict) -> dict:
    """
    Combine two dictionaries, if they have the same key then subtract the values of second dictionary from first

    Parameters
    ----------
    dict_1: first dictionary
    dict_2: second dictionary to subtract from first

    Returns
    -------
    object

    """
    # # combine the keys that are unique to dict_1 or dict_2 into new dict_3 (order matters for subtraction!)
    dict_3 = {**dict_1, **dict_2}
    # for the keys that appear in both subtract values in dict_2 from value in dict_1
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            dict_3[key] = -np.array(value) + np.array(dict_1[key])
    return dict_3


def multiply_dicts(dict_1: dict, dict_2: dict) -> dict:
    """
    Combine two dictionaries, if they have the same key then multiply the values of second dictionary by first

    Parameters
    ----------
    dict_1: first dictionary
    dict_2: second dictionary

    Returns
    -------
    object

    """
    # combine the keys that are unique to dict_1 or dict_2 into new dict_3 (order doesn't matter for multiplication)
    dict_3 = {**dict_1, **dict_2}
    # for the keys that appear in both multiply values in dict_2 by value in dict_1
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            dict_3[key] = np.array(value) * np.array(dict_1[key])
    return dict_3


def divide_dicts(dict_1: dict, dict_2: dict) -> dict:
    """
    Combine two dictionaries, if they have the same key then divide the values of first dictionary by second

    Parameters
    ----------
    dict_1: first dictionary to divide by second
    dict_2: second dictionary

    Returns
    -------
    object

    """
    # combine the keys that are unique to dict_1 or dict_2 into new dict_3 (order matters for division!)
    dict_3 = {**dict_1, **dict_1}
    # for the keys that appear in both divide values in dict_1 by value in dict_2
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            dict_3[key] = np.array(dict_1[key]) / np.array(value)
    return dict_3


# define a dictionary mapping operations to functions
combination_options = {
    "+": add_dicts,
    "-": subtract_dicts,
    "*": multiply_dicts,
    "/": divide_dicts,
}

combination_options_array = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}


class CombinedModel(LightcurveModel):
    """
    A combined model of multiple models for the lightcurve.
    """

    def __init__(self, name: str = "combined", **kw):
        """
        Initialise the combined model.

        Parameters
        ----------
        name: the name of the model ('combined' by default)
        kw: any keywords to pass to the lightcurve model
        """
        super().__init__(name, **kw)
        self.name = name
        self.metadata = {}
        self.parameters = {}
        self.model = self.combined_model
        self.type_of_model = "combined"
        self.has_gp = False

    def __repr__(self):
        """
        Print the combined model (and its constituents).
        """
        if hasattr(self, "_chromatic_models"):
            string_to_print = f"<chromatic combined model '{self.name}' 🌈, models: "
            for i, (model_name, model) in enumerate(self._chromatic_models.items()):
                if i >= len(self.how_to_combine):
                    string_to_print += f"{model}"
                else:
                    string_to_print += f"{model} {self.how_to_combine[i]} "
            return string_to_print
        else:
            return f"<chromatic combined model '{self.name}' 🌈>"

    def combine(
        self, first: LightcurveModel, second: LightcurveModel, how_to_combine: str
    ):
        """
        Combine at least two LightcurveModels into a new CombinedModel

        Parameters
        ----------
        first: first LightcurveModel (this can already be a CombinedModel) to combine in CombinedModel
        second: second LightcurveModel (this can already be a CombinedModel) to combine with first in CombinedModel
        how_to_combine: instructions on how to combine the two models
        """
        if isinstance(first, CombinedModel) and isinstance(second, CombinedModel):
            # if both first and second are CombinedModels
            chromatic_models = add_dicts(
                first._chromatic_models.copy(), second._chromatic_models.copy()
            )
            self.how_to_combine = first.how_to_combine + [
                how_to_combine
            ]  # second.how_to_combine
            self.attach_models(
                chromatic_models, how_to_combine=second.how_to_combine
            )  # how_to_combine)
            # self.how_to_combine = first.how_to_combine + second.how_to_combine
            # self.attach_models(chromatic_models, how_to_combine=how_to_combine)
        elif isinstance(first, CombinedModel):
            # if the first is a CombinedModel but second is not
            chromatic_models = first._chromatic_models.copy()
            chromatic_models[f"{second.name}"] = second
            self.how_to_combine = first.how_to_combine
            self.attach_models(chromatic_models, how_to_combine=how_to_combine)
        elif isinstance(second, CombinedModel):
            # if the second is a CombinedModel but first is not
            chromatic_models = second._chromatic_models.copy()
            chromatic_models[f"{first.name}"] = first
            self.how_to_combine = second.how_to_combine
            self.attach_models(chromatic_models, how_to_combine=how_to_combine)
        else:
            # if neither is a CombinedModel
            self.attach_models(
                {f"{first.name}": first, f"{second.name}": second},
                how_to_combine=how_to_combine,
            )

    def attach_models(self, models: dict, how_to_combine: Union[str, list] = "+"):
        """
        Attach multiple LightCurveModel in dictionary to the CombinedModel

        Parameters
        ----------
        models: dictionary of models to combine
        how_to_combine: string or list of instructions of how to combine the models (default='+')
        """
        new_models = {}

        # create/ammend the 'how_to_combine' attribute with this new model + operation
        # if we have already attached models with instructions on how to combine then add this operation
        # to self.how_to_combine. This is useful for recursively adding new models to the CombinedModel object
        if hasattr(self, "how_to_combine"):
            self.how_to_combine.extend(how_to_combine)
        else:
            if type(how_to_combine) == str:
                # if a string operation ("+","-", etc.) has been passed then repeat this operation for every model
                self.how_to_combine = [how_to_combine] * (len(models.keys()) - 1)
            else:
                # if we have passed a list of operations of the same length as (n_models-1) then save
                if len(how_to_combine) == len(models.keys()) - 1:
                    self.how_to_combine = how_to_combine
                else:
                    # if the wrong length for how_to_combine has been passed, warn the user
                    print(
                        f"WARNING: You have passed {len(how_to_combine)} operations for {len(models.keys())} models!"
                    )
                    return

        for name, model in models.items():
            # check that the models passed to this function are LightcurveModels
            if isinstance(model, LightcurveModel):
                # make a "copy" of each model:
                class_inputs = model.extract_extra_class_inputs()
                new_model = model.__class__(**class_inputs)
                # link the pymc3 model for this constituent model to the overall ChromaticModel (avoids weird
                # inheritance issues):
                new_model._pymc3_model = self._pymc3_model
                model_params = {}

                # for every parameter in the separate models redefine them in the separate models within CombinedModel
                # and the new CombinedModel
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
                new_model.defaults = add_string_before_each_dictionary_key(
                    new_model.defaults, new_model.name
                )
                new_model.required_parameters = [
                    f"{new_model.name}_{a}" for a in new_model.required_parameters
                ]
                new_model.setup_parameters(**model_params)
                new_models[name] = new_model
            else:
                print(
                    "WARNING: This class can only be used to combine LightcurveModels!"
                )

        # set up parameters in new combined model
        self._chromatic_models = new_models

        for mod in self._chromatic_models.values():
            if isinstance(mod, GPModel):
                self.has_gp = True

    def apply_operation_to_constituent_models(
        self, operation: str, *args: object, **kwargs: object
    ) -> object:
        """
        Apply an operation to all models within a combined model

        Parameters
        ----------
        operation: string name of the operation to carry out
        args: arguments to pass to the operation
        kwargs: keywords to pass to the operation

        Returns
        -------
        object
        """
        results = []
        # for each constituent model apply the chosen operation
        for m in self._chromatic_models.values():
            try:
                op = getattr(m, operation)
                result = op(*args, **kwargs)
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"Error applying {operation} to {m}: {e}")

        # if there are returned value(s) from the operation then return these, otherwise return None
        if len(results) == 0:
            return None
        else:
            return results

    def summarize_parameters(self):
        """
        Print a summary of each of the model's parameters
        """
        print(
            "A CombinedModel itself does not have any parameters, however each of its constituent models do:\n"
        )
        self.apply_operation_to_constituent_models("summarize_parameters")

    def what_are_parameters(self):
        """
        Print a summary of what each parameter is for each model in CombinedModel
        # """
        print(
            "A CombinedModel itself does not have any parameters, however each of its constituent models do:\n"
        )
        self.apply_operation_to_constituent_models("what_are_parameters")

    def attach_data(self, r: chromatic.Rainbow):
        """
        Connect a `chromatic` Rainbow dataset to this object and the constituent models.

        Parameters
        ----------
        r: Rainbow object with the light curve data

        """
        self.data = r._create_copy()
        # also attach data to each model
        self.apply_operation_to_constituent_models("attach_data", r)

    def choose_optimization_method(self, optimization_method: str = "simultaneous"):
        """

        Parameters
        ----------
        optimization_method: the method of optimization, options are 'simultaneous', 'separate' or 'white_light'
        (default='simultaneous')
        """
        super().choose_optimization_method(optimization_method)
        for m in self._chromatic_models.values():
            m.optimization = optimization_method

        if optimization_method == "separate":
            # if we chose to optimize each wavelength independently then we need to change all the parameters types
            # to WaveLike (have a different value for each wavelength)
            self.apply_operation_to_constituent_models("change_all_priors_to_Wavelike")

    def setup_orbit(self):
        """
        Create an `exoplanet` orbit model, given the stored parameters.
        """
        self.apply_operation_to_constituent_models("setup_orbit")

    def plot_orbit(self):
        """
        Plot an `exoplanet` orbit model, given the stored parameters.
        """
        self.apply_operation_to_constituent_models("plot_orbit")

    def setup_lightcurves(self, store_models: bool = False, **kw):
        """
        Set-up lightcurves in combined model : for each consituent model set-up the lightcurves according to their
        model definition

        Parameters
        ----------
        store_models: boolean for whether to store the transit model during fitting (for faster
        plotting/access later), default=False
        """

        self.every_light_curve = {}
        self.initial_guess = {}

        # we can decide to store the LC models during the fit (useful for plotting later, however, uses large amounts
        # of RAM)
        self.store_models = store_models
        for cm in self._chromatic_models.values():
            cm.store_models = store_models

        # for each constituent model set-up the lightcurves according to their model type:
        self.apply_operation_to_constituent_models("setup_lightcurves", **kw)

        i_gp =0
        for i, mod in enumerate(self._chromatic_models.values()):
            # for each lightcurve in the combined model, add/subtract/multiply/divide their lightcurve into the combined
            # model
            if hasattr(mod, "initial_keplerian_system"):
                self.initial_keplerian_system = mod.initial_keplerian_system

            if i == 0:
                if hasattr(mod, "every_light_curve"):
                    self.every_light_curve = mod.every_light_curve
                else:
                    print(f"No light curve saved for chromatic model {mod}")

                if hasattr(mod, "initial_guess"):
                    self.initial_guess = mod.initial_guess

                else:
                    print(f"No initial guess saved for chromatic model {mod}")
            else:
                if hasattr(mod, "every_light_curve"):
                    self.every_light_curve = combination_options[
                        self.how_to_combine[i - 1]
                    ](self.every_light_curve, mod.every_light_curve)
                else:
                    print(f"No light curve saved for chromatic model {mod}")

                if hasattr(mod, "initial_guess"):
                    self.initial_guess = combination_options[
                        self.how_to_combine[i - 1]
                    ](self.initial_guess, mod.initial_guess)
                else:
                    print(f"No initial guess saved for chromatic model {mod}")

            if i_gp == 0:
                # if hasattr(mod, "kernel_func"):
                #     self.kernel_func = [mod.kernel_func]

                if hasattr(mod, "covar"):
                    self.independant_variable = mod.independant_variable
                    self.covar = mod.covar

            else:
                # if hasattr(mod, "kernel_func"):
                #     self.kernel_func.append(mod.kernel_func)

                if hasattr(mod, "covar"):
                    if type(self.independant_variable) is not list:
                        if mod.independant_variable == self.independant_variable:
                            self.covar = combination_options[self.how_to_combine[i - 1]](
                                self.covar, mod.covar
                            )
                        else:
                            warnings.warn("GPs using different independent variables! This has not been tested!!")
                            self.independant_variable = [self.independant_variable]
                            self.independant_variable.append(mod.independant_variable)
                            self.covar = [self.covar]
                            self.covar.append(mod.covar)
                    else:
                        warnings.warn("GPs using different independent variables! This has not been tested!!")
                        matches_any_variable = False
                        for num_mod, iv in enumerate(self.independant_variable):
                            if mod.independant_variable == iv:
                                self.covar[num_mod] = combination_options[self.how_to_combine[i - 1]](
                                    self.covar[num_mod], mod.covar
                                )
                                matches_any_variable = True

                        if not matches_any_variable:
                            self.independant_variable.append(mod.independant_variable)
                            self.covar.append(mod.covar)

            if isinstance(mod, GPModel):
                i_gp += 1

        datas, models = self.choose_model_based_on_optimization_method()

        # (if we've chosen to) add a Deterministic parameter to the model for easy extraction/plotting
        # later:
        if self.store_models:
            for i, (mod, data) in enumerate(zip(models, datas)):
                with mod:
                    # for w in range(data.nwave):
                    # k = f"wavelength_{i + w}"
                    Deterministic(
                        f"{self.name}_model", self.every_light_curve[f"wavelength_{i}"]
                    )

    def setup_likelihood(
        self,
        **kw,
    ):
        if "marginalize_map_analytically" in kw.keys():
            if kw["marginalize_map_analytically"]:
                PhaseCurveModel.setup_likelihood(self, **kw)
                return
            else:
                kw.pop("marginalize_map_analytically")
                LightcurveModel.setup_likelihood(self, **kw)
                return

        if self.has_gp:
            if not hasattr(self, "mean"):
                warnings.warn(
                    "\nYou have not provided a mean to the GP. Are you sure this is right?\nIf you want to "
                    "add a mean please run {self}.add_mean. We will proceed assuming a mean of 0.\n"
                )
                self.mean = Fixed(0.0)
            if not hasattr(self, "log_jitter"):
                warnings.warn(
                    "\nYou have not provided a jitter to the GP. Are you sure this is right?\nIf you want to "
                    "add a mean please run {self}.add_jitter. We will proceed assuming a jitter of 0.\n"
                )
                self.log_jitter = Fixed(np.log(0.0))

            # if type(self.independant_variable) is not list:
            #     GPModel.setup_likelihood(self, **kw)
            #     return
            # else:
            #     warnings.warn("GP models use different independent variables - this does not work yet!!")
            #     return
        LightcurveModel.setup_likelihood(self, **kw)


    def sample(
        self,
        **kw,
    ):
        if "sampling_method" not in kw.keys():
            sampling_method = sample
            for m in self._chromatic_models.values():
                if isinstance(m, EclipseModel):
                    sampling_method = sample_ext
        else:
            sampling_method = kw["sampling_method"]
            kw.pop("sampling_method")

        LightcurveModel.sample(self, sampling_method=sampling_method, **kw)

    def add_jitter(self, log_jitter_parameter):
        if self.has_gp:
            self.apply_operation_to_constituent_models("add_jitter", log_jitter_parameter=log_jitter_parameter)
            GPModel.add_jitter(self, log_jitter_parameter=log_jitter_parameter)
        else:
            warnings.warn("You can't add jitter to a non-GP model")

    def add_mean(self, mean_parameter):
        if self.has_gp:
            self.apply_operation_to_constituent_models("add_mean", mean_parameter=mean_parameter)
            GPModel.add_mean(self, mean_parameter=mean_parameter)
        else:
            # TO DO - expand this for all models!
            warnings.warn("You can't add a mean to a non-GP model")

    def kernel_func(self, i=0, **params):
        num_gps = 0
        for imod, mod in enumerate(self._chromatic_models.values()):
            if isinstance(mod, GPModel):
                new_params = return_only_keys_with_string_at_start_of_dictionary(params, f"{mod.name}_")
                new_params = remove_all_keys_with_string_from_dictionary(new_params, 'interval')
                new_params = remove_string_from_each_dictionary_key(new_params, f"{mod.name}_")
                new_params = remove_string_from_each_dictionary_key(new_params, f"[{i}]")
                new_params = extract_index_from_each_dictionary_key(new_params, index=i)
                #         print(new_params)

                model_params_without_name = remove_string_from_each_dictionary_key(mod.parameters, f"{mod.name}_")

                specific_params = {}
                #         print(new_params.keys(), model_params_without_name.keys())
                for p in model_params_without_name.keys():
                    if p in new_params.keys():
                        specific_params[p] = new_params[p]
                    elif isinstance(model_params_without_name[p], WavelikeFixed):
                        specific_params[p] = model_params_without_name[p].value[i]
                    elif isinstance(model_params_without_name[p], Fixed):
                        specific_params[p] = model_params_without_name[p].value

                #         print(specific_params)
                k = mod.kernel_func(**specific_params)

                if num_gps == 0:
                    k_func = k
                else:
                    k_func = combination_options_array[self.how_to_combine[imod - 1]](k_func, k)

                num_gps += 1

        return k_func


    def get_results(self, **kw):
        """
        Extract the 'best-fit' parameter mean + error values from the summary table for each constituent model
        """
        results = []
        for mod in self._chromatic_models.values():
            results.append(mod.get_results(**kw))
        # combine the results from all models:
        df = pd.concat(results, axis=1)
        # remove duplicated columns (wavelength):
        df = df.loc[:, ~df.columns.duplicated()].copy()
        return df

    def get_model(self, store: bool = True, **kw: object):
        """
        Extract each of the 'best-fit' models (for each chromatic_fitting model) from the arviz summary tables.
        """
        all_models, total_model = {}, {}
        # for each constituent model get its 'best-fit' model:
        models = self.apply_operation_to_constituent_models("get_model", **kw)

        as_array = False
        if "as_array" in kw:
            if kw["as_array"]:
                as_array = True

        kw["store"] = store

        if models is not None:
            for i, (name, m) in enumerate(self._chromatic_models.items()):
                models_dict = {}
                # get 'best-fit' model from constituent model:
                all_models[name] = models[i]

                if as_array:
                    for w in range(len(models[i])):
                        models_dict[f"w{w}"] = models[i][w]
                else:
                    models_dict = models[i]

                # add this model to the total model:
                if not self.store_models:
                    total_model = combination_options[self.how_to_combine[i - 1]](
                        total_model, models_dict
                    )

            if self.store_models:
                all_models["total"] = self.extract_deterministic_model()
            else:
                all_models["total"] = total_model

            if store:
                self._fit_models = all_models

            # return all_models
            if "as_array" in kw:
                if kw["as_array"]:
                    return np.array(
                        list([list(mod.values()) for mod in all_models.values()])
                    )
            else:
                return all_models

        else:
            warnings.warn(
                "There are no current saved models. You can, however, generate models by passing a dictionary with "
                "all the necessary parameters"
            )

    def combined_model(self, **kw):
        all_models, total_model = {}, []
        # for each constituent model get its 'best-fit' model:
        models = self.apply_operation_to_constituent_models("model", **kw)

        if models is not None:
            for j in range(len(self._chromatic_models.keys())):
                # add this model to the total model:
                # total_model = combination_options[self.how_to_combine[i - 1]](
                #     total_model, models[i]
                # )
                if j == 0:
                    total_model = models[j]
                else:
                    total_model = combination_options_array[self.how_to_combine[j - 1]](
                        total_model, models[j]
                    )

        return total_model

    def add_model_to_rainbow(self):
        """
        Add the 'best-fit' model(s) to the `rainbow` object.
        """

        transit_model, systematics_model, total_model = {}, {}, {}
        i_transit, i_sys = 0, 0
        for i, mod in enumerate(self._chromatic_models.values()):
            # if there's a planet model in the CombinedModel then separate this out to add to Rainbow
            # if isinstance(mod, TransitModel):
            if mod.type_of_model == "planet":
                if i_transit == 0:
                    transit_model = mod.get_model()
                else:
                    # I'm not sure that this works if the combination option is multiply or divide
                    transit_model = combination_options[self.how_to_combine[i - 1]](
                        transit_model, mod.get_model()
                    )
                i_transit += 1
            else:
                # otherwise combine all other models in the CombinedModel as systematics to add to Rainbow
                if i_sys == 0:
                    systematics_model = mod.get_model()
                else:
                    systematics_model = combination_options[self.how_to_combine[i - 1]](
                        systematics_model, mod.get_model()
                    )
                i_sys += 1

            if i == 0:
                total_model = mod.get_model()
            else:
                total_model = combination_options[self.how_to_combine[i - 1]](
                    total_model, mod.get_model()
                )

        # add the models to the rainbow object:
        if self.outlier_flag:
            data = self.data_without_outliers
        else:
            data = self.data

        if self.optimization == "white_light":
            data = self.white_light

        # add (planet + systematics) models to Rainbow
        r_with_model = data.attach_model(
            model=np.array(list(total_model.values())),
            planet_model=np.array(list(transit_model.values())),
            systematics_model=np.array(list(systematics_model.values())),
        )

        # save the rainbow_with_model for plotting:
        self.data_with_model = r_with_model

    def make_transmission_spectrum_table(self, **kw: object) -> object:
        """
        Return a transmission spectrum table using the best fit parameters

        Parameters
        ----------
        kw: keywords to pass to 'make_transmission_spectrum_table' methods

        Returns
        -------
        object
        """
        results = self.apply_operation_to_constituent_models(
            "make_transmission_spectrum_table", **kw
        )

        num_transspec = 0
        for v in self._chromatic_models.values():
            if hasattr(v, "transmission_spectrum"):
                transspec = v.transmission_spectrum
                num_transspec += 1

        if num_transspec == 1:
            self.transmission_spectrum = transspec
        elif num_transspec > 1:
            warnings.warn(
                "WARNING: More than one of the models in CombinedModel have a transmission spectrum - "
                "are you sure this is right?"
            )

        if len(results) == 1:
            results = results[0]
        return results

    def generate_gp_model_from_params(self, params, i=0, **kw):
        params = params.copy()
        for name, default_value in zip(["log_jitter", "mean"], [1.0, 0.0]):
            if name not in params.keys():
                params[name] = default_value

        kernel = self.kernel_func(**params)

        x = self.data.get(self.independant_variable)
        if self.independant_variable == "time":
            x = x.to_value("day")
        if len(np.shape(x)) > 1:
            xi = x[i, :]
        else:
            xi = x

        gp = GaussianProcess(
            kernel,
            mean=params["mean"],
            t=xi,
            diag=self.data.uncertainty[i, :] + np.exp(params["log_jitter"]),
        )

        return gp

    def plot_prediction(self, gp, i=0, plot_var=True, legend=True):
        if self.has_gp:
            GPModel.plot_prediction(self, gp=gp, i=i, plot_var=plot_var, legend=legend)

    def plot_traces(self, num_traces=5, i=0):
        if self.has_gp:
            GPModel.plot_traces(self, num_traces=num_traces, i=i)

    def plot_transmission_spectrum(self, **kw: object) -> object:
        """
        Return a transmission spectrum plot using the best fit parameters

        Parameters
        ----------
        kw: keywords to pass to 'plot_transmission_spectrum' methods

        Returns
        -------
        object
        """
        self.apply_operation_to_constituent_models("plot_transmission_spectrum", **kw)

    def plot_eclipse_spectrum(self, **kw: object) -> object:
        """
        Return an eclipse spectrum plot using the best fit parameters

        Parameters
        ----------
        kw: keywords to pass to 'plot_eclipse_spectrum' methods

        Returns
        -------
        object
        """
        self.apply_operation_to_constituent_models("plot_eclipse_spectrum", **kw)

    # def plot_all_models(
    #     self,
    #     wavelength=None,
    #     ax: matplotlib.axes.Axes = None,
    #     show_data=True,
    #     **kw: object,
    # ):
    #     if ax is None:
    #         ax = []
    #         if wavelength is None:
    #             for i in range(self.data.nwave):
    #                 # make sure ax is set up
    #                 fi = plt.figure(
    #                     figsize=(8, 4),
    #                     # plt.matplotlib.rcParams["figure.figsize"][::-1],
    #                     constrained_layout=True,
    #                 )
    #                 ax.append(plt.subplot())
    #             # plt.sca(ax)
    #         else:
    #             kw["wavelength"] = wavelength
    #             # make sure ax is set up
    #             fi = plt.figure(
    #                 figsize=(8, 4),
    #                 # plt.matplotlib.rcParams["figure.figsize"][::-1],
    #                 constrained_layout=True,
    #             )
    #             ax.append(plt.subplot())
    #
    #     kw["ax"] = ax
    #     self.apply_operation_to_constituent_models("plot_model", **kw)
    #
    #     if hasattr(self, "_fit_models"):
    #         all_models = self._fit_models
    #     else:
    #         all_models = self.get_model()
    #     if wavelength is None:
    #         for i in range(self.data.nwave):
    #             ax[i].plot(self.data.time, all_models["total"][f"w{i}"], label="total")
    #     else:
    #         ax[0].plot(
    #             self.data.time, all_models["total"][f"w{wavelength}"], label="total"
    #         )
