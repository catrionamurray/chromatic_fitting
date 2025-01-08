import astropy.units.quantity
from ..imports import *
from .lightcurve import *

class SinusoidModel(LightcurveModel):
    """
    An exponential model for the lightcurve.
    """

    def __init__(
            self,
            sinusoid: str,
            independant_variable: str = "time",
            type_of_model: str = "systematic",
            **kw: object,
    ) -> None:
        """
        Initialize the cosine model.

        Parameters
        ----------
        sinusoid: {sine/sin/s} or {cosine/cos/c}?
        independant_variable: the independant variable of the exponential (default = time)
        kw: keyword arguments for initialising the chromatic model
        """
        # only require a constant (0th order) term:
        self.required_parameters = ["A", "w", "phase", "baseline"]

        if sinusoid == "sine" or sinusoid == "sin" or sinusoid == "s":
            self.sinusoid_function = np.sin
            name = "sin"
        elif sinusoid == "cosine" or sinusoid == "cos" or sinusoid == "c":
            self.sinusoid_function = np.cos
            name = "cos"
        else:
            warnings.warn(f"The value for sinusoid entered ({sinusoid}) is not one of 'sin' or 'cos'!")
            return
        self.model = self.sinusoid_model
        self.sinusoid = sinusoid

        super().__init__(**kw)
        self.independant_variable = independant_variable
        self.set_defaults()
        self.set_name(name)
        self.metadata = {}
        self.model = self.sinusoid_model

        if type_of_model in allowed_types_of_models:
            self.type_of_model = type_of_model
        else:
            warnings.warn(
                f"{type_of_model} is not a valid type of model. Please select one of: {allowed_types_of_models}"
            )

    def __repr__(self):
        """
        Print the exponential model.
        """
        return f"<chromatic exponential model '{self.name}' 🌈>"

    def set_defaults(self):
        """
        Set the default parameters for the model.
        """
        self.defaults = dict(A=0.01, w=0.1, phase=0.0, baseline=1.0)

    def what_are_parameters(self):
        """
        Print a summary of what each parameter is
        # """
        self.parameter_descriptions = dict(
            A="The amplitude of the exponential at t0 (defined by the user when initializing).",
            w="",
            phase="",
            baseline="",
        )

        for k, v in self.parameter_descriptions.items():
            print(f"{k}: {v}")

    def setup_lightcurves(self, store_models: bool = False, **kwargs):
        """
        Create an exponential model, given the stored parameters.
        [This should be run after .attach_data()]

        Parameters
        ----------
        store_models: boolean to determine whether to store the lightcurve model during the MCMC fit

        """

        # if the optimization method is "separate" then loop over each wavelength's model/data
        datas, models = self.choose_model_based_on_optimization_method()
        kw = {"shape": datas[0].nwave}

        # if the model has a name then add this to each parameter's name (needed to prevent overwriting parameter names
        # if you combine >1 polynomial model)
        if hasattr(self, "name"):
            name = self.name + "_"
        else:
            name = ""

        # if the .every_light_curve attribute (final lc model) is not already present then create it now
        if not hasattr(self, "every_light_curve"):
            self.every_light_curve = {}
        if not hasattr(self, "initial_guess"):
            self.initial_guess = {}

        # we can decide to store the LC models during the fit (useful for plotting later, however, uses large amounts
        # of RAM)
        if store_models == True:
            self.store_models = store_models

        parameters_to_loop_over = {
            f"{name}A": [],
            f"{name}w": [],
            f"{name}phase": [],
            f"{name}baseline": [],
        }
        for j, (mod, data) in enumerate(zip(models, datas)):
            if self.optimization == "separate":
                kw["i"] = j

            with mod:
                for pname in parameters_to_loop_over.keys():
                    parameters_to_loop_over[pname].append(
                        self.parameters[pname].get_prior_vector(**kw)
                    )

                # get the independent variable from the Rainbow object:
                x = data.get(self.independant_variable)
                # if the independant variable is time, convert to days:
                if self.independant_variable == "time":
                    x = x.to_value("day")
                else:
                    try:
                        x = x.to_value()
                    except AttributeError:
                        pass

                sinusoid, initial_guess = [], []
                for i, w in enumerate(data.wavelength):
                    if len(np.shape(x)) > 1:
                        xi = x[i, :]
                    else:
                        xi = x

                    param_i = {}
                    for pname, param in parameters_to_loop_over.items():
                        if isinstance(self.parameters[pname], WavelikeFitted):
                            param_i[pname] = param[j][i]
                        else:
                            param_i[pname] = param[j]

                    #                     exp.append(
                    #                         param_i[f"{name}A"]
                    #                         * np.exp(-(xi - self.t0) / param_i[f"{name}decay_time"])
                    #                         + param_i[f"{name}baseline"]
                    #                     )
                    sinusoid.append(param_i[f"{name}baseline"] + param_i[f"{name}A"]
                                    * self.sinusoid_function((param_i[f"{name}w"] * xi) + param_i[f"{name}phase"]))

                    initial_guess.append(eval_in_model(sinusoid[-1]))

                # (if we've chosen to) add a Deterministic parameter to the model for easy extraction/plotting
                # later:
                if self.store_models:
                    Deterministic(
                        f"{name}model", pm.math.stack(sinusoid, axis=0)
                    )  # pm.math.sum(poly, axis=0))

                # add the exponential model to the overall lightcurve:
                if f"wavelength_{j}" not in self.every_light_curve.keys():
                    self.every_light_curve[f"wavelength_{j}"] = pm.math.stack(
                        sinusoid, axis=0
                    )
                else:
                    self.every_light_curve[f"wavelength_{j}"] += pm.math.stack(
                        sinusoid, axis=0
                    )

                # add the initial guess to the model:
                if f"wavelength_{j}" not in self.initial_guess.keys():
                    self.initial_guess[f"wavelength_{j}"] = np.array(initial_guess)
                else:
                    self.initial_guess[f"wavelength_{j}"] += initial_guess

    def sinusoid_model(self, params: dict, i: int = 0) -> np.array:
        """
        Return a exponential model, given a dictionary of parameters.

        Parameters
        ----------
        params: dictionary with the parameters of the exponential model
        i: number of the wavelength to fit (default=0)

        Returns
        -------
        np.array: exponential model with the given parameters
        """
        # cosine = []

        # if the optimization method is "separate" then extract wavelength {i}'s data
        if self.optimization == "separate":
            data = self.get_data(i)
        else:
            data = self.get_data()

        x = data.get(self.independant_variable)
        # if the independant variable is time, convert to days:
        if self.independant_variable == "time":
            x = x.to_value("day")

        if len(np.shape(x)) > 1:
            x = x[i, :]

        self.check_and_fill_missing_parameters(params, i)

        cosine = params[f"{self.name}_baseline"] + params[f"{self.name}_A"] \
                 * self.sinusoid_function((params[f"{self.name}_w"] * x) + params[f"{self.name}_phase"])

        return cosine

    def add_model_to_rainbow(self):
        """
        Add the exponential model to the Rainbow object.
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
        r_with_model = data.attach_model(model=model, systematics_model=model)
        # save the Rainbow_with_model for later
        self.data_with_model = r_with_model