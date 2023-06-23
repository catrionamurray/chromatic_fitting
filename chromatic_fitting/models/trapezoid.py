from ..imports import *

from .lightcurve import *


class TrapezoidModel(LightcurveModel):
    """
    A trapezoid model for the lightcurve.
    """

    def __init__(
        self,
        name: str = "trapezoid",
        type_of_model: str = "planet",
        **kw: object,
    ) -> None:
        """
        Initialize the trapezoid model.

        Parameters
        ----------
        name: the name of the model (default = "trapezoid")
        kw: keyword arguments for initialising the chromatic model
        """
        # only require a constant (0th order) term:
        self.required_parameters = ["delta", "P", "t0", "T", "tau", "baseline"]

        super().__init__(**kw)
        self.set_defaults()
        self.set_name(name)
        self.metadata = {}
        self.model = self.trapezoid_model

        if type_of_model in allowed_types_of_models:
            self.type_of_model = type_of_model
        else:
            warnings.warn(
                f"{type_of_model} is not a valid type of model. Please select one of: {allowed_types_of_models}"
            )

    def __repr__(self):
        """
        Print the trapezoid model.
        """
        return f"<chromatic trapezoid model '{self.name}' 🌈>"

    def set_defaults(self):
        """
        Set the default parameters for the model.
        """
        self.defaults = dict(delta=0.01, P=1, t0=0, T=0.1, tau=0.01, baseline=1.0)

    def what_are_parameters(self):
        """
        Print a summary of what each parameter is
        # """
        self.parameter_descriptions = dict(
            delta="",
            P="The orbital period [d].",
            t0="The epoch of the mid-transit time [d].",
            T="The duration of the transit [d].",
            tau="The duration of the ingress/egress [d].",
            baseline="The out-of-transit flux.",
        )

        for k, v in self.parameter_descriptions.items():
            print(f"{k}: {v}")

    def setup_lightcurves(self, store_models: bool = False, **kwargs):
        """
        Create a trapezoid model, given the stored parameters.
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

        # we can decide to store the LC models during the fit (useful for plotting later, however, uses large amounts
        # of RAM)
        if store_models == True:
            self.store_models = store_models

        # P, t0, tau, T, baseline, delta = [], [], [], [], [], []
        parameters_to_loop_over = {
            f"{name}P": [],
            f"{name}t0": [],
            f"{name}delta": [],
            f"{name}tau": [],
            f"{name}T": [],
            f"{name}baseline": [],
        }
        for j, (mod, data) in enumerate(zip(models, datas)):
            if self.optimization == "separate":
                kw["i"] = j

            with mod:
                # for every wavelength set up a step model
                for pname in parameters_to_loop_over.keys():
                    parameters_to_loop_over[pname].append(
                        self.parameters[pname].get_prior_vector(**kw)
                    )
                # t0.append(self.parameters[f"{name}t0"].get_prior_vector(**kw))
                # P.append(self.parameters[f"{name}P"].get_prior_vector(**kw))
                # delta.append(self.parameters[f"{name}delta"].get_prior_vector(**kw))
                # tau.append(self.parameters[f"{name}tau"].get_prior_vector(**kw))
                # T.append(self.parameters[f"{name}T"].get_prior_vector(**kw))
                # baseline.append(
                #     self.parameters[f"{name}baseline"].get_prior_vector(**kw)
                # )

                trap = []
                for i, w in enumerate(data.wavelength):
                    param_i = {}
                    for pname, param in parameters_to_loop_over.items():
                        if isinstance(self.parameters[pname], WavelikeFitted):
                            param_i[pname] = param[j][i]
                        else:
                            param_i[pname] = param[j]

                    # calculate a phase-folded time (still in units of days)
                    x = (
                        data.time.to_value(u.day)
                        - param_i[f"{name}t0"]
                        + 0.5 * param_i[f"{name}P"]
                    ) % param_i[f"{name}P"] - 0.5 * param_i[f"{name}P"]

                    # Compute the four points where the trapezoid changes slope
                    # x1 <= x2 <= x3 <= x4
                    if eval_in_model(
                        pm.math.gt(param_i[f"{name}tau"], param_i[f"{name}T"])
                    ):
                        x1 = -param_i[f"{name}tau"]
                        x2 = 0
                        x3 = 0
                        x4 = param_i[f"{name}tau"]
                    else:
                        x1 = -(param_i[f"{name}T"] + param_i[f"{name}tau"]) / 2.0
                        x2 = -(param_i[f"{name}T"] - param_i[f"{name}tau"]) / 2.0
                        x3 = (param_i[f"{name}T"] - param_i[f"{name}tau"]) / 2.0
                        x4 = (param_i[f"{name}T"] + param_i[f"{name}tau"]) / 2.0

                    # Compute model values in pieces between the change points
                    range_a = pm.math.and_(pm.math.ge(x, x1), pm.math.lt(x, x2))
                    range_b = pm.math.and_(pm.math.ge(x, x2), pm.math.lt(x, x3))
                    range_c = pm.math.and_(pm.math.ge(x, x3), pm.math.lt(x, x4))

                    if param_i[f"{name}tau"] == 0:
                        slope = pm.math.inf
                    else:
                        slope = param_i[f"{name}delta"] / param_i[f"{name}tau"]
                    val_a = -slope * (x - x1)
                    val_b = -param_i[f"{name}delta"]
                    val_c = -slope * (x4 - x)

                    trap.append(
                        param_i[f"{name}baseline"]
                        * (
                            1
                            + (range_a * val_a)
                            + (range_b * val_b)
                            + (range_c * val_c)
                        )
                    )

                # (if we've chosen to) add a Deterministic parameter to the model for easy extraction/plotting
                # later:
                if self.store_models:
                    Deterministic(f"{name}model", pm.math.stack(trap, axis=0))

                # add the step model to the overall lightcurve:
                if f"wavelength_{j}" not in self.every_light_curve.keys():
                    self.every_light_curve[f"wavelength_{j}"] = pm.math.stack(
                        trap, axis=0
                    )
                else:
                    self.every_light_curve[f"wavelength_{j}"] += pm.math.stack(
                        trap, axis=0
                    )

    def trapezoid_model(self, trap_params: dict, i: int = 0) -> np.array:
        """
        Return a step model, given a dictionary of parameters.

        Parameters
        ----------
        step_params: dictionary with the parameters of the step model
        i: number of the wavelength to fit (default=0)

        Returns
        -------
        np.array: step model with the given parameters
        """

        # if the optimization method is "separate" then extract wavelength {i}'s data
        if self.optimization == "separate":
            data = self.get_data(i)
        else:
            data = self.get_data()

        self.check_and_fill_missing_parameters(trap_params, i)

        P = trap_params[f"{self.name}_P"]
        t0 = trap_params[f"{self.name}_t0"]
        tau = trap_params[f"{self.name}_tau"]
        T = trap_params[f"{self.name}_T"]
        delta = trap_params[f"{self.name}_delta"]
        baseline = trap_params[f"{self.name}_baseline"]

        # calculate a phase-folded time (still in units of days)
        x = (data.time.to_value(u.day) - t0 + 0.5 * P) % P - 0.5 * P

        # Compute the four points where the trapezoid changes slope
        # x1 <= x2 <= x3 <= x4
        if tau > T:
            x1 = -tau
            x2 = 0
            x3 = 0
            x4 = tau
        else:
            x1 = -(T + tau) / 2.0
            x2 = -(T - tau) / 2.0
            x3 = (T - tau) / 2.0
            x4 = (T + tau) / 2.0

        # Compute model values in pieces between the change points
        range_a = np.logical_and(x >= x1, x < x2)
        range_b = np.logical_and(x >= x2, x < x3)
        range_c = np.logical_and(x >= x3, x < x4)

        if tau == 0:
            slope = np.inf
        else:
            slope = delta / tau
        val_a = 1 - slope * (x - x1)
        val_b = 1 - delta
        val_c = 1 - slope * (x4 - x)
        flux = (
            np.select([range_a, range_b, range_c], [val_a, val_b, val_c], default=1)
            * baseline
        )
        return flux

    def add_model_to_rainbow(self):
        """
        Add the trapezoid model to the Rainbow object.
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
        r_with_model = data.attach_model(model=model, planet_model=model)
        # save the Rainbow_with_model for later
        self.data_with_model = r_with_model
