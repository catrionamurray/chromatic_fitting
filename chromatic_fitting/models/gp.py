from ..imports import *
from .lightcurve import *
from celerite2.theano import terms, GaussianProcess

allowed_kernels = ["sho", "quasi_periodic"]
possible_params = {
    "sho": ["sigma", "rho", "Q", "w0", "tau"],
    "quasi_periodic": ["sigma", "period", "Q0", "dQ", "f"],
}
required_params = {
    "sho": ["sigma"],
    "quasi_periodic": ["sigma", "period", "Q0", "dQ", "f"],
}
defaults = {
    "sho": dict(sigma=1.0),
    "quasi_periodic": dict(sigma=1.0, period=1.0, dQ=1.0, Q0=1.0 / np.sqrt(2.0), f=0.0),
}
kernel_func = {"sho": terms.SHOTerm, "quasi_periodic": terms.RotationTerm}


class GPModel(LightcurveModel):
    def __init__(
        self,
        kernel,
        name="gp",
        independant_variable="time",
        type_of_model: str = "systematic",
        **kw,
    ):

        if kernel in allowed_kernels:
            self.kernel = kernel
        else:
            warnings.warn(
                f"{kernel} is not a valid type of model. Please select one of: {allowed_kernels}"
            )

        if type_of_model in allowed_types_of_models:
            self.type_of_model = type_of_model
        else:
            warnings.warn(
                f"{type_of_model} is not a valid type of model. Please select one of: {allowed_types_of_models}"
            )

        super().__init__(**kw)
        self.required_parameters = required_params[kernel]
        self.possible_parameters = possible_params[kernel]
        self.independant_variable = independant_variable
        self.set_defaults()
        self.metadata = {}
        self.set_name(name)
        self.model = self.gp_model
        self.kernel_func = kernel_func[self.kernel]

    def set_defaults(self):
        self.defaults = defaults[self.kernel]

    def __repr__(self):
        return f"<chromatic GP model '{self.name}', with {self.kernel} kernel 🌈>"

    def setup_parameters(self, **kw):
        super().setup_parameters(**kw)

        # run quick check on parameters with kernel:
        test_params = {}
        for p in self.parameters.keys():
            test_params[p.replace(f"{self.name}_", "")] = 0.1
        try:
            kernel_func[self.kernel](**test_params)
        except Exception as e:
            print(e)

    def add_jitter(self, log_jitter_parameter):
        self.log_jitter = log_jitter_parameter

    def add_mean(self, mean_parameter):
        self.mean = mean_parameter

    def setup_lightcurves(self, store_models: bool = False):
        """
        Create a GP model, given the stored parameters.
        [This should be run after .attach_data()]
        """

        datas, models = self.choose_model_based_on_optimization_method()
        kw = {"shape": datas[0].nwave}

        # if the model has a name then add this to each parameter's name
        if hasattr(self, "name"):
            name = self.name + "_"
        else:
            name = ""

        if not hasattr(self, "every_light_curve"):
            self.every_light_curve = {}
        if not hasattr(self, "initial_guess"):
            self.initial_guess = {}
        if not hasattr(self, "mean"):
            warnings.warn(
                "\nYou have not provided a mean to the GP. Are you sure this is right?\nIf you want to "
                "add a mean please run {self}.add_mean. We will proceed assuming a mean of 0.\n"
            )
            self.mean = Fixed(0.0)
        if not hasattr(self, "log_jitter"):
            warnings.warn(
                "\nYou have not provided a jitter to the GP. Are you sure this is right?\nIf you want to "
                "add jitter please run {self}.add_jitter. We will proceed assuming no jitter.\n"
            )
            self.log_jitter = Fixed(np.log(0.0))

        if store_models == True:
            self.store_models = store_models

        self.covar = {}

        parameters_to_loop_over = {}
        for p in self.parameters.keys():
            parameters_to_loop_over[p] = []

        for j, (mod, data) in enumerate(zip(models, datas)):
            if self.optimization == "separate":
                kw["i"] = j

            x = data.get(self.independant_variable)
            if self.independant_variable == "time":
                x = x.to_value("day")
            else:
                try:
                    x = x.to_value()
                except AttributeError:
                    pass

            with mod:
                for pname in parameters_to_loop_over.keys():
                    parameters_to_loop_over[pname].append(
                        self.parameters[pname].get_prior_vector(**kw)
                    )

                gp_model, kernel, initial_guess = [], [], []
                for i, w in enumerate(data.wavelength):
                    if len(np.shape(x)) > 1:
                        xi = x[i, :]
                    else:
                        xi = x

                    param_i = {}
                    for pname, param in parameters_to_loop_over.items():
                        if isinstance(self.parameters[pname], WavelikeFitted):
                            param_i[pname.replace(f"{self.name}_", "")] = param[j][i]
                        elif isinstance(self.parameters[pname], Fixed):
                            param_i[pname.replace(f"{self.name}_", "")] = param[0]
                        else:
                            param_i[pname.replace(f"{self.name}_", "")] = param[j][0]

                    print(param_i)
                    kernel.append(self.kernel_func(**param_i))

                    initial_gp = GaussianProcess(
                        kernel[-1],
                        mean=self.mean.get_prior(i + j),
                        t=xi,
                        diag=data.uncertainty[i] ** 2
                        + pm.math.exp(self.log_jitter.get_prior(i + j)),
                        quiet=True,
                    )
                    initial_gp_predict = initial_gp.predict(data.flux[i], t=xi)
                    initial_guess.append(eval_in_model(initial_gp_predict))

                #                 print(eval_in_model(pm.math.zeros_like(pm.math.stack(initial_guess))))

                self.covar[f"wavelength_{j}"] = kernel

                if f"wavelength_{j}" not in self.every_light_curve.keys():
                    self.every_light_curve[f"wavelength_{j}"] = pm.math.zeros_like(
                        pm.math.stack(initial_guess)
                    )
                else:
                    self.every_light_curve[f"wavelength_{j}"] += pm.math.zeros_like(
                        pm.math.stack(initial_guess)
                    )

                if f"wavelength_{j}" not in self.initial_guess.keys():
                    self.initial_guess[f"wavelength_{j}"] = np.array(initial_guess)
                else:
                    self.initial_guess[f"wavelength_{j}"] += np.array(initial_guess)

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
            self.data_without_outliers = remove_data_outliers(datas, data_mask)

        if inflate_uncertainties:
            self.parameters["nsigma"] = WavelikeFitted(
                Uniform, lower=1.0, upper=3.0, testval=1.01
            )
            self.parameters["nsigma"].set_name("nsigma")

        nsigma = []
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

                #                 try:
                x = data.get(self.independant_variable)
                if self.independant_variable == "time":
                    x = x.to_value("day")

                light_curve_name = f"wavelength_{j}"
                self.gp = []
                for i in range(data.nwave):
                    self.gp.append(
                        GaussianProcess(
                            self.covar[light_curve_name][i],
                            t=x,
                            diag=uncertainties[i] ** 2
                            + pm.math.exp(self.log_jitter.get_prior(i + j)),
                            mean=self.mean.get_prior(i + j),
                            quiet=True,
                        )
                    )

                    self.gp[-1].marginal(
                        f"gp_w{j + i}",
                        observed=flux[i] - self.every_light_curve[light_curve_name][i],
                    )
                    # Deterministic(f"gp_pred_w{j+i}", self.gp.predict(flux[i] - self.every_light_curve[light_curve_name][i]))

    def gp_model(self, y):
        return self.gp.predict(y=y)

    def generate_gp_model_from_params(self, params, i=0):
        new_params = {}
        for parname, par in params.items():
            if "interval" not in parname:
                parname_without_modelname = parname.replace(f"{p.name}_", "")
                if f"[{i}]" in parname_without_modelname:
                    try:
                        new_params[
                            parname_without_modelname.replace(f"[{i}]", "")
                        ] = par[i]
                    except:
                        new_params[
                            parname_without_modelname.replace(f"[{i}]", "")
                        ] = par
                else:
                    try:
                        new_params[parname_without_modelname] = par[i]
                    except:
                        new_params[parname_without_modelname] = par

        extra_params = {}
        for name, default_value in zip(["log_jitter", "mean"], [1.0, 0.0]):
            if name in new_params.keys():
                extra_params[name] = new_params[name]
                new_params.pop(name)
            else:
                extra_params[name] = default_value

        kernel = self.kernel_func(**new_params)

        x = self.data.get(self.independant_variable)
        if self.independant_variable == "time":
            x = x.to_value("day")

        gp = GaussianProcess(
            kernel,
            mean=extra_params["mean"],
            t=x,
            diag=self.data.uncertainty[i, :] + np.exp(extra_params["log_jitter"]),
        )

        return gp
