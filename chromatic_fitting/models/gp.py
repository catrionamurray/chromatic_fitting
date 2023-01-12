from ..imports import *
from .lightcurve import *
import warnings


class GPModel(LightcurveModel):
    """
    An gaussian process model for the lightcurve.
    """

    try:
        from celerite2.theano import terms, GaussianProcess
    except ImportError:
        print(
            "Celerite2 is not installed, please install this program first before creating a GP Model."
        )

    def __init__(
        self,
        independant_variable: str = "time",
        name: str = "gp",
        **kw: object,
    ) -> None:
        """
        Initialize the GP model.

        Parameters
        ----------
        independant_variable: the independant variable of the GP (default = time)
        name: the name of the model (default = "gp")
        kw: keyword arguments for initialising the chromatic model
        """

        # only require a constant (0th order) term:
        self.required_parameters = ["A", "decay_time", "baseline"]

        super().__init__(**kw)
        self.independant_variable = independant_variable
        self.set_defaults()
        self.set_name(name)
        self.metadata = {}
        # self.model = self.exponential_model

    def __repr__(self):
        """
        Print the GP model.
        """
        return f"<chromatic GP model '{self.name}' 🌈>"

    def set_defaults(self):
        """
        Set the default parameters for the model.
        """
        self.defaults = dict()

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
        try:
            from celerite2.theano import GaussianProcess
        except ImportError:
            print(
                "Celerite2 is not installed, please install this program first before creating a GP Model."
            )

        if hasattr(self, "every_light_curve"):
            if f"wavelength_0" not in self.every_light_curve.keys():
                print(".setup_lightcurves() has not been run yet, running now...")
                self.setup_lightcurves(**setup_lightcurves_kw)
        else:
            print(".setup_lightcurves() has not been run yet, running now...")
            self.setup_lightcurves(**setup_lightcurves_kw)

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
                Uniform, lower=1.0, upper=3.0, testval=1.01
            )
            self.parameters["nsigma"].set_name("nsigma")

        nsigma = []
        self.gp = {}

        for j, (mod, data) in enumerate(zip(models, datas)):
            x = data.get(self.independant_variable)
            if self.independant_variable == "time":
                x = x.to_value("day")
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
                    # data_name = f"data"
                    light_curve_name = f"wavelength_{j}"

                    self.gp[light_curve_name] = GaussianProcess(
                        self.kernel,
                        t=x,
                        diag=uncertainties**2
                        + pm.math.exp(2 * self._gp_meta[f"{self.name}_log_jitter"]),
                        mean=self._gp_meta[f"{self.name}_mean"],
                        # self.parameters[f"{self.name}_mean"].get_prior(i + j),
                        quiet=True,
                    )

                    self.gp.marginal(
                        f"{k}_gp",
                        observed=flux - self.every_light_curve[light_curve_name],
                    )

                    Deterministic(
                        f"gp_pred_w{j}",
                        self.gp.predict(
                            flux - self.every_light_curve[light_curve_name]
                        ),
                    )
                except Exception as e:
                    print(e)

    def gp_model(self, y):
        return self.gp.predict(y=y)


class SHOModel(GPModel):
    def __init__(self, name="gp_sho", independant_variable="time", **kw):
        self.required_parameters = ["log_sigma", "log_rho", "Q", "log_jitter", "mean"]

        super().__init__(name, independant_variable, **kw)
        self.independant_variable = independant_variable
        self.set_defaults()
        self.set_name(name)

    def __repr__(self):
        return "<chromatic GP (simple harmonic oscillator) model 🌈>"

    def set_defaults(self):
        self.defaults = dict(
            log_sigma=0.0, log_rho=0.0, Q=1.0 / np.sqrt(2.0), log_jitter=0.0, mean=1.0
        )

    def setup_lightcurves(self, store_models=False):
        """
        Create a GP model, given the stored parameters.
        [This should be run after .attach_data()]
        """

        self._gp_meta = {}

        # if the optimization method is "separate" then loop over each wavelength's model/data
        if self.optimization == "separate":
            models = self._pymc3_model
            datas = [self.get_data(i) for i in range(self.data.nwave)]
        else:
            models = [self._pymc3_model]
            datas = [self.get_data()]
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

        parameters_to_loop_over = {
            f"{name}log_sigma": [],
            f"{name}log_rho": [],
            f"{name}Q": [],
            f"{name}log_jitter": [],
            f"{name}mean": [],
        }

        for j, (mod, data) in enumerate(zip(models, datas)):
            if self.optimization == "separate":
                kw["i"] = j

            with mod:
                for pname in parameters_to_loop_over.keys():
                    parameters_to_loop_over[pname].append(
                        self.parameters[pname].get_prior_vector(**kw)
                    )

                self._gp_meta[f"{name}log_jitter"] = parameters_to_loop_over[
                    f"{name}log_jitter"
                ]
                self._gp_meta[f"{name}mean"] = parameters_to_loop_over[f"{name}mean"]

                kernel = []
                for i, w in enumerate(data.wavelength):
                    param_i = {}
                    for pname, param in parameters_to_loop_over.items():
                        if isinstance(self.parameters[pname], WavelikeFitted):
                            param_i[pname] = param[j][i]
                        else:
                            param_i[pname] = param[j]

                    kernel.append(
                        terms.SHOTerm(
                            sigma=pm.math.exp(param_i[f"{name}log_sigma"]),
                            rho=pm.math.exp(param_i[f"{name}log_rho"]),
                            Q=param_i[f"{name}Q"],
                        )
                    )

                # gp.append(GaussianProcess(
                #     kernel,
                #     t=x,
                #     diag=data.uncertainty ** 2 + tt.exp(2 * log_jitter),
                #     mean=pm.math.exp(param_i[f"{name}mean"],
                #     quiet=True,
                # )

                # (if we've chosen to) add a Deterministic parameter to the model for easy extraction/plotting
                # later:
                if self.store_models:
                    Deterministic(f"{name}model", pm.math.stack(kernel, axis=0))
                self.kernel = pm.math.stack(kernel, axis=0)

                # add the exponential model to the overall lightcurve:
                if f"wavelength_{j}" not in self.every_light_curve.keys():
                    self.every_light_curve[f"wavelength_{j}"] = np.zeros(len(data.time))
                else:
                    self.every_light_curve[f"wavelength_{j}"] += np.zeros(
                        len(data.time)
                    )


# class QuasiPeriodicModel(GPModel):
#     def __init__(self, name="gp_qp", independant_variable="time", **kw):
#         self.required_parameters = [
#             "log_sigma",
#             "log_period",
#             "log_Q",
#             "log_dQ",
#             "f",
#             "log_jitter",
#             "mean",
#         ]
#
#         super().__init__(name, independant_variable, **kw)
#         self.independant_variable = independant_variable
#         self.set_defaults()
#         self.set_name(name)
#
#     def __repr__(self):
#         return "<chromatic GP (rotation kernel) model 🌈>"
#
#     def set_defaults(self):
#         self.defaults = dict(
#             log_sigma=0.0,
#             log_period=0.0,
#             log_dQ=0.0,
#             log_Q=np.log(1.0 / np.sqrt(2.0)),
#             f=0.0,
#             log_jitter=0.0,
#             mean=1.0,
#         )
#
#     def setup_lightcurves(self):
#         """
#         Create a GP model, given the stored parameters.
#         [This should be run after .attach_data()]
#         """
#
#         if self.optimization == "separate":
#             models = self.pymc3_model
#             datas = [self.get_data(i) for i in range(self.data.nwave)]
#         else:
#             models = [self.pymc3_model]
#             datas = [self.get_data()]
#
#         # if the model has a name then add this to each parameter's name
#         if hasattr(self, "name"):
#             name = self.name + "_"
#         else:
#             name = ""
#
#         if not hasattr(self, "every_light_curve"):
#             self.every_light_curve = {}
#
#         self.kernel = {}
#
#         for j, (mod, data) in enumerate(zip(models, datas)):
#             x = data.get(self.independant_variable)
#             if self.independant_variable == "time":
#                 x = x.to_value("day")
#             with mod:
#                 for i, w in enumerate(data.wavelength):
#                     if len(np.shape(x)) > 1:
#                         x = x[i, :]
#
#                     kernel = terms.RotationTerm(
#                         sigma=pm.math.exp(
#                             self.parameters[f"{name}log_sigma"].get_prior(i + j)
#                         ),
#                         period=pm.math.exp(
#                             self.parameters[f"{name}log_period"].get_prior(i + j)
#                         ),
#                         Q0=pm.math.exp(
#                             self.parameters[f"{name}log_Q"].get_prior(i + j)
#                         ),
#                         dQ=pm.math.exp(
#                             self.parameters[f"{name}log_dQ"].get_prior(i + j)
#                         ),
#                         f=self.parameters[f"{name}f"].get_prior(i + j),
#                     )
#
#                     self.kernel[f"wavelength_{i + j}"] = kernel
#
#                     if f"wavelength_{i + j}" not in self.every_light_curve.keys():
#                         self.every_light_curve[f"wavelength_{i + j}"] = np.zeros(
#                             len(data.time)
#                         )  # self.gp
#                     else:
#                         self.every_light_curve[f"wavelength_{i + j}"] += np.zeros(
#                             len(data.time)
#                         )  # self.gp
