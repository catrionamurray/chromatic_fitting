import astropy.units.quantity
from ..imports import *
from .lightcurve import *
from scipy import special


def flare_eqn_mendoza2022(t, ampl, pymc):
    '''
    The equation that defines the shape for the Continuous Flare Model
    '''
    # Values were fit & calculated using MCMC 256 walkers and 30000 steps

    A, B, C, D1, D2, f1 = [0.9687734504375167, -0.251299705922117, 0.22675974948468916,
                           0.15551880775110513, 1.2150539528490194, 0.12695865022878844]

    # We include the corresponding errors for each parameter from the MCMC analysis

    A_err, B_err, C_err, D1_err, D2_err, f1_err = [0.007941622683556804, 0.0004073709715788909, 0.0006863488251125649,
                                                   0.0013498012884345656, 0.00453458098656645, 0.001053149344530907]

    f2 = 1 - f1

    if pymc:
        erfc = pm.math.erfc
    else:
        erfc = special.erfc

    eqn = ((1 / 2) * np.sqrt(np.pi) * A * C * f1 * np.exp(-D1 * t + ((B / C) + (D1 * C / 2)) ** 2)
           * erfc(((B - t) / C) + (C * D1 / 2))) + ((1 / 2) * np.sqrt(np.pi) * A * C * f2
                                                    * np.exp(
                -D2 * t + ((B / C) + (D2 * C / 2)) ** 2) * erfc(((B - t) / C) + (C * D2 / 2)))

    if pymc:
        eqn = pm.math.where(pm.math.eq(erfc(((B - t) / C) + (C * D2 / 2)), 0),
                            pm.math.zeros_like(eqn),
                            eqn)
    else:
        eqn[erfc(((B - t) / C) + (C * D2 / 2)) == 0] = 0

    return eqn * ampl


def flare_model_mendoza2022(t, tpeak, fwhm, ampl, pymc=False):
    '''
    The Continuous Flare Model evaluated for single-peak (classical) flare events.
    Use this function for fitting classical flares with most curve_fit
    tools.
    References
    --------------
    Tovar Mendoza et al. (2022) DOI 10.3847/1538-3881/ac6fe6
    Davenport et al. (2014) http://arxiv.org/abs/1411.3723
    Jackman et al. (2018) https://arxiv.org/abs/1804.03377
    Parameters
    ----------
    t : 1-d array
        The time array to evaluate the flare over
    tpeak : float
        The center time of the flare peak
    fwhm : float
        The Full Width at Half Maximum, timescale of the flare
    ampl : float
        The amplitude of the flare
    Returns
    -------
    flare : 1-d array
        The flux of the flare model evaluated at each time
        A continuous flare template whose shape is defined by the convolution of a Gaussian and double exponential
        and can be parameterized by three parameters: center time (tpeak), FWHM, and ampitude
    '''

    t_new = (t - tpeak) / fwhm
    flare = flare_eqn_mendoza2022(t_new, ampl, pymc)

    return flare

# def aflare(t, p):
#     """
#     This is the Analytic Flare Model from the flare-morphology paper.
#     Reference Davenport et al. (2014) http://arxiv.org/abs/1411.3723
#
#     Note: this model assumes the flux before the flare is zero centered
#
#     Note: many sub-flares can be modeled by this method by changing the
#     number of parameters in "p". As a result, this routine may not work
#     for fitting with methods like scipy.optimize.curve_fit, which require
#     a fixed number of free parameters. Instead, for fitting a single peak
#     use the aflare1 method.
#
#     Parameters
#     ----------
#     t : 1-d array
#         The time array to evaluate the flare over
#     p : 1-d array
#         p == [tpeak, fwhm (units of time), amplitude (units of flux)] x N
#
#     Returns
#     -------
#     flare : 1-d array
#         The flux of the flare model evaluated at each time
#     """
#     _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
#     _fd = [0.689008, -1.60053, 0.302963, -0.278318]
#
#     Nflare = int(np.floor((len(p) / 3.0)))
#
#     flare = np.zeros_like(t)
#     # compute the flare model for each flare
#     for i in range(Nflare):
#         outm = np.piecewise(t, [(t <= p[0 + i * 3]) * (t - p[0 + i * 3]) / p[1 + i * 3] > -1.,
#                                 (t > p[0 + i * 3])],
#                             [lambda x: (_fr[0] +  # 0th order
#                                         _fr[1] * ((x - p[0 + i * 3]) / p[1 + i * 3]) +  # 1st order
#                                         _fr[2] * ((x - p[0 + i * 3]) / p[1 + i * 3]) ** 2. +  # 2nd order
#                                         _fr[3] * ((x - p[0 + i * 3]) / p[1 + i * 3]) ** 3. +  # 3rd order
#                                         _fr[4] * ((x - p[0 + i * 3]) / p[1 + i * 3]) ** 4.),  # 4th order
#                              lambda x: (_fd[0] * np.exp(((x - p[0 + i * 3]) / p[1 + i * 3]) * _fd[1]) +
#                                         _fd[2] * np.exp(((x - p[0 + i * 3]) / p[1 + i * 3]) * _fd[3]))]
#                             ) * p[2 + i * 3]  # amplitude
#         flare = flare + outm
#
#     return flare


def flare_model_davenport2014(t, tpeak, fwhm, ampl, pymc=False):
    '''
    The Analytic Flare Model evaluated for a single-peak (classical).
    Reference Davenport et al. (2014) http://arxiv.org/abs/1411.3723

    Use this function for fitting classical flares with most curve_fit
    tools.

    Note: this model assumes the flux before the flare is zero centered

    Parameters
    ----------
    t : 1-d array
        The time array to evaluate the flare over
    tpeak : float
        The time of the flare peak
    fwhm : float
        The "Full Width at Half Maximum", timescale of the flare
    ampl : float
        The amplitude of the flare

    Returns
    -------
    flare : 1-d array
        The flux of the flare model evaluated at each time
    '''
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]

    fl = np.piecewise(t, [(t <= tpeak) * (t - tpeak) / fwhm > -1.,
                          (t > tpeak)],
                      [lambda x: (_fr[0] +  # 0th order
                                  _fr[1] * ((x - tpeak) / fwhm) +  # 1st order
                                  _fr[2] * ((x - tpeak) / fwhm) ** 2. +  # 2nd order
                                  _fr[3] * ((x - tpeak) / fwhm) ** 3. +  # 3rd order
                                  _fr[4] * ((x - tpeak) / fwhm) ** 4.),  # 4th order
                       lambda x: (_fd[0] * np.exp(((x - tpeak) / fwhm) * _fd[1]) +
                                  _fd[2] * np.exp(((x - tpeak) / fwhm) * _fd[3]))]
                      )

    flare = fl * np.abs(ampl) / np.max(fl)

    return flare


class FlareModel(LightcurveModel):
    """
    A flare model for the lightcurve.
    """

    def __init__(
            self,
            independant_variable: str = "time",
            name: str = "flare",
            flare_model="mendoza2022",
            type_of_model: str = "systematic",
            **kw: object,
    ) -> None:
        """
        Initialize the flare model.

        Parameters
        ----------
        independant_variable: the independant variable of the flare (default = time)
        name: the name of the model (default = "flare")
        kw: keyword arguments for initialising the chromatic model
        """
        # only require a constant (0th order) term:
        self.required_parameters = ["logA", "tpeak", "logfwhm"]

        super().__init__(**kw)
        self.independant_variable = independant_variable
        self.set_defaults()
        self.set_name(name)
        self.metadata = {}
        self.model = self.flare_model
        self.flare_model_method = flare_model

        if type_of_model in allowed_types_of_models:
            self.type_of_model = type_of_model
        else:
            warnings.warn(
                f"{type_of_model} is not a valid type of model. Please select one of: {allowed_types_of_models}"
            )

    def __repr__(self):
        """
        Print the flare model.
        """
        return f"<chromatic flare model '{self.name}' 🌈>"

    def set_defaults(self):
        """
        Set the default parameters for the model.
        """
        self.defaults = dict(logA=-2, tpeak=0.0, logfwhm=-2)

    def what_are_parameters(self):
        """
        Print a summary of what each parameter is
        # """
        self.parameter_descriptions = dict(
            logA="The log amplitude of the flare.",
            tpeak="The time of the flare peak.",
            logfwhm="The log fwhm (width) of the flare",
        )

        for k, v in self.parameter_descriptions.items():
            print(f"{k}: {v}")

    def setup_lightcurves(self, store_models: bool = False, **kwargs):
        """
        Create a flare model, given the stored parameters.
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

        parameters_to_loop_over = {}
        for p in self.parameters.keys():
            parameters_to_loop_over[p] = []

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

                fl, initial_guess = [], []
                for i, w in enumerate(data.wavelength):
                    if len(np.shape(x)) > 1:
                        xi = x[i, :]
                    else:
                        xi = x

                    param_i = {}
                    for pname, param in parameters_to_loop_over.items():
                        if isinstance(self.parameters[pname], WavelikeFitted):
                            param_i[pname] = param[j][i]
                        elif isinstance(self.parameters[pname], Fixed):
                            param_i[pname] = param[j]
                        else:
                            param_i[pname] = param[j][0]

                    try:
                        fwhm = Deterministic(f"{name}fwhm[{i + j}]", 10 ** param_i[f"{name}logfwhm"])
                    except:
                        fwhm = 10 ** param_i[f"{name}logfwhm"]

                    try:
                        A = Deterministic(f"{name}A[{i + j}]", 10 ** param_i[f"{name}logA"])
                    except:
                        A = 10 ** param_i[f"{name}logA"]

                    fl.append(self.flare(param_i, xi, pymc=True))
                    initial_guess.append(eval_in_model(fl[-1]))

                # (if we've chosen to) add a Deterministic parameter to the model for easy extraction/plotting
                # later:
                if self.store_models:
                    Deterministic(
                        f"{name}model", pm.math.stack(fl, axis=0)
                    )  # pm.math.sum(poly, axis=0))

                # add the exponential model to the overall lightcurve:
                if f"wavelength_{j}" not in self.every_light_curve.keys():
                    self.every_light_curve[f"wavelength_{j}"] = pm.math.stack(
                        fl, axis=0
                    )
                else:
                    self.every_light_curve[f"wavelength_{j}"] += pm.math.stack(
                        fl, axis=0
                    )

                # add the initial guess to the model:
                if f"wavelength_{j}" not in self.initial_guess.keys():
                    self.initial_guess[f"wavelength_{j}"] = np.array(initial_guess)
                else:
                    self.initial_guess[f"wavelength_{j}"] += initial_guess

    def flare(self, params, x, pymc=False):
        # if pymc:
        #     fwhm = Deterministic(f"{self.name}_fwhm", 10 ** params[f"{self.name}_logfwhm"])
        #     A = Deterministic(f"{self.name}_A", 10 ** params[f"{self.name}_logA"])

        if self.flare_model_method == "mendoza2022":
            flare = flare_model_mendoza2022(x, params[f"{self.name}_tpeak"],
                                            10 ** params[f"{self.name}_logfwhm"],
                                            10 ** params[f"{self.name}_logA"], pymc)
        elif self.flare_model_method == "davenport2014":
            flare = flare_model_davenport2014(x, params[f"{self.name}_tpeak"],
                                              10 ** params[f"{self.name}_logfwhm"],
                                              10 ** params[f"{self.name}_logA"], pymc)
        return flare

    def flare_model(self, params: dict, i: int = 0) -> np.array:
        """
        Return a flare model, given a dictionary of parameters.

        Parameters
        ----------
        params: dictionary with the parameters of the flare model
        i: number of the wavelength to fit (default=0)

        Returns
        -------
        np.array: flare model with the given parameters
        """
        flare = []

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
        flare = self.flare(params, x)

        return flare

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