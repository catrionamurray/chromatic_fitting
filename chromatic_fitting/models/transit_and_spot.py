from ..imports import *

from .lightcurve import *


class TransitSpotModel(LightcurveModel):
    """
    A transit and spot model for the lightcurve.
    """

    def __init__(
            self,
            name: str = "transitspot",
            type_of_model: str = "planet",
            **kw: object,
    ) -> None:
        """
        Initialize the transit+spot model.

        Parameters
        ----------
        t0: where the exponential = the amplitude (traditionally the first data point)
        independant_variable: the independant variable of the exponential (default = time)
        name: the name of the model (default = "exponential")
        kw: keyword arguments for initialising the chromatic model
        """
        # only require a constant (0th order) term:
        self.required_parameters = ["A", "rs", "ms", "prot", "u",
                                    "spot_contrast", "spot_radius", "spot_latitude", "spot_longitude",
                                    "mp", "rp", "inc", "period", "omega", "ecc", "t0"]

        super().__init__(**kw)
        self.set_defaults()
        self.set_name(name)
        self.metadata = {}
        self.model = self.transit_spot_model

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
        self.defaults = dict(A=1, rs=1, ms=1, prot=3, u=[0.1,0.1],
                            spot_contrast=0.5, spot_radius=20, spot_latitude=0.0, spot_longitude=0.0,
                            mp=1, rp=1, inc=90, period=3, omega=100, ecc=0.0, t0=0.0)

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
            f"{name}decay_time": [],
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

                flux_model, initial_guess = [], []
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

                    star = starry.Primary(
                        starry.Map(ydeg=5, udeg=2, amp=1),
                        r=param_i["rs"],
                        m=param_i["ms"],
                        prot=param_i['prot'],
                        length_unit=u.R_sun,
                        mass_unit=u.M_sun,
                    )
                    star.map[1:] = param_i["u"]

                    star.map.spot(contrast=param_i['spot_contrast'],
                                  radius=param_i['spot_radius'],
                                  lat=param_i['spot_latitude'],
                                  lon=param_i['spot_longitude'])

                    planet = starry.kepler.Secondary(
                        starry.Map(ydeg=5, amp=5e-3),  # the surface map
                        m=param_i['mp'],  # mass in solar masses
                        r=param_i['rp'],  # radius
                        inc=param_i["inc"],
                        length_unit=u.R_earth,
                        mass_unit=u.M_earth,
                        porb=param_i['period'],  # orbital period in days
                        prot=param_i['period'],  # rotation period in days (synchronous)
                        omega=param_i['omega'],  # longitude of ascending node in degrees
                        ecc=param_i['ecc'],  # eccentricity
                        t0=param_i['t0'],  # time of transit in days
                    )

                    sys = starry.System(star, planet)
                    flux_model.append(param_i['A'] * sys.flux(data.time.to_value('d')))

                    # exp.append(
                    #     param_i[f"{name}A"]
                    #     * np.exp(-(xi - self.t0) / param_i[f"{name}decay_time"])
                    #     + param_i[f"{name}baseline"]
                    # )

                    initial_guess.append(eval_in_model(flux_model[-1]))

                # (if we've chosen to) add a Deterministic parameter to the model for easy extraction/plotting
                # later:
                if self.store_models:
                    Deterministic(
                        f"{name}model", pm.math.stack(exp, axis=0)
                    )  # pm.math.sum(poly, axis=0))

                # add the exponential model to the overall lightcurve:
                if f"wavelength_{j}" not in self.every_light_curve.keys():
                    self.every_light_curve[f"wavelength_{j}"] = pm.math.stack(
                        flux_model, axis=0
                    )
                else:
                    self.every_light_curve[f"wavelength_{j}"] += pm.math.stack(
                        flux_model, axis=0
                    )

                # add the initial guess to the model:
                if f"wavelength_{j}" not in self.initial_guess.keys():
                    self.initial_guess[f"wavelength_{j}"] = np.array(initial_guess)
                else:
                    self.initial_guess[f"wavelength_{j}"] += initial_guess

    def transit_spot_model(self, params: dict, i: int = 0) -> np.array:
        """
        Return a exponential model, given a dictionary of parameters.

        Parameters
        ----------
        params: dictionary with the parameters of the model
        i: number of the wavelength to fit (default=0)

        Returns
        -------
        np.array: exponential model with the given parameters
        """
        # transit_spot = []

        # if the optimization method is "separate" then extract wavelength {i}'s data
        if self.optimization == "separate":
            data = self.get_data(i)
        else:
            data = self.get_data()

        self.check_and_fill_missing_parameters(params, i)

        star = starry.Primary(
            starry.Map(ydeg=5, udeg=2, amp=1),
            r=params[f"{self.name}_rs"],
            m=params[f"{self.name}_ms"],
            prot=params[f'{self.name}_prot'],
            length_unit=u.R_sun,
            mass_unit=u.M_sun,
        )
        star.map[1:] = params[f"{self.name}_u"]

        star.map.spot(contrast=params[f'{self.name}_spot_contrast'],
                      radius=params[f'{self.name}_spot_radius'],
                      lat=params[f'{self.name}_spot_latitude'],
                      lon=params[f'{self.name}_spot_longitude'])

        planet = starry.kepler.Secondary(
            starry.Map(ydeg=5, amp=5e-3),  # the surface map
            m=params[f'{self.name}_mp'],  # mass in solar masses
            r=params[f'{self.name}_rp'],  # radius
            inc=params[f"{self.name}_inc"],
            length_unit=u.R_earth,
            mass_unit=u.M_earth,
            porb=params[f'{self.name}_period'],  # orbital period in days
            prot=params[f'{self.name}_period'],  # rotation period in days (synchronous)
            omega=params[f'{self.name}_omega'],  # longitude of ascending node in degrees
            ecc=params[f'{self.name}_ecc'],  # eccentricity
            t0=params[f'{self.name}_t0'],  # time of transit in days
        )

        sys = starry.System(star, planet)
        transit_spot = params[f"{self.name}_A"] * sys.flux(data.time.to_value('d'))
        return transit_spot

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
        r_with_model = data.attach_model(model=model, planet_model=model)
        # save the Rainbow_with_model for later
        self.data_with_model = r_with_model
