from ..imports import *

from .lightcurve import *

import starry

starry.config.lazy = True
starry.config.quiet = True

import theano

theano.config.gcc__cxxflags += " -fexceptions"


class TransitSpotModel(LightcurveModel):
    """
    A transit and spot model for the lightcurve.
    """

    def __init__(
            self,
            name: str = "transitspot",
            method: str = "starry",
            ydeg: int = 20,
            nspots: int = 1,
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
        if ydeg >= 25:
            warnings.warn(
                "You have selected >=25 spherical harmonic degrees. Starry will be very slow!")
        elif ydeg >= 35:
            warnings.warn(
                "You have selected >=35 spherical harmonic degrees. Starry does not behave nicely at this high a resolution!")

        self.nspots = nspots

        # only require a constant (0th order) term:
        self.required_parameters = ["A", "rs", "ms", "prot", "u", "stellar_amp", "stellar_inc", "stellar_obl",
                                    "mp", "rp", "inc", "amp", "period", "omega", "ecc", "t0"]
        for n in range(self.nspots):
            self.required_parameters.extend([f"spot_{n + 1}_contrast",
                                             f"spot_{n + 1}_radius",
                                             f"spot_{n + 1}_latitude",
                                             f"spot_{n + 1}_longitude"])

        super().__init__(**kw)
        self.set_defaults()
        self.set_name(name)
        self.metadata = {}
        self.model = self.transit_spot_model
        self.ydeg = ydeg
        self.method = method

        if self.method != "starry":
            warnings.warn("Only the starry spot method is currently implemented.")

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
        self.defaults = dict(A=1, rs=1, ms=1, stellar_amp=1, stellar_inc=90, stellar_obl=0.0, prot=3, u=[0.1, 0.1],
                             # spot_contrast=0.5, spot_radius=20, spot_latitude=0.0, spot_longitude=0.0,
                             mp=1, rp=1, inc=90, amp=5e-3, period=3, omega=100, ecc=0.0, t0=0.0)
        for n in range(self.nspots):
            self.defaults[f"spot_{n + 1}_contrast"] = 0.5
            self.defaults[f"spot_{n + 1}_radius"] = 20
            self.defaults[f"spot_{n + 1}_latitude"] = 0.0
            self.defaults[f"spot_{n + 1}_longitude"] = 0.0

    def setup_star_and_planet(self, name, method, param_i, time, flux_model):
        if method == "starry":
            star = starry.Primary(
                starry.Map(ydeg=self.ydeg, udeg=2, amp=param_i[f"{name}stellar_amp"],
                           inc=param_i[f"{name}stellar_inc"], obl=param_i[f"{name}stellar_obl"]),
                r=param_i[f"{name}rs"],
                m=param_i[f"{name}ms"],
                prot=param_i[f"{name}prot"],
                length_unit=u.R_sun,
                mass_unit=u.M_sun,
            )

            star.map[1:] = param_i[f"{name}u"]

            for spot_i in range(self.nspots):
                star.map.spot(contrast=param_i[f"{name}spot_{spot_i + 1}_contrast"],
                              radius=param_i[f"{name}spot_{spot_i + 1}_radius"],
                              lat=param_i[f"{name}spot_{spot_i + 1}_latitude"],
                              lon=param_i[f"{name}spot_{spot_i + 1}_longitude"])

            planet = starry.kepler.Secondary(
                starry.Map(ydeg=5, amp=param_i[f"{name}amp"]),  # the surface map
                m=param_i[f"{name}mp"],  # mass in solar masses
                r=param_i[f"{name}rp"],  # radius
                inc=param_i[f"{name}inc"],
                length_unit=u.R_earth,
                mass_unit=u.M_earth,
                porb=param_i[f"{name}period"],  # orbital period in days
                prot=param_i[f"{name}period"],  # rotation period in days (synchronous)
                omega=param_i[f"{name}omega"],  # longitude of ascending node in degrees
                ecc=param_i[f"{name}ecc"],  # eccentricity
                t0=param_i[f"{name}t0"],  # time of transit in days
            )

            sys = starry.System(star, planet)
            flux_model.append(param_i[f"{name}A"] * sys.flux(time))

        elif method == "fleck":
            print("fleck method not implemented yet")
            return None, None

        return flux_model, sys

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

        # parameters_to_loop_over = {
        #     f"{name}A": [],
        #     f"{name}decay_time": [],
        #     f"{name}baseline": [],
        # }

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

                flux_model, initial_guess = [], []
                for i, w in enumerate(data.wavelength):

                    param_i = {}
                    for pname, param in parameters_to_loop_over.items():
                        if isinstance(self.parameters[pname], WavelikeFitted):
                            param_i[pname] = param[j][i]
                        elif isinstance(self.parameters[pname], Fixed):
                            param_i[pname] = param[j]
                        else:
                            param_i[pname] = param[j][0]

                    flux_model, _ = self.setup_star_and_planet(name, self.method, param_i, data.time.to_value('d'), flux_model)

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
                        f"{name}model", pm.math.stack(flux_model, axis=0)
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

        with pm.Model() as temp_model:
            flux_model, sys = self.setup_star_and_planet(f"{self.name}_", self.method, params, data.time.to_value('d'), [])
            # star = starry.Primary(
            #     starry.Map(ydeg=self.ydeg,
            #                udeg=2,
            #                amp=params[f"{self.name}_stellar_amp"],
            #                inc=params[f"{self.name}_stellar_inc"],
            #                obl=params[f"{self.name}_stellar_obl"],
            #                ),
            #     r=params[f"{self.name}_rs"],
            #     m=params[f"{self.name}_ms"],
            #     prot=params[f'{self.name}_prot'],
            #     length_unit=u.R_sun,
            #     mass_unit=u.M_sun,
            # )
            # star.map[1:] = params[f"{self.name}_u"]
            #
            # for spot_i in range(self.nspots):
            #     star.map.spot(contrast=params[f"{self.name}_spot_{spot_i + 1}_contrast"],
            #                   radius=params[f"{self.name}_spot_{spot_i + 1}_radius"],
            #                   lat=params[f"{self.name}_spot_{spot_i + 1}_latitude"],
            #                   lon=params[f"{self.name}_spot_{spot_i + 1}_longitude"])
            #
            # planet = starry.kepler.Secondary(
            #     starry.Map(ydeg=5, amp=params[f"{self.name}_amp"]),  # the surface map
            #     m=params[f'{self.name}_mp'],  # mass in solar masses
            #     r=params[f'{self.name}_rp'],  # radius
            #     inc=params[f"{self.name}_inc"],
            #     length_unit=u.R_earth,
            #     mass_unit=u.M_earth,
            #     porb=params[f'{self.name}_period'],  # orbital period in days
            #     prot=params[f'{self.name}_period'],  # rotation period in days (synchronous)
            #     omega=params[f'{self.name}_omega'],  # longitude of ascending node in degrees
            #     ecc=params[f'{self.name}_ecc'],  # eccentricity
            #     t0=params[f'{self.name}_t0'],  # time of transit in days
            # )

            # sys = starry.System(star, planet)
            self.keplerian_system = sys
            transit_spot = eval_in_model(flux_model[0])
            # transit_spot = params[f"{self.name}_A"] * eval_in_model(sys.flux(data.time.to_value('d')))
        return transit_spot

    def multiwavelength_map(self, params: dict):

        for i in range(self.nwave):
            self.check_and_fill_missing_parameters(params, i)

        with pm.Model() as temp_model:
            star = starry.Primary(
                starry.Map(ydeg=self.ydeg,
                           nw=self.nwave,
                           udeg=2,
                           amp=params[f"{self.name}_stellar_amp"],
                           inc=params[f"{self.name}_stellar_inc"],
                           obl=params[f"{self.name}_stellar_obl"]),
                r=params[f"{self.name}_rs"],
                m=params[f"{self.name}_ms"],
                prot=params[f'{self.name}_prot'],
                length_unit=u.R_sun,
                mass_unit=u.M_sun,
            )
            star.map[1:] = params[f"{self.name}_u"]

            # star.map.spot(contrast=params[f'{self.name}_spot_contrast'],
            #               radius=params[f'{self.name}_spot_radius'],
            #               lat=params[f'{self.name}_spot_latitude'],
            #               lon=params[f'{self.name}_spot_longitude'])

            for spot_i in range(self.nspots):
                star.map.spot(contrast=params[f"{self.name}_spot_{spot_i + 1}_contrast"],
                              radius=params[f"{self.name}_spot_{spot_i + 1}_radius"],
                              lat=params[f"{self.name}_spot_{spot_i + 1}_latitude"],
                              lon=params[f"{self.name}_spot_{spot_i + 1}_longitude"])

    def show(self, **kw):
        if self.method == "starry":
            if hasattr(self, 'keplerian_system'):
                self.keplerian_system.show(**kw)

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
