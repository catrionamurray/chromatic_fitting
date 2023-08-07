from ..imports import *

from .lightcurve import *
import warnings

# these are important to hard code in for starry to play nice with pymc3
starry.config.lazy = True
starry.config.quiet = True

"""
Example of setting up a PhaseCurveModel:

def create_new_eclipse_model():
    # create eclipse model:
    e = EclipseModel()

    # add out parameters:
    e.setup_parameters(
                stellar_radius =  1.0,   #r_sun
                stellar_mass = 1.0,      #m_sun
                stellar_amplitude = 1.0, #baseline for lightcurve
                stellar_prot = 1.0,      #days
                period = 5.3587657,      #planet period
                t0 = 2458412.70851,      #t0 of transit
                planet_log_amplitude = WavelikeFitted(pm.Uniform,lower=-6.0,upper=-2.0,testval=-4.0),#Whitelight fit
                inclination = 89.68,     #planet inclination
                planet_mass = 0.00903,   #m_jup
                planet_radius = 0.1164,  #r_jup
                ecs =  Fitted(pm.TruncatedNormal,mu=np.array([0.0,0.0]),sigma=np.array([0.01,0.01]),upper=np.array([0.1,0.1]),lower=np.array([-0.1,-0.1]),testval=np.array([0.001,0.001]), shape=2),#2-d parameter which is [ecosw, esinw]
                limb_darkening = np.array([0.4,0.2]),         #limb darkening quadratic coeff np.array([1,2])
            )    

    # attach a Rainbow object, r, to the model:
    e.attach_data(r)

    # setup the lightcurves for the eclipse model and relate the "actual" data to the model (using a Normal likelihood function)
    e.setup_likelihood()    

    # MCMC (NUTS) sample the parameters:
    e.sample(tune=2000, draws=2000, chains=4, cores=4)

    # summarize the results:
    e.summarize(round_to=7, fmt='wide')

    return e

"""


class PhaseCurveModel(LightcurveModel):
    """
    A phase curve model for the lightcurve.
    """

    def __init__(self,
                 name: str = "phasecurve",
                 n_spherical_harmonics: int = 1,  # simple dipole map
                 # spherical_harmonics_coeffs=[1.0, 0.0, 0.5, 0.0],  # simple dipole map
                 # fit_spherical_harmonic_coeffs=False,
                 source_npts: int = 1,
                 type_of_model: str = "planet",
                 **kw: object) -> None:
        """
        Initialise the phase curve model.

        Parameters
        ----------
        name: the name of the model ('phasecurve' by default)
        n_spherical_harmonics: Number of spherical harmonic orders to use to model the star (higher value = resolving smaller features)
        spherical_harmonics_coeffs: Coefficients for each spherical harmonic order: [Y_0,0, Y_1,-1, Y_1,0, Y_1,1, Y_2,-2, ...]. Default is a simple dipole map.
        See https://starry.readthedocs.io/en/latest/notebooks/Basics/ for more info on how to define the coefficients.
        source_npts: Number of points to model illumination source, 1=a point source, so for close-in planets we want this to be >30-50
        type_of_model: is this a planet or systematic model?
        kw: any keywords to pass to the lightcurve model
        """

        # require the following phase curve parameters to initialize the model:
        self.required_parameters = [
            "stellar_radius",
            "stellar_mass",
            "stellar_amplitude",
            "stellar_prot",
            "stellar_inc",
            "stellar_obl",
            "period",
            "t0",
            "planet_log_amplitude",
            "phase_offset",
            "inclination",
            "planet_mass",
            "planet_radius",
            "ecs",
            "limb_darkening",
            "planet_surface_map"
        ]

        if n_spherical_harmonics >= 25:
            warnings.warn(
                "You have selected >=25 spherical harmonic degrees. Starry will be very slow!")
        elif n_spherical_harmonics >= 35:
            warnings.warn(
                "You have selected >=35 spherical harmonic degrees. Starry does not behave nicely at this high a resolution!")

        # if fit_spherical_harmonic_coeffs == False:
        #     if n_spherical_harmonics == 1 and spherical_harmonics_coeffs == [1.0, 0.0, 0.5, 0.0]:
        #         warnings.warn("You are running the PhaseCurveModel with default planet map = a simple dipole model.")
        #     else:
        #         n_coeffs = 0
        #         for degree in range(n_spherical_harmonics + 1):
        #             n_coeffs = n_coeffs + (2 * degree) + 1
        #         if len(spherical_harmonics_coeffs) != n_coeffs:
        #             warnings.warn(f"""Number of spherical harmonics does not match number of coeffs.\nIf deg=0, n_coeffs=1 (Y_0,0), if deg=1, n_coeffs=4 (Y_0,0, Y_1,-1, Y_1,0, Y_1,1) etc.
        #                     \nYou've passed {n_spherical_harmonics} degrees therefore you should have an array for `spherical_harmonics_coeffs` of length={n_coeffs}
        #                  """)
        #             return

        self.n_spherical_harmonics = n_spherical_harmonics
        # self.spherical_harmonics_coeffs = spherical_harmonics_coeffs
        # self.fit_spherical_harmonic_coeffs = fit_spherical_harmonic_coeffs

        # if spherical_harmonics_coeffs[0] != 1.0:
        #     warnings.warn(
        #         "The Y{0,0} coefficient has been changed. Starry does not let you change this coeff directly - you need to change the planet map's amplitude")

        self.source_npts = source_npts

        super().__init__(**kw)

        self.set_defaults()
        self.set_name(name)
        self.metadata = {}
        self.model = self.phase_curve_model

        if type_of_model in allowed_types_of_models:
            self.type_of_model = type_of_model

    def __repr__(self):
        """
        Print the transit model.
        """
        return f"<chromatic phase curve model '{self.name}' 🌈>"

    def what_are_parameters(self):
        """
        Print a summary of what each parameter is
        # """
        self.parameter_descriptions = dict(
            stellar_radius="The stellar radius [R_sun].",
            stellar_mass="The stellar mass [M_sun].",
            stellar_amplitude="The stellar amplitude, or the out-of-transit baseline.",
            stellar_prot="The rotational period of the star [d].",
            stellar_inc="Stellar inclination, or vertical angle between spin axis and line-of-sight [degrees]",
            stellar_obl="Stellar obliquity, or horizontal tilt of the spin axis from line-of-sight [degrees]",
            period="Orbital period [d].",
            t0="Epoch of transit center [d].",
            planet_log_amplitude="The log-amplitude of the planet (determines the depth of the eclipse). Equivalent to log(albedo)",
            phase_offset="The phase offset for maximum reflection, this can be caused by hot spots on the planet [degrees].",
            # roughness="How 'rough' the surface of a planet is according to Oren and Nayar (1994). Default=0 for fully Lambertian scattering. [degrees]",
            inclination="The inclination of the planet [degrees].",
            planet_mass="The mass of the planet [M_jupiter].",
            planet_radius="The radius of the planet [R_jupiter].",
            ecs="[ecosw, esinw], where e is eccentricity.",
            limb_darkening="2-d Quadratic limb-darkening coefficients.",
            planet_surface_map="",
        )

        for k, v in self.parameter_descriptions.items():
            print(f"{k}: {v}")

    def set_defaults(self):
        """
        Set the default parameters for the model.
        """
        self.defaults = dict(
            stellar_radius=1,
            stellar_mass=1,
            stellar_amplitude=1.0,
            stellar_prot=1.0,
            stellar_inc=90.0,
            stellar_obl=0.0,
            period=1.0,
            t0=1.0,
            planet_log_amplitude=-2.8,
            # roughness=0,
            phase_offset=0.0,
            inclination=90.0,
            planet_mass=0.01,
            planet_radius=0.01,
            ecs=np.array([0.0, 0.0]),
            limb_darkening=np.array([0.4, 0.2]),
            # planet_surface_map=np.array([0.0, 0.5, 0.0])
            # map1_1=0.0,
            # map10=0.5,
            # map11=0.0
        )

        if self.n_spherical_harmonics > 0:
            ncoeff = starry.Map(ydeg=self.n_spherical_harmonics).Ny - 1
            planet_surface_map = np.zeros(ncoeff)
            planet_surface_map[1] = 0.5
        else:
            planet_surface_map = []
        self.defaults['planet_surface_map'] = planet_surface_map

        # for degree in range(1, self.n_spherical_harmonics + 1):
        #     for harmonic_order in range(-degree, degree + 1):
        #         self.defaults[f'map{degree}{str(harmonic_order).replace("-","_")}'] = 0.0

    def check_limb_darkened_map_is_physical(self, map):
        """
        Parameters
        -------
        map: Starry spherical harmonic limb-darkened map.

        Returns
        -------
        True if the map is non-negative everywhere and the intensity is monotonically decreasing towards the limb

        """
        return map.limbdark_is_physical()

    def setup_lightcurves(self, store_models: bool = False, **kwargs):
        """
        Create an `starry` light curve model, given the stored parameters.
        [This should be run after .attach_data()]

        Parameters
        ----------
        store_models: boolean for whether to store the eclipse model during fitting (for faster
        plotting/access later), default=False
        """
        if starry.config.lazy == False:
            print(
                "Starry will not play nice with pymc3 in greedy mode.\n Please set starry.config.lazy=True and try again!"
            )
            return

        # ensure that attach data has been run before setup_lightcurves
        if not hasattr(self, "data"):
            print("You need to attach some data to this chromatic model!")
            return

        # if the model has a name then add this to each parameter's name
        if hasattr(self, "name"):
            name = self.name + "_"
        else:
            name = ""
            print("No name set for the model.")

        datas, models = self.choose_model_based_on_optimization_method()

        kw = {"shape": datas[0].nwave}

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
        for k in self.parameters.keys():
            parameters_to_loop_over[k] = []

        for j, (mod, data) in enumerate(zip(models, datas)):
            if self.optimization == "separate":
                kw["i"] = j

            with mod:
                for pname in parameters_to_loop_over.keys():
                    parameters_to_loop_over[pname].append(
                        self.parameters[pname].get_prior_vector(**kw)
                    )

                y_model = []
                initial_guess = []

                for i, w in enumerate(data.wavelength):
                    param_i = {}
                    for param_name, param in parameters_to_loop_over.items():
                        if isinstance(self.parameters[param_name], WavelikeFitted):
                            param_i[param_name] = param[j][i]
                        elif isinstance(self.parameters[param_name], Fitted) and eval_in_model(np.shape(param[j]))[
                            0] == 1:
                            param_i[param_name] = param[j][0]
                        else:
                            param_i[param_name] = param[j]

                    # **FUNCTION TO MODEL - MAKE SURE IT MATCHES self.temp_model()!**
                    # extract light curve from Starry model at given times
                    star = starry.Primary(
                        starry.Map(
                            ydeg=0,
                            udeg=2,
                            amp=param_i[f"{name}stellar_amplitude"],
                            inc=param_i[f"{name}stellar_inc"],
                            obl=param_i[f"{name}stellar_obl"],
                        ),
                        inc=param_i[f"{name}stellar_inc"],
                        m=param_i[f"{name}stellar_mass"],
                        r=param_i[f"{name}stellar_radius"],
                        prot=param_i[f"{name}stellar_prot"],
                    )
                    star.map[1] = param_i[f"{name}limb_darkening"][0]
                    star.map[2] = param_i[f"{name}limb_darkening"][1]
                    if self.check_limb_darkened_map_is_physical(star.map) == False:
                        print("The limb-darkening is unphysical! The map is either negative or not monotonically " \
                              "decreasing towards the limb!")

                    omega = (theano.tensor.arctan2(param_i[f"{name}ecs"][1], param_i[f"{name}ecs"][0]) * 180.0) / np.pi
                    eccentricity = pm.math.sqrt(param_i[f"{name}ecs"][0] ** 2 + param_i[f"{name}ecs"][1] ** 2)

                    planet = starry.kepler.Secondary(
                        starry.Map(
                            ydeg=self.n_spherical_harmonics,
                            udeg=0,
                            amp=10 ** param_i[f"{name}planet_log_amplitude"],
                            inc=param_i[f"{name}inclination"],
                            obl=0.0,
                        ),
                        # the surface map
                        inc=param_i[f"{name}inclination"],
                        m=param_i[f"{name}planet_mass"],  # mass in Jupiter masses
                        r=param_i[f"{name}planet_radius"],  # radius in Jupiter radii
                        porb=param_i[f"{name}period"],  # orbital period in days
                        prot=param_i[f"{name}period"],  # orbital period in days
                        ecc=eccentricity,  # eccentricity
                        w=omega,  # longitude of pericenter in degrees
                        t0=param_i[f"{name}t0"],  # time of transit in days
                        length_unit=u.R_jup,
                        mass_unit=u.M_jup,
                    )
                    # count = 0
                    if self.n_spherical_harmonics > 0:
                        planet.map[1:, :] = param_i[f'{name}planet_surface_map']
                        # for degree in range(1, self.n_spherical_harmonics + 1):
                        #     for harmonic_order in range(-degree, degree + 1):
                        #         planet.map[degree, harmonic_order] = param_i[f'{name}map{degree}{str(harmonic_order).replace("-", "_")}']

                    planet.theta0 = 180.0 + param_i[f"{name}phase_offset"]

                    print("spherical harmonic coeffs:", eval_in_model(planet.map.y))
                    # planet.roughness = param_i[f"{name}roughness"]

                    # save the radius ratio for generating the transmission spectrum later:
                    rr = Deterministic(f"{name}radius_ratio[{i + j}]",
                                       (param_i[f"{name}planet_radius"] * (1 * u.R_jup).to_value("R_sun")) / param_i[
                                           f"{name}stellar_radius"])

                    system = starry.System(star, planet)
                    flux_model = system.flux(data.time.to_value("day"))
                    y_model.append(flux_model)

                    initial_guess.append(eval_in_model(flux_model))

                # (if we've chosen to) add a Deterministic parameter to the model for easy extraction/plotting
                # later:
                if self.store_models:
                    Deterministic(f"{name}model", pm.math.stack(y_model, axis=0))

                # add the model to the overall lightcurve:
                if f"wavelength_{j}" not in self.every_light_curve.keys():
                    self.every_light_curve[f"wavelength_{j}"] = pm.math.stack(
                        y_model, axis=0
                    )
                else:
                    self.every_light_curve[f"wavelength_{j}"] += pm.math.stack(
                        y_model, axis=0
                    )

                # add the initial guess to the model:
                if f"wavelength_{j}" not in self.initial_guess.keys():
                    self.initial_guess[f"wavelength_{j}"] = np.array(initial_guess)
                else:
                    self.initial_guess[f"wavelength_{j}"] += initial_guess

                self.model_chromatic_eclipse_flux = [
                    self.every_light_curve[k] for k in tqdm(self.every_light_curve)
                ]

    def sample(
            self,
            **kw,
    ):
        if "sampling_method" not in kw.keys():
            print(
                "Starry doesn't support pymc3.sample(), using pymc3_ext.sample() instead."
            )
            LightcurveModel.sample(self, sampling_method=sample_ext, **kw)
        else:
            warnings.warn("WARNING: Starry doesn't seem to support pymc3.sample()")
            LightcurveModel.sample(self, **kw)

    def add_model_to_rainbow(self):
        """
        Add the phase curve model to the Rainbow object.
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

    def phase_curve_model(
            self, params: dict, i: int = 0, time: list = None
    ) -> np.array:
        """
        Create a phase curve model given the passed parameters.

        Parameters
        ----------
        params: A dictionary of parameters to be used in the eclipse model.
        i: wavelength index
        time: If we don't want to use the default time then the user can pass a time array on which to calculate the model

        Returns
        -------
        object
        """

        # if the model has a name then add this to each parameter's name
        if hasattr(self, "name"):
            name = self.name + "_"
        else:
            name = ""

        if time is None:
            if self.optimization == "separate":
                data = self.get_data(i)
            else:
                data = self.get_data()
            time = list(data.time.to_value("day"))

        self.check_and_fill_missing_parameters(params, i)

        star = starry.Primary(
            starry.Map(
                ydeg=0,
                udeg=2,
                amp=params[f"{name}stellar_amplitude"],
                inc=params[f"{name}stellar_inc"],
                obl=params[f"{name}stellar_obl"],
            ),
            m=params[f"{name}stellar_mass"],
            r=params[f"{name}stellar_radius"],
            prot=params[f"{name}stellar_prot"],
        )

        star.map[1] = params[f"{name}limb_darkening"][0]
        star.map[2] = params[f"{name}limb_darkening"][1]

        omega = theano.tensor.arctan2(params[f"{name}ecs"][1], params[f"{name}ecs"][0]) * 180.0 / np.pi
        eccentricity = pm.math.sqrt(params[f"{name}ecs"][0] ** 2 + params[f"{name}ecs"][1] ** 2)

        planet = starry.kepler.Secondary(
            starry.Map(
                ydeg=self.n_spherical_harmonics,
                udeg=0,
                amp=10 ** params[f"{name}planet_log_amplitude"],
                # inc=90,
                inc=params[f"{name}inclination"],
                obl=0.0,
            ),
            # the surface map
            inc=params[f"{name}inclination"],
            m=params[f"{name}planet_mass"],  # mass in Jupiter masses
            r=params[f"{name}planet_radius"],  # radius in Jupiter radii
            porb=params[f"{name}period"],  # orbital period in days
            prot=params[f"{name}period"],  # orbital period in days
            ecc=eccentricity,  # eccentricity
            w=omega,  # longitude of pericenter in degrees
            t0=params[f"{name}t0"],  # time of transit in days
            length_unit=u.R_jup,
            mass_unit=u.M_jup,
        )

        # count = 0

        if self.n_spherical_harmonics > 0:
            planet.map[1:, :] = params[f'{name}planet_surface_map']
            # for degree in range(1, self.n_spherical_harmonics + 1):
            #     for harmonic_order in range(-degree, degree + 1):
            #         planet.map[degree, harmonic_order] = params[f'{name}map{degree}{str(harmonic_order).replace("-", "_")}']

        planet.theta0 = 180.0 + params[f"{name}phase_offset"]
        system = starry.System(star, planet)
        flux_model = system.flux(time).eval()

        if hasattr(self, 'keplerian_system'):
            self.keplerian_system[f'w{i}'] = system
            self.sec[f'w{i}'] = planet
            self.pri[f'w{i}'] = star
            self.planet_map[f'w{i}'] = planet.map
            self.star_map[f'w{i}'] = star.map
        else:
            self.keplerian_system = {f'w{i}': system}
            self.sec = {f'w{i}': planet}
            self.pri = {f'w{i}': star}
            self.planet_map = {f'w{i}': planet.map}
            self.star_map = {f'w{i}': star.map}

        return flux_model

    def make_emission_spectrum_table(
            self, uncertainty=["hdi_16%", "hdi_84%"], svname=None
    ):
        """
        Generate and return a emission spectrum table
        """
        return EclipseModel.make_emission_spectrum_table(self, uncertainty=uncertainty, svname=svname)

    def make_transmission_spectrum_table(
            self, uncertainty=["hdi_16%", "hdi_84%"], svname=None
    ):
        """
        Generate and return a transmission spectrum table
        """

        # THIS IS 100% A HACK TO INCLUDE RADIUS_RATIO IN THE RESULTS TABLE:
        with self._pymc3_model:
            try:
                self.parameters[f'{self.name}_radius_ratio'] = Normal('radius_ratio', mu=0.1, sigma=0.1)
            except:
                pass

        return TransitModel.make_transmission_spectrum_table(self, uncertainty=uncertainty, svname=svname)

    def plot_eclipse_spectrum(
            self, table=None, uncertainty=["hdi_16%", "hdi_84%"], ax=None, plotkw={}, **kw
    ):
        """
        Plot the eclipse spectrum (specifically the planet amplitude as a function of wavelength).

        Parameters
        ----------
        table: [Optional] Table to use as eclipse spectrum (otherwise the default is to use the MCMC sampling results.
        The table must have the following columns: "{self.name}_depth", "{self.name}_depth_neg_error",
        "{self.name}_depth_pos_error", "wavelength".
        uncertainty: [Optional] List of the names of parameters to use as the lower and upper errors. Options: "hdi_16%", "hdi_84%",
        "sd" etc. Default = ["hdi_16%", "hdi_84%"].
        ax: [Optional] Pass a preexisting matplotlib axis is to be used instead of creating a new one.Default = None.
        plotkw: [Optional] Dict of kw to pass to the transmission specrum plotting function.
        kw: [Optional] kw to pass to the TransitModel.plot_transmission_spectrum.

        Returns
        -------

        """
        EclipseModel.plot_eclipse_spectrum(self, table=table, uncertainty=uncertainty, ax=ax, plotkw=plotkw, **kw)

    def plot_transmission_spectrum(
            self, table=None, uncertainty=["hdi_16%", "hdi_84%"], ax=None, plotkw={}, **kw
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
        kw: [Optional] kw to pass to the TransitModel.plot_transmission_spectrum.

        Returns
        -------

        """
        TransitModel.plot_transmission_spectrum(self, table=table, uncertainty=uncertainty, ax=ax, plotkw=plotkw, **kw)

    def show_system(self, i=0, **kw):
        """
        Displays the recovered orbital system + surface maps for the planet and star for the time array passed to this
        function.

        Parameters
        ----------
        kw

        Returns
        -------

        """
        # if self.method == "starry":
        if hasattr(self, 'keplerian_system'):
            self.keplerian_system[f'w{i}'].show(**kw)

    def show_planet_map(self, secondary_i=0, i=0, **kw):
        """
        Displays the recovered planetary surface map. If {theta} is passed to this function it will show a rotating body.
        A projection of the surface map can be displayed by passing the {projection} kw.

        Parameters
        ----------
        secondary_i: [not used at the moment]
        kw: keywords to pass to Starry.map.show()

        Returns
        -------

        """
        if hasattr(self, 'planet_map'):
            lat, lon, value = eval_in_model(self.planet_map[f'w{i}'].minimize(), model=self._pymc3_model)
            self.planet_map[f'w{i}'].show(**kw)

            if value < 0:
                print(f"WARNING: This Starry map goes negative at lat={lat:.3f}, lon={lon:.3f}! This is unphysical!" \
                      " We should ideally implement Pixel Sampling, however, this is not in place yet!")
            # self.keplerian_system.secondaries[secondary_i].show(**kw)

    def show_star_map(self, i=0, **kw):
        """
        Displays the recovered stellar surface map.

        Parameters
        ----------
        kw

        Returns
        -------

        """
        if hasattr(self, 'star_map'):
            self.star_map[f'w{i}'].show(**kw)
