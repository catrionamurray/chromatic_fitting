from ..imports import *

from .lightcurve import *
import warnings

"""
Example of setting up a TransitModel:

def create_new_transit_model():
    # create transit model:
    t = TransitModel()
    
    # add empty pymc3 model:
    t.initialize_empty_model()
    
    # add our parameters:
    t.setup_parameters(
                      period=1, # a fixed value!
                       epoch=Fitted(Uniform,lower=-0.05,upper=0.05), # one fitted value across all wavelengths
                       stellar_radius = Fitted(Uniform, lower=0.8, upper=1.2,testval=1),
                       stellar_mass =Fitted(Uniform, lower=0.8, upper=1.2,testval=1),
                       radius_ratio=WavelikeFitted(Normal, mu=0.5, sigma=0.05), # a value fitted for every wavelength!
                       impact_parameter=Fitted(ImpactParameter,ror=0.15,testval=0.44),
                       limb_darkening=WavelikeFitted(QuadLimbDark,testval=[0.05,0.35]),
                        baseline = WavelikeFitted(Normal, mu=1.0, sigma=0.05), # I will keep this fixed for when we add a polynomial model!
                    )
    
    # attach a Rainbow object, r, to the model:
    t.attach_data(r)
    
    # setup the lightcurves for the transit model:
    t.setup_lightcurves()
    
    # relate the "actual" data to the model (using a Normal likelihood function)
    t.setup_likelihood()    
    
    # MCMC (NUTS) sample the parameters:
    t.sample(tune=2000, draws=2000, chains=4, cores=4)
    
    # summarize the results:
    t.summarize(round_to=7, fmt='wide')
    
    return t
                
"""


class TransitModel(LightcurveModel):
    """
    A transit model for the lightcurve.
    """

    def __init__(
        self, name: str = "transit", type_of_model: str = "planet", **kw: object
    ) -> None:
        """
        Initialise the transit model.

        Parameters
        ----------
        name: the name of the model ('transit' by default)
        kw: any keywords to pass to the lightcurve model
        """

        # require the following transit parameters to initialize the model:
        self.required_parameters = [
            "stellar_radius",
            "stellar_mass",
            "radius_ratio",
            "period",
            "epoch",
            "baseline",
            "impact_parameter",
            "limb_darkening",
        ]

        super().__init__(**kw)
        # self.xlims = xlims
        self.set_defaults()
        self.set_name(name)
        self.model = self.transit_model

        if type_of_model in allowed_types_of_models:
            self.type_of_model = type_of_model
        else:
            warnings.warn(
                f"{type_of_model} is not a valid type of model. Please select one of: {allowed_types_of_models}"
            )

    def __repr__(self):
        """
        Print the transit model.
        """
        return f"<chromatic transit model '{self.name}' 🌈>"

    def what_are_parameters(self):
        """
        Print a summary of what each parameter is
        # """
        self.parameter_descriptions = dict(
            stellar_radius="The stellar radius [R_sun].",
            stellar_mass="The stellar mass [M_sun].",
            radius_ratio="The radius ratio between planet and star.",
            period="Orbital period [d].",
            epoch="Epoch of transit center [d].",
            baseline="Out-of-transit baseline",
            impact_parameter="The impact parameter of the planet, b.",
            eccentricity="The eccentricity of the planet (THIS IS INACTIVE AS OF V0.10.1.2)",
            omega="The argument of periapse [degrees].",
            limb_darkening="Quadratic limb-darkening coefficients.",
        )

        for k, v in self.parameter_descriptions.items():
            print(f"{k}: {v}")

    def set_defaults(self):
        """
        Set the default parameters for the model.
        """
        self.defaults = dict(
            stellar_radius=1.0,
            stellar_mass=1.0,
            radius_ratio=1.0,
            period=1.0,
            epoch=0.0,
            baseline=1.0,
            impact_parameter=0.5,
            eccentricity=0.0,
            omega=0,  # np.pi / 2.0,
            limb_darkening=[0.2, 0.2],
        )

    def setup_orbit(self):
        """
        Create an `exoplanet` orbit model, given the stored parameters.
        [If run manually it should be run after .setup_parameters()]

        """

        # if the optimization method is separate wavelengths then set up for looping
        if self.optimization == "separate":
            models = self._pymc3_model
            datas = [self.get_data(i) for i in range(self.data.nwave)]
        else:
            models = [self._pymc3_model]
            datas = [self.get_data()]

        kw = {"shape": datas[0].nwave}

        # if the model has a name then add this to each parameter"s name
        if hasattr(self, "name"):
            name = self.name + "_"
        else:
            name = ""
            print("No name set for the model.")

        self.orbit = []
        # we need to separate the orbits in wavelength otherwise they're treated as
        # multiple planets.py in the same system
        for j, (mod, data) in enumerate(zip(models, datas)):
            if self.optimization == "separate":
                kw["i"] = j
            with mod:
                # Set up a Keplerian orbit for the planets.py (for now without eccentricity)
                orbit = xo.orbits.KeplerianOrbit(
                    period=self.parameters[name + "period"].get_prior_vector(**kw),
                    t0=self.parameters[name + "epoch"].get_prior_vector(**kw),
                    b=self.parameters[name + "impact_parameter"].get_prior_vector(**kw),
                    r_star=self.parameters[name + "stellar_radius"].get_prior_vector(
                        **kw
                    ),
                    m_star=self.parameters[name + "stellar_mass"].get_prior_vector(
                        **kw
                    ),
                    # ecc=self.parameters[name + "eccentricity"].get_prior(j),
                    # omega=self.parameters[name + "omega"].get_prior(j),
                )

                if self.optimization == "separate":
                    # store a separate orbit for each model if we are optimizing wavelengths separately
                    self.orbit.append(orbit)
                else:
                    self.orbit = orbit

    def setup_lightcurves(self, store_models: bool = False, **kwargs):
        """
        Create an `exoplanet` light curve model, given the stored parameters.
        [This should be run after .attach_data()]

        Parameters
        ----------
        store_models: boolean for whether to store the transit model during fitting (for faster
        plotting/access later), default=False
        """

        # ensure that attach data has been run before setup_lightcurves
        if not hasattr(self, "data"):
            print("You need to attach some data to this chromatic model!")
            return

        # ensure that setup_orbit has been run before setup_lightcurves
        if not hasattr(self, "orbit"):
            self.setup_orbit()

        # if the model has a name then add this to each parameter's name
        if hasattr(self, "name"):
            name = self.name + "_"
        else:
            name = ""
            print("No name set for the model.")

        # set up the models, data and orbits in a format for looping
        datas, models, orbits = self.choose_model_based_on_optimization_method(
            self.orbit
        )

        # kw = {"shape": self.get_data().nwave}
        kw = {"shape": datas[0].nwave}

        # if the .every_light_curve attribute (final lc model) is not already present then create it now
        if not hasattr(self, "every_light_curve"):
            self.every_light_curve = {}
        # if the .every_light_curve attribute (final lc model) is not already present then create it now
        if not hasattr(self, "initial_guess"):
            self.initial_guess = {}

        # we can decide to store the LC models during the fit (useful for plotting later, however, uses large amounts
        # of RAM)
        if store_models == True:
            self.store_models = store_models

        # stored_a_r = False

        # limb_darkening, planet_radius, baseline = [], [], []
        parameters_to_loop_over = {
            f"{name}limb_darkening": [],
            f"{name}radius_ratio": [],
            f"{name}stellar_radius": [],
            f"{name}baseline": [],
        }

        for j, (mod, data, orbit) in enumerate(zip(models, datas, orbits)):
            if self.optimization == "separate":
                kw["i"] = j

            with mod:
                for pname in parameters_to_loop_over.keys():
                    if pname != f"{name}planet_radius":
                        parameters_to_loop_over[pname] = self.parameters[
                            pname
                        ].get_prior_vector(**kw)
                    if pname == f"{name}limb_darkening":
                        if isinstance(self.parameters[pname], Fitted):
                            if self.parameters[pname].distribution == QuadLimbDark:
                                # print(eval_in_model(parameters_to_loop_over[pname]))
                                parameters_to_loop_over[pname] = [parameters_to_loop_over[pname]]

                # store Deterministic parameter {a/R*} for use later
                Deterministic(
                    f"{name}a_R*",
                    orbit.a / parameters_to_loop_over[name + "stellar_radius"],
                )
                # stored_a_r = True

                planet_radius = Deterministic(
                    f"{name}planet_radius",
                    parameters_to_loop_over[name + "radius_ratio"]
                    * parameters_to_loop_over[name + "stellar_radius"],
                )
                parameters_to_loop_over[f"{name}planet_radius"] = planet_radius

                light_curves, initial_guess = [], []
                for i, w in enumerate(data.wavelength):
                    param_i = {}
                    for pname, param in parameters_to_loop_over.items():
                        if pname == f"{name}planet_radius":
                            try:
                                param_i[pname] = param[i]
                            except:
                                param_i[pname] = param
                        else:
                            if isinstance(self.parameters[pname], WavelikeFitted):
                                param_i[pname] = param[i]
                            else:
                                param_i[pname] = param

                    # extract light curve from Exoplanet model at given times
                    light_curves.append(
                        xo.LimbDarkLightCurve(
                            param_i[f"{name}limb_darkening"]
                        ).get_light_curve(
                            orbit=orbit,
                            r=param_i[f"{name}planet_radius"],
                            t=list(data.time.to_value("day")),
                        )
                        + param_i[f"{name}baseline"]
                    )

                    initial_guess.append(
                        eval_in_model(pm.math.sum(light_curves, axis=-1)[-1])
                    )

                # (if we've chosen to) add a Deterministic parameter to the model for easy extraction/plotting
                # later:
                if self.store_models:
                    Deterministic(f"{name}model", pm.math.sum(light_curves, axis=-1))

                # add the transit to the final light curve
                if f"wavelength_{j}" not in self.every_light_curve.keys():
                    self.every_light_curve[f"wavelength_{j}"] = pm.math.sum(
                        light_curves, axis=-1
                    )  # [i]
                else:
                    self.every_light_curve[f"wavelength_{j}"] += pm.math.sum(
                        light_curves, axis=-1
                    )  # [i]

                # self.model_chromatic_transit_flux = [
                #     self.every_light_curve[k] for k in tqdm(self.every_light_curve)
                # ]

                # add the initial guess to the final light curve
                if f"wavelength_{j}" not in self.initial_guess.keys():
                    self.initial_guess[f"wavelength_{j}"] = np.array(initial_guess)
                else:
                    self.initial_guess[f"wavelength_{j}"] += initial_guess

    def transit_model(
        self, params: dict, i: int = 0, time: list = None
    ) -> np.array:
        """
        Create a transit model given the passed parameters.

        Parameters
        ----------
        params: A dictionary of parameters to be used in the transit model.
        i: wavelength index
        time: If we don't want to use the default time then the user can pass a time array on which to calculate the model

        Returns
        -------
        object
        """

        if self.optimization == "separate":
            data = self.get_data(i)
        else:
            data = self.get_data()

        # if the model has a name then add this to each parameter"s name
        if hasattr(self, "name"):
            name = self.name + "_"
        else:
            name = ""

        if time is None:
            time = list(data.time.to_value("day"))

        self.check_and_fill_missing_parameters(params, i)

        orbit = xo.orbits.KeplerianOrbit(
            period=params[f"{name}period"],
            t0=params[f"{name}epoch"],
            b=params[f"{name}impact_parameter"],
            # ecc=params["eccentricity"],
            # omega=params["omega"],
            r_star=params[f"{name}stellar_radius"],
            m_star=params[f"{name}stellar_mass"],
        )

        ldlc = (
            xo.LimbDarkLightCurve(params[f"{name}limb_darkening"])
            .get_light_curve(
                orbit=orbit,
                r=params[f"{name}radius_ratio"]
                * params[f"{name}stellar_radius"],
                t=time,
            )
            .eval()
        )

        return ldlc.transpose()[0] + params[f"{name}baseline"]

    def plot_orbit(self, timedata: object = None, filename: str = None):
        """
        Plot the orbit model (to check that the planet is transiting at the times we've set).

        Parameters
        ----------
        timedata: An array of times to plot the orbit at, only used if .attach_data() hasn't been
        run yet (default=None)
        svname: Name if we want to save the plot (default=None)

        """

        # If the data hasn't been attached yet, then use the timedata passed to the function
        if not hasattr(self, "data"):
            if timedata is None:
                warnings.warn(
                    "No data attached to this object and no time data provided. Plotting orbit will not work."
                )
                print(
                    "No data attached to this object and no time data provided. Plotting orbit will not work."
                )
                return
        else:
            timedata = self.data.time

        # if the optimization method is separate wavelengths then set up for looping
        if self.optimization == "separate":
            models = self._pymc3_model
            orbits = self.orbit
        else:
            models = [self._pymc3_model]
            orbits = [self.orbit]

        # plot the orbit
        for j, (mod, orbit) in enumerate(zip(models, orbits)):
            with mod:
                # find the x,y,z position of the planet at each timestamp relative to the centre of the star
                x, y, z = [
                    eval_in_model(bla, point=mod.test_point)
                    for bla in orbit.get_planet_position(timedata)
                ]
                plt.figure(figsize=(10, 3))
                theta = np.linspace(0, 2 * np.pi)
                plt.fill_between(np.cos(theta), np.sin(theta), color="gray")
                plt.scatter(x, y, c=timedata)
                plt.axis("scaled")
                plt.ylim(-1, 1)
                if filename is None:
                    plt.show()
                else:
                    plt.savefig(filename)
                plt.close()

    def make_transmission_spectrum_table(
        self, **kw
    ):
        """
        Generate and return a transmission spectrum table

        Parameters
        ----------
        uncertainty: [Optional] List of the names of parameters to use as the lower and upper errors. Options: "hdi_16%",
        "hdi_84%", "sd" etc. Default = ["hdi_16%", "hdi_84%"].
        svname: [Optional] String csv filename to save the transmission table. Default = None (do not save)

        Returns
        -------

        """
        return self.make_spectrum_table(param="radius_ratio", **kw)

    def plot_transmission_spectrum(
        self, **kw
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
        # if table is not None:
        #     transmission_spectrum = table
        #     try:
        #         # ensure the correct columns exist in the transmission spectrum table
        #         assert transmission_spectrum[f"{self.name}_radius_ratio"]
        #         assert transmission_spectrum[f"{self.name}_radius_ratio_neg_error"]
        #         assert transmission_spectrum[f"{self.name}_radius_ratio_pos_error"]
        #         assert transmission_spectrum["wavelength"]
        #     except:
        #         print(
        #             f"The given table doesn't have the correct columns 'wavelength', '{self.name}_radius_ratio', "
        #             f"{self.name}_radius_ratio_pos_error' and '{self.name}_radius_ratio_neg_error'"
        #         )
        # else:
        #     kw["uncertainty"] = uncertainty
        #     transmission_spectrum = self.make_transmission_spectrum_table(**kw)
        #     transmission_spectrum["wavelength"] = [
        #         t.to_value("micron") for t in transmission_spectrum["wavelength"].values
        #     ]
        #
        # if ax is None:
        #     fig, ax = plt.subplots(figsize=(10, 4))
        #
        # plt.sca(ax)
        # plt.title("Transmission Spectrum")
        # plt.plot(
        #     transmission_spectrum["wavelength"],
        #     transmission_spectrum[f"{self.name}_radius_ratio"],
        #     "kx",
        #     **plotkw,
        # )
        # plt.errorbar(
        #     transmission_spectrum["wavelength"],
        #     transmission_spectrum[f"{self.name}_radius_ratio"],
        #     yerr=[
        #         transmission_spectrum[f"{self.name}_radius_ratio_neg_error"],
        #         transmission_spectrum[f"{self.name}_radius_ratio_pos_error"],
        #     ],
        #     color="k",
        #     capsize=2,
        #     linestyle="None",
        #     **plotkw,
        # )
        # plt.xlabel("Wavelength (microns)")
        # plt.ylabel("Radius Ratio")
        self.plot_spectrum(param="radius_ratio", name_of_spectrum="Transmission", **kw)

    def add_model_to_rainbow(self):
        """
        Add the transit model to the Rainbow object.
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
