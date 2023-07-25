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
                 ydeg: int = 15,
                 source_npts: int = 1,
                 type_of_model: str = "planet",
                 **kw: object) -> None:
        """
        Initialise the phase curve model.

        Parameters
        ----------
        name: the name of the model ('phasecurve' by default)
        ydeg: Number of spherical harmonic orders to use to model the star (higher value = resolving smaller features)
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
        ]

        if ydeg >= 25:
            warnings.warn(
                "You have selected >=25 spherical harmonic degrees. Starry will be very slow!")
        elif ydeg >= 35:
            warnings.warn(
                "You have selected >=35 spherical harmonic degrees. Starry does not behave nicely at this high a resolution!")

        self.ydeg = ydeg
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
            phase_offset="",
            # roughness="How 'rough' the surface of a planet is according to Oren and Nayar (1994). Default=0 for fully Lambertian scattering. [degrees]",
            inclination="The inclination of the planet [degrees].",
            planet_mass="The mass of the planet [M_jupiter].",
            planet_radius="The radius of the planet [R_jupiter].",
            ecs="[ecosw, esinw], where e is eccentricity.",
            limb_darkening="2-d Quadratic limb-darkening coefficients.",
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
            phase_offset = 0.0,
            inclination=90.0,
            planet_mass=0.01,
            planet_radius=0.01,
            ecs=np.array([0.0, 0.0]),
            limb_darkening=np.array([0.4, 0.2]),
        )

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

        parameters_to_loop_over = {
            f"{name}stellar_amplitude": [],
            f"{name}stellar_radius": [],
            f"{name}stellar_mass": [],
            f"{name}stellar_prot": [],
            f"{name}stellar_inc": [],
            f"{name}stellar_obl": [],
            f"{name}period": [],
            f"{name}t0": [],
            f"{name}planet_log_amplitude": [],
            f"{name}phase_offset": [],
            f"{name}inclination": [],
            f"{name}planet_mass": [],
            f"{name}planet_radius": [],
            f"{name}ecs": [],
            f"{name}limb_darkening": [],
        }

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
                        elif isinstance(self.parameters[param_name], Fitted) and eval_in_model(np.shape(param[j]))[0] == 1:
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
                        m=param_i[f"{name}stellar_mass"],
                        r=param_i[f"{name}stellar_radius"],
                        prot=param_i[f"{name}stellar_prot"],
                    )
                    star.map[1] = param_i[f"{name}limb_darkening"][0]
                    star.map[2] = param_i[f"{name}limb_darkening"][1]
                    omega = (theano.tensor.arctan2(param_i[f"{name}ecs"][1], param_i[f"{name}ecs"][0]) * 180.0) / np.pi
                    eccentricity = pm.math.sqrt(param_i[f"{name}ecs"][0] ** 2 + param_i[f"{name}ecs"][1] ** 2)

                    planet = starry.kepler.Secondary(
                        starry.Map(
                            ydeg=self.ydeg,
                            udeg=0,
                            amp=10 ** param_i[f"{name}planet_log_amplitude"],
                            inc=90.0,
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
                    planet.map[1, 0] = 0.5
                    planet.theta0 = 180.0 + param_i[f"{name}phase_offset"]
                    # planet.roughness = param_i[f"{name}roughness"]

                    rr = Deterministic(f"{name}radius_ratio[{i+j}]", param_i[f"{name}planet_radius"]/param_i[f"{name}stellar_radius"])

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
                ydeg=1,
                udeg=0,
                amp=10 ** params[f"{name}planet_log_amplitude"],
                inc=90,
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

        # planet.theta0 = 180.0
        planet.map[1, 0] = 0.5
        planet.theta0 = 180.0 + params[f"{name}phase_offset"]
        system = starry.System(star, planet)
        flux_model = system.flux(time).eval()

        return flux_model

    def make_emission_spectrum_table(
            self, uncertainty=["hdi_16%", "hdi_84%"], svname=None
    ):
        """
        Generate and return a emission spectrum table
        """
        results = self.get_results(uncertainty=uncertainty)[
            [
                "wavelength",
                f"{self.name}_planet_log_amplitude",
                f"{self.name}_planet_log_amplitude_{uncertainty[0]}",
                f"{self.name}_planet_log_amplitude_{uncertainty[1]}",
            ]
        ]
        emiss_table = results[["wavelength"]]
        emiss_table[f"{self.name}_depth"] = 10 ** results[f"{self.name}_planet_log_amplitude"]
        if "hdi" in uncertainty[0]:
            emiss_table[f"{self.name}_depth_neg_error"] = (
                    10 ** results[f"{self.name}_planet_log_amplitude"]
                    - 10 ** results[f"{self.name}_planet_log_amplitude_{uncertainty[0]}"]
            )
            emiss_table[f"{self.name}_depth_pos_error"] = (
                    10 ** results[f"{self.name}_planet_log_amplitude_{uncertainty[1]}"]
                    - 10 ** results[f"{self.name}_planet_log_amplitude"]
            )
        else:
            emiss_table[f"{self.name}_depth_neg_error"] = 10 ** results[
                f"{self.name}_planet_log_amplitude_{uncertainty[0]}"
            ]
            emiss_table[f"{self.name}_depth_pos_error"] = 10 ** results[
                f"{self.name}_planet_log_amplitude_{uncertainty[1]}"
            ]

        if svname is not None:
            assert isinstance(svname, object)
            emiss_table.to_csv(svname)
        else:
            return emiss_table

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

        results = self.get_results(uncertainty=uncertainty)[
            [
                "wavelength",
                f"{self.name}_radius_ratio",
                f"{self.name}_radius_ratio_{uncertainty[0]}",
                f"{self.name}_radius_ratio_{uncertainty[1]}",
            ]
        ]
        trans_table = results[["wavelength", f"{self.name}_radius_ratio"]]
        if "hdi" in uncertainty[0]:
            trans_table[f"{self.name}_radius_ratio_neg_error"] = (
                    results[f"{self.name}_radius_ratio"]
                    - results[f"{self.name}_radius_ratio_{uncertainty[0]}"]
            )
            trans_table[f"{self.name}_radius_ratio_pos_error"] = (
                    results[f"{self.name}_radius_ratio_{uncertainty[1]}"]
                    - results[f"{self.name}_radius_ratio"]
            )
        else:
            trans_table[f"{self.name}_radius_ratio_neg_error"] = results[
                f"{self.name}_radius_ratio_{uncertainty[0]}"
            ]
            trans_table[f"{self.name}_radius_ratio_pos_error"] = results[
                f"{self.name}_radius_ratio_{uncertainty[1]}"
            ]

        if svname is not None:
            assert isinstance(svname, object)
            trans_table.to_csv(svname)
        else:
            return trans_table

    def plot_eclipse_spectrum(
            self, table=None, uncertainty=["hdi_16%", "hdi_84%"], ax=None, plotkw={}, **kw
    ):
        if table is not None:
            emission_spectrum = table
            try:
                # ensure the correct columns exist in the emission spectrum table
                assert emission_spectrum[f"{self.name}_depth"]
                assert emission_spectrum[f"{self.name}_depth_neg_error"]
                assert emission_spectrum[f"{self.name}_depth_pos_error"]
                assert emission_spectrum["wavelength"]
            except:
                print(
                    f"The given table doesn't have the correct columns 'wavelength', '{self.name}_depth', "
                    f"{self.name}_depth_pos_error' and '{self.name}_depth_neg_error'"
                )
        else:
            kw["uncertainty"] = uncertainty
            emission_spectrum = self.make_emission_spectrum_table(**kw)
            emission_spectrum["wavelength"] = [
                t.to_value("micron") for t in emission_spectrum["wavelength"].values
            ]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        plt.sca(ax)
        plt.title("Emission Spectrum")
        plt.plot(
            emission_spectrum["wavelength"],
            emission_spectrum[f"{self.name}_depth"],
            "kx",
            **plotkw,
        )
        plt.errorbar(
            emission_spectrum["wavelength"],
            emission_spectrum[f"{self.name}_depth"],
            yerr=[
                emission_spectrum[f"{self.name}_depth_neg_error"],
                emission_spectrum[f"{self.name}_depth_pos_error"],
            ],
            color="k",
            capsize=2,
            linestyle="None",
            **plotkw,
        )
        plt.xlabel("Wavelength (microns)")
        plt.ylabel("Eclipse depth")

    def plot_transmission_spectrum(
        self, table=None, uncertainty=["hdi_16%", "hdi_84%"], ax=None, plotkw={}, **kw
    ):
        if table is not None:
            transmission_spectrum = table
            try:
                # ensure the correct columns exist in the transmission spectrum table
                assert transmission_spectrum[f"{self.name}_radius_ratio"]
                assert transmission_spectrum[f"{self.name}_radius_ratio_neg_error"]
                assert transmission_spectrum[f"{self.name}_radius_ratio_pos_error"]
                assert transmission_spectrum["wavelength"]
            except:
                print(
                    f"The given table doesn't have the correct columns 'wavelength', '{self.name}_radius_ratio', "
                    f"{self.name}_radius_ratio_pos_error' and '{self.name}_radius_ratio_neg_error'"
                )
        else:
            kw["uncertainty"] = uncertainty
            transmission_spectrum = self.make_transmission_spectrum_table(**kw)
            transmission_spectrum["wavelength"] = [
                t.to_value("micron") for t in transmission_spectrum["wavelength"].values
            ]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        plt.sca(ax)
        plt.title("Transmission Spectrum")
        plt.plot(
            transmission_spectrum["wavelength"],
            transmission_spectrum[f"{self.name}_radius_ratio"],
            "kx",
            **plotkw,
        )
        plt.errorbar(
            transmission_spectrum["wavelength"],
            transmission_spectrum[f"{self.name}_radius_ratio"],
            yerr=[
                transmission_spectrum[f"{self.name}_radius_ratio_neg_error"],
                transmission_spectrum[f"{self.name}_radius_ratio_pos_error"],
            ],
            color="k",
            capsize=2,
            linestyle="None",
            **plotkw,
        )
        plt.xlabel("Wavelength (microns)")
        plt.ylabel("Radius Ratio")

