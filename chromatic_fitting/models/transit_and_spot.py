from ..imports import *

from .lightcurve import *

import starry

starry.config.lazy = True
starry.config.quiet = True

import theano
# import aesara_theano_fallback.tensor as tt

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
            spot_smoothing = None,
            nspots: int = 1,
            type_of_model: str = "planet",
            plot_min_radius = False,
            fit_for_n_spots = False,
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
        self.ydeg = ydeg
        self.fit_for_n_spots = fit_for_n_spots

        if self.nspots > 0:
            if spot_smoothing is not None:
                self.spot_smoothing = spot_smoothing
            else:
                self.spot_smoothing = 2/ydeg

            if plot_min_radius:
                self.plot_minimum_spot_radius()

        # only require a constant (0th order) term:
        self.required_parameters = ["A", "rs", "ms", "prot", "u", "stellar_amp", "stellar_inc", "stellar_obl",
                                    "mp", "rp", "inc", "amp", "period", "omega", "ecc", "t0",
                                    "spot_contrast"]

        if fit_for_n_spots:
            for nspots in range(self.nspots):
                for n in range(nspots):
                    self.required_parameters.extend([f"{nspots+1}_spot_{n + 1}_radius",
                                                     f"{nspots+1}_spot_{n + 1}_latitude",
                                                     f"{nspots+1}_spot_{n + 1}_longitude"])
            self.required_parameters.extend(["nspots"])
        else:
            for n in range(self.nspots):
                self.required_parameters.extend([f"{self.nspots}_spot_{n + 1}_radius",
                                                 f"{self.nspots}_spot_{n + 1}_latitude",
                                                 f"{self.nspots}_spot_{n + 1}_longitude"])

        super().__init__(**kw)
        self.set_defaults()
        self.set_name(name)
        self.metadata = {}
        self.model = self.transit_spot_model
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
        self.defaults = dict(A=1, rs=1, ms=1, stellar_amp=1, stellar_inc=90, stellar_obl=0.0, prot=1000, u=[0.1, 0.1],
                             # spot_contrast=0.5, spot_radius=20, spot_latitude=0.0, spot_longitude=0.0,
                             mp=1, rp=1, inc=90, amp=5e-3, omega=0.0, period=10, ecc=0.0, t0=0.0, spot_contrast=0.5) #omega=100,

        if self.fit_for_n_spots:
            for nspots in range(self.nspots):
                for n in range(nspots):
                    self.defaults[f"{nspots+1}_spot_{n + 1}_radius"] = 20
                    self.defaults[f"{nspots+1}_spot_{n + 1}_latitude"] = 0.0
                    self.defaults[f"{nspots+1}_spot_{n + 1}_longitude"] = 0.0
            self.defaults['nspots'] = self.nspots
        else:
            for n in range(self.nspots):
                self.defaults[f"{self.nspots}_spot_{n + 1}_radius"] = 20
                self.defaults[f"{self.nspots}_spot_{n + 1}_latitude"] = 0.0
                self.defaults[f"{self.nspots}_spot_{n + 1}_longitude"] = 0.0

    def plot_optimized_map(self, opt):
        opt_copy = opt.copy()
        for k, v in self.parameters.items():
            if k not in opt_copy.keys():
                print(k)
                opt_copy[k] = v.value
        opt_copy[f'{self.name}_u'] = opt_copy[f'{self.name}_u'][0]
        flux, starry_model = self.setup_star_and_planet(param_i=opt_copy, name=f"{self.name}_", method='starry',
                                                     time=self.data.time,
                                                     flux_model=[])

        starry_model.show(t=opt_copy[f"{self.name}_t0"])
        starry_model.primary.map.show(theta=starry_model.primary.theta0)
        return flux, starry_model

    def plot_minimum_spot_radius(self):
        smstrs = [0.0, 1.0, 2.0, self.spot_smoothing*self.ydeg]
        ydegs = np.arange(10, 36)  # [10, 15, 20, 25, 30]
        radii = np.arange(1, 30)
        tol = 0.1
        rmin = np.zeros((len(smstrs), len(ydegs)))
        for i, smstr in enumerate(smstrs):
            for j, ydeg in enumerate(ydegs):
                map = starry.Map(ydeg)
                # error = np.zeros(len(radii))
                for radius in radii:
                    map.reset()
                    map.spot(contrast=1, radius=radius, spot_smoothing=smstr / ydeg)
                    if (
                            np.abs(np.mean(map.intensity(lon=np.linspace(0, 0.75 * radius, 50))))
                            < tol
                    ):
                        rmin[i, j] = radius
                        break

        plt.figure(figsize=(6, 6))
        plt.plot(ydegs, rmin[0], "-o", label=r"$0$")
        plt.plot(ydegs, rmin[1], "-o", label=r"$1 / l_\mathrm{max}$")
        plt.plot(ydegs, rmin[2], "-o", label=r"$2 / l_\mathrm{max}$ (default)")
        plt.plot(ydegs, rmin[3], "-o", c='k', label=f"${self.spot_smoothing*self.ydeg}"+r"$ / l_\mathrm{max}$ (user)")
        plt.axvline(self.ydeg, c='k', alpha=0.8)
        ind = np.where(np.array(ydegs) == self.ydeg)[0][0]
        plt.axhline(rmin[0][ind], c='k', alpha=0.2)
        plt.axhline(rmin[1][ind], c='k', alpha=0.2)
        plt.axhline(rmin[2][ind], c='k', alpha=0.2)
        plt.axhline(rmin[3][ind], c='k', alpha=0.8)
        plt.legend(title="smoothing", fontsize=10)
        plt.xticks([10, 15, 20, 25, 30])
        plt.xlabel(r"spherical harmonic degree $l_\mathrm{max}$")
        plt.ylabel(r"minimum spot radius [degrees]")
        plt.show()

        print(f"""
        With {self.ydeg} orders of spherical harmonics the minimum spot radius we can model
        (with <{tol*100}% error on the contrast) is {rmin[3][ind]} degrees.
        """)


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
                inc=param_i[f"{name}stellar_inc"],
                obl=param_i[f"{name}stellar_obl"],
                t0=param_i[f"{name}t0"],
                theta0=0.0,#360*(param_i[f"{name}t0"]/param_i[f"{name}prot"]) # set the initial orientation of the map as
                                                                        # mid transit
            )

            star.map[1:] = param_i[f"{name}u"]

            if self.fit_for_n_spots:
                nspots = param_i[f"{name}nspots"]
                if pm.math.eq(nspots, 0) == True:
                    pass
                else:
                    for ns in range(self.nspots):
                        if pm.math.eq(nspots, ns + 1):
                            for spot_i in range(ns + 1):
                                star.map.spot(contrast=param_i[f"{name}spot_contrast"],
                                              radius=param_i[f"{name}{ns + 1}_spot_{spot_i + 1}_radius"],
                                              lat=param_i[f"{name}{ns + 1}_spot_{spot_i + 1}_latitude"],
                                              lon=param_i[f"{name}{ns + 1}_spot_{spot_i + 1}_longitude"],
                                              spot_smoothing=self.spot_smoothing)

            else:
                for spot_i in range(self.nspots):
                    star.map.spot(contrast=param_i[f"{name}spot_contrast"], #param_i[f"{name}spot_{spot_i + 1}_contrast"],
                                  radius=param_i[f"{name}{self.nspots}_spot_{spot_i + 1}_radius"],
                                  lat=param_i[f"{name}{self.nspots}_spot_{spot_i + 1}_latitude"],
                                  lon=param_i[f"{name}{self.nspots}_spot_{spot_i + 1}_longitude"],
                                  spot_smoothing=self.spot_smoothing)

            planet = starry.kepler.Secondary(
                starry.Map(ydeg=0, amp=param_i[f"{name}amp"]),  # the surface map
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
            flux_model.append(param_i[f"{name}A"]-1 + sys.flux(time))

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


                    flux_model, _ = self.setup_star_and_planet(name=name, method=self.method, param_i=param_i,
                                                               time=data.time.to_value('d'),
                                                               flux_model=flux_model)

                    # save the radius ratio for generating the transmission spectrum later:
                    rr = Deterministic(f"{name}radius_ratio[{i + j}]",
                                       (param_i[f"{name}rp"] * (1 * u.R_earth).to_value("R_sun")) / param_i[
                                           f"{name}rs"])
                    transit_depth = Deterministic(f"{name}depth[{i + j}]", rr ** 2)

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

    def transit_spot_model(self, params: dict, i: int = 0, save_keplerian_system=False) -> np.array:
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
            # self.keplerian_system = sys
            if save_keplerian_system:
                if hasattr(self, 'keplerian_system'):
                    self.keplerian_system[f'w{i}'] = sys
                else:
                    self.keplerian_system = {f'w{i}': sys}
            transit_spot = eval_in_model(flux_model[0])
        return transit_spot

    def show_system(self, i=0, **kw):
        if self.method == "starry":
            if hasattr(self, 'keplerian_system'):
                self.keplerian_system[f'w{i}'].show(**kw)

    def make_transmission_spectrum_table(
            self, uncertainty=["hdi_16%", "hdi_84%"], svname=None
    ):
        """
        Generate and return a transmission spectrum table
        """

        # THIS IS 100% A HACK TO INCLUDE RADIUS_RATIO IN THE RESULTS TABLE:
        _, pm_models = self.choose_model_based_on_optimization_method()
        with pm_models[0]:
            try:
                self.parameters[f'{self.name}_radius_ratio'] = Normal('radius_ratio', mu=0.1, sigma=0.1)
            except:
                pass

        return TransitModel.make_transmission_spectrum_table(self, uncertainty=uncertainty, svname=svname)

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
        self.parameters[f'{self.name}_radius_ratio'] = self.summary['mean'][f'{self.name}_radius_ratio[0]']
        TransitModel.plot_transmission_spectrum(self, table=table, uncertainty=uncertainty, ax=ax, plotkw=plotkw, **kw)

    def plot_spot_spectrum(
        self, **kw
    ):
        """
        Plot the spot spectrum (specifically the spot contrast as a function of wavelength).

        Parameters
        ----------
        table: [Optional] Table to use as spot spectrum (otherwise the default is to use the MCMC sampling results.
        The table must have the following columns: "{self.name}_spot_contrast", "{self.name}_spot_contrast_neg_error",
        "{self.name}_spot_contrast_pos_error", "wavelength".
        uncertainty: [Optional] List of the names of parameters to use as the lower and upper errors. Options: "hdi_16%", "hdi_84%",
        "sd" etc. Default = ["hdi_16%", "hdi_84%"].
        ax: [Optional] Pass a preexisting matplotlib axis is to be used instead of creating a new one.Default = None.
        plotkw: [Optional] Dict of kw to pass to the spot specrum plotting function.
        kw: [Optional] kw to pass to .make_spot_spectrum_table()

        Returns
        -------

        """
        self.plot_spectrum(param="spot_contrast", name_of_spectrum="Stellar Spot", **kw)

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

    def sample(
            self,
            summarize_step_by_step=False,
            summarize_kw={"round_to": 7, "hdi_prob": 0.68, "fmt": "wide"},
            sampling_method=pmx.sample,
            sampling_kw={"init": "adapt_full", "mp_ctx": "spawn"},
            **kw,
        ):
        LightcurveModel.sample(
            summarize_step_by_step=summarize_step_by_step,
            summarize_kw=summarize_kw,
            sampling_method=sampling_method,
            sampling_kw=sampling_kw,
            **kw
        )