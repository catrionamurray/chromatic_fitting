import matplotlib.pyplot as plt

from src.inject_spectrum import *
# from src.plot.interactive_plots import *
from src.weighted_average_lc import *
from src.recover_transit import *
from src.utils import *
# import theano
from collections import OrderedDict
from astropy import units as u

# Define some constants
R_sun = 696340000 * u.m
M_sun = 1.989 * 10 ** 30 * u.kg
R_earth = 6371000 * u.m

# Define some parameters
bintime=5  #minutes
binwave=0.2  #microns
RANDOM_SEED = 58

class chromatic_model:
    def __repr__(self):
        return "<chromatic_model 🌈>"

    def sample_posterior(self, tune_steps=4000, draws=8000, cores=12, chains=4, target_accept=0.9,plot=True):
        """
        This function samples the posterior distribution of the model parameters
        :param map_soln:
            The result of the pyMC3 fit
        :param model:
            The pyMC3 model object
        :param tune_steps: int
            The number of tuning steps
        :param draws: int
            The number of samples to draw
        :param cores: int
            The number of cores to use
        :param chains: int
            The number of MCMC chains to use
        :param target_accept: float
            The target acceptance rate
        :param plot: bool
            Whether to plot the posterior distribution
        :return:
            The trace object from the MCMC sampling
        """

        # time the sampling
        start_time = ttime.time()

        # set the random seed
        np.random.seed(42)

        with self.model:
            trace = pmx.sample(
                tune=tune_steps,
                draws=draws,
                start=self.result,
                cores=cores,
                chains=chains,
                target_accept=target_accept,
            )
        print("Sampling the posterior took --- %s seconds ---" % (ttime.time() - start_time))

        self.trace = trace

        if plot:
            # plot the priors vs posterior again, now with posterior sample!
            self.plot_fit()

    def plot_trace(self):
        """ Plot the trace of the posterior distribution (see the MCMC chains)
        """
        az.plot_trace(self.trace)

    def cornerplot(self):
        """ Use corner.py to generate the corner plot of the posterior distributions for each variable
        """
        cornerplot(self.model, self.trace, self.P, self.r, self.t0, self.b, self.u_ld, self.mean)

    def summarise(self):
        """ Summarise the results (mean, stddev etc.) of the posterior sampling
        Returns
        ----------
            Arviz summary
        """
        return summarise(self.model, self.trace)

    def change_prior(self,new_prior):
        """ Change the prior distribution of a variable
        """

        priors = [self.r_s_prior, self.m_s_prior, self.r_prior, self.logP_prior, self.t0_prior, self.mean_prior]
        priors_dict = {}

        for p in priors:
            if p['name']==new_prior['name']:
                priors_dict[p['name']] = new_prior
            else:
                priors_dict[p['name']] = p

        self.initialise_model(*priors_dict.values(), init_b=self.init_b,init_u=self.init_u)

    def Normal(self,name,mu,sigma,observed=None):
        """ Specify a normal distribution for the given variable based on PyMC3 format
        Parameters
        ----------
            name : str
                The name of the parameter
            mu : float
                The initial mu value of the normal distribution
            sigma : float
                The initial sigma value of the normal distribution
            observed : float
                (optional, default=None)
                The fixed value of the parameter if we do not want to optimise it
        Returns
        ----------
            dict
                The prior in a dictionary format
        """
        prior = {'name':name,'dist':"Normal",'mu':mu,'sigma':sigma}
        if observed is not None:
            prior['observed']=observed
        return prior

    def Uniform(self,name,testval,lower,upper,observed=None):
        """ Specify a uniform distribution for the given variable based on PyMC3 format
        Parameters
        ----------
            name : str
                The name of the parameter
            testval : float
                The initial value of the uniform distribution
            lower : float
                The lower bound for the uniform distribution
            upper : float
                The upper bound for the uniform distribution
            observed : float
                (optional, default=None)
                The fixed value of the parameter if we do not want to optimise it
        Returns
        ----------
            dict
                The prior in a dictionary format
        """
        prior = {"name":name,"dist":"Uniform", "testval":testval,"lower":lower, "upper":upper}
        if observed is not None:
            prior['observed']=observed
        return prior

    def initialise_model(self,r_s_prior, m_s_prior, r_prior, logP_prior, t0_prior,mean_prior, init_b, init_u, reinit=False):
        """ Create chromatic model (only the wavelength-independent part)

        Thoughts:

        - at the moment we have to reinitialise to reuse this model.

        - I hope there's a better way, but pyMC3/theano doesn't like you reassigning variable names.

        - I have tried creating a subclass of 'chromatic_model' but it also updates the master model when you optimise the submodel.

        - copy.deepclone() doesn't seem to work either
s
        Parameters
        ----------
        logP_prior : dictionary
            priors on log(period)
        t0_prior : dictionary
            priors on the transit epoch
        init_b : float
            Initial guess for impact parameter
        reinit : boolean
            Boolean is true if we are reinitialising the model
        """
        start_time = ttime.time()

        self.priors = []
        self.trace = []
        self.result = []

        if reinit == False:
            self.r_s_prior = r_s_prior
            self.m_s_prior = m_s_prior
            self.logP_prior = logP_prior
            self.t0_prior = t0_prior
            self.init_b = init_b
            self.r_prior = r_prior
            self.mean_prior = mean_prior
            self.init_u = init_u

            if r_prior['dist'] == "Uniform":
                self.init_r = r_prior['testval']
            elif r_prior['dist'] == "Normal":
                self.init_r = r_prior['mu']
            if mean_prior['dist'] == "Uniform":
                self.init_mean = mean_prior['testval']
            elif mean_prior['dist'] == "Normal":
                self.init_mean = mean_prior['mu']

        with pm.Model() as model:

            def init_prior(prior):
                if prior['dist'] == "Uniform":
                    if 'observed' in prior.keys():
                        return pm.Uniform(prior['name'], lower=prior['lower'], upper=prior['upper'],
                                          observed=prior['observed']), prior['observed']
                    else:
                        return pm.Uniform(prior['name'], lower=prior['lower'], upper=prior['upper'],
                                      testval=np.array(prior['testval'])), prior['testval']
                if prior['dist'] == "Normal":
                    if 'observed' in prior.keys():
                        return pm.Normal(prior['name'], mu=prior['mu'], sigma=prior['sigma'],
                                         observed=prior['observed']), prior['observed']
                    else:
                        return pm.Normal(prior['name'], mu=prior['mu'], sigma=prior['sigma']), prior['mu']

            # Initialise (wavelength-indep) priors:
            self.r_s, self.init_r_s = init_prior(r_s_prior)
            self.m_s, self.init_m_s = init_prior(m_s_prior)
            self.logP, self.init_logP = init_prior(logP_prior)
            self.t0, self.init_t0 = init_prior(t0_prior)
            self.P = pm.Deterministic("period", pm.math.exp(self.logP))
            self.init_period = math.exp(self.init_logP)

            # self.b = pm.Uniform('b', lower=0., upper=1., testval=init_b)
            self.b = xo.distributions.ImpactParameter("b", ror=np.array(self.init_r), testval=np.array(init_b))

            # Set up a Keplerian orbit for the planets
            self.orbit = xo.orbits.KeplerianOrbit(period=self.P, t0=self.t0, b=self.b, r_star=self.r_s, m_star=self.m_s)
            # pm.Deterministic('a', self.orbit.a)
            # pm.Deterministic('i', self.orbit.incl * 180 / np.pi)
            # pm.Deterministic('a/r_s', self.orbit.a / self.orbit.r_star)

            self.model = model
        print("Initialising (static) model took --- %s seconds ---" % (ttime.time() - start_time))

    def initialise_model_staticwavelength(self):
        """ Update chromatic model (with the wavelength-dependent part)
        """
        start_time = ttime.time()

        with self.model as model:
            def init_prior(prior):
                if prior['dist'] == "Uniform":
                    if 'observed' in prior.keys():
                        return pm.Uniform(prior['name'], lower=prior['lower'], upper=prior['upper'],
                                          observed=prior['observed']), prior['observed']
                    else:
                        return pm.Uniform(prior['name'], lower=prior['lower'], upper=prior['upper'],
                                          testval=np.array(prior['testval'])), prior['testval']
                if prior['dist'] == "Normal":
                    if 'observed' in prior.keys():
                        return pm.Normal(prior['name'], mu=prior['mu'], sigma=prior['sigma'],
                                         observed=prior['observed']), prior['observed']
                    else:
                        return pm.Normal(prior['name'], mu=prior['mu'], sigma=prior['sigma']), prior['mu']

            self.r, self.init_r = init_prior(self.r_prior)
            self.mean, self.init_mean = init_prior(self.mean_prior)

            # The Kipping (2013) parameterization for quadratic limb darkening paramters
            self.u_ld = xo.distributions.QuadLimbDark("u", testval=np.array(self.init_u))

        print("Initialising r/u/mean took --- %s seconds ---" % (ttime.time() - start_time))

    def reinitialise(self):
        """ Reinitialise chromatic model to redo the optimisation - we can't overwrite variable names
        """
        self.initialise_model(self.r_s_prior, self.m_s_prior, self.r_prior, self.logP_prior, self.t0_prior, self.mean_prior, self.init_b, self.init_u, reinit=True)

    def set_model(self,model):
        """ Replace the model with another (PROBABLY NOT WORKING YET)
        Parameters
        ----------
            model : PyMC3 model
                The PyMC3 we want to replace the current chromatic_model.model with
        """
        self.model = model

    def optimise_model(self,plot=True):
        """ Optimise the model using PyMC3 and the defined prior distributions
        Parameters
        ----------
            plot : boolean
                (optional, default=True)
                Boolean to decide if we plot the results of the optimisation process
        Returns
        ----------
            PyMC3 optimisation result
                The result of the optimisation for each parameter
            PyMC3 model
                The optimised PyMC3 model
        """
        # use our predefined model to optimise the parameters
        # start_time = ttime.time()
        with self.model as model:
            self.r_p = pm.Deterministic('r_p', self.r * self.r_s) # radius of planet in solar radii

            # Compute the model light curve using starry
            light_curves = xo.LimbDarkLightCurve(self.u_ld[0], self.u_ld[1]).get_light_curve(
                orbit=self.orbit, r=self.r_p, t=list(self.x)
            )
            light_curve = pm.math.sum(light_curves, axis=-1) + self.mean

            # Here we track the value of the model light curve for plotting purposes
            pm.Deterministic("light_curves", light_curves)

            # The likelihood function assuming known Gaussian uncertainty
            pm.Normal("obs", mu=light_curve, sd=self.yerr, observed=self.y)
            # Draw 50 samples from the prior distributions (gives an idea of how good your priors are)
            prior_checks = pm.sample_prior_predictive(samples=50, random_seed=RANDOM_SEED)

            # Fit for the maximum a posteriori parameters given the actual lightcurve
            start_time = ttime.time()
            map_soln = pmx.optimize(start=model.test_point)
            print("Optimising model took --- %s seconds ---" % (ttime.time() - start_time))

            self.result = map_soln
            self.priors = prior_checks

            if plot:
                start_time = ttime.time()
                # plot the priors vs posterior BEFORE sampling (only shows priors + best-fit model)
                self.plot_fit()
                print("Plotting took --- %s seconds ---" % (ttime.time() - start_time))

        # print("Optimising model took --- %s seconds ---" % (ttime.time() - start_time))
        return map_soln, model


    def run(self, r, optimisation="weighted_average",plot=True,nwave=10):
        """ Run the optimisation process, using the method chosen by the user

        Parameters
        ----------
            r : rainbow object
                Rainbow object (chromatic) of the spectrum to fit
            optimisation : str
                (optional, default="weighted_average")
                The optimisation method chosen by the user - currently three available: weighted_average, simultaneous and separate_wavelengths
            plot : boolean
                (optional, default=True)
                Boolean to decide if we plot the results of the optimisation process
            nwave : int
                (optional, default=10)
                Number of wavelengths to fit simultaneously if optimisation="simultaneous"
        """

        flux, flux_error, time, wavelength = r.to_nparray()

        if len(time) < 100:
            endpoints = 3
        else:
            endpoints = 50

        if optimisation == "simultaneous":
            datasets = OrderedDict([])
            count = 1
            xs, ys, yerrs, waves = [], [], [], []
            for rf, re, rw in zip(flux[:nwave], flux_error[:nwave], wavelength[:nwave]):
                # x = np.array(time - min(time))#[~np.isnan(rf)]
                x = time
                if len(x) > 0:
                    rf, [re, x] = remove_nans(rf, re, x)
                    y = np.array(rf)
                    yerr = np.array(re)
                    y = y / np.nanmedian(y[-endpoints:])
                    y = y - np.nanmedian(y[-endpoints:])

                    datasets["wavelength_" + str(count)] = (x, y, yerr, self.init_r, self.init_mean, self.init_u)
                    xs.append(x)
                    ys.append(y)
                    yerrs.append(yerr)
                    waves.append(rw)
                    count += 1
            start_time = ttime.time()
            map_soln, model = self.optimise_model_sim(datasets,plot=plot)
            print("Optimising model took --- %s seconds ---" % (ttime.time() - start_time))
            self.x = xs
            self.y = ys
            self.yerr = yerrs
            self.wavelength = waves

        elif optimisation == "weighted_average":
            # weighted_lc = np.nanmedian(flux,axis=0)
            # weighted_err = np.nanmedian(flux_error,axis=0)
            start_time = ttime.time()
            weighted_lc, weighted_err = weighted_avg_lc(time, flux, flux_error, wavelength,
                                                        wavelength_range=[np.min(wavelength), np.max(wavelength)])
            weighted_lc, [time, weighted_err] = remove_nans(weighted_lc, time, weighted_err)
            # x = time - min(time)
            x = time
            y = weighted_lc - np.nanmedian(weighted_lc[:endpoints])
            yerr = weighted_err
            # x = np.array([i.to_value() for i in x])
            print("Weighted Average LC took --- %s seconds ---" % (ttime.time() - start_time))

            self.x = x
            self.y = y
            self.yerr = yerr

            if plot:
                plt.plot(x, y, 'k.')
                plt.show()
                plt.close()

            self.initialise_model_staticwavelength()
            map_soln, model = self.optimise_model(plot=plot)


        elif optimisation == "separate_wavelengths":
            firstrun = True
            xs,ys, yerrs, models,results, waves= [],[],[],[],[],[]

            for rf, re, rw in zip(flux, flux_error, wavelength):
                print("Wavelength: ",rw)
                try:

                    x = np.array(time) # - min(time))
                    y = np.array(rf)
                    y = y / np.nanmedian(y[-endpoints:])
                    y = y - np.nanmedian(y[-endpoints:])
                    yerr = np.array(re)

                    y,[x,yerr] = remove_nans(y,x,yerr)

                    self.x = x
                    self.y = y
                    self.yerr = yerr
                    xs.append(x)
                    ys.append(y)
                    yerrs.append(yerr)

                    if not firstrun:
                        self.reinitialise()
                    else:
                        firstrun=False

                    self.initialise_model_staticwavelength()
                    map_soln, model = self.optimise_model(plot=plot)
                    models.append(model)
                    results.append(map_soln)
                    waves.append(rw)

                except Exception as e:
                    print(e)

            # if we run 'separate_wavelengths' then save lists rather than individual params (model, result etc)
            self.x = xs
            self.y = ys
            self.yerr = yerrs
            self.model = models
            self.result = results
            self.wavelength = waves

        else:
            print("Unrecognised optimisation type, current options are: simultaneous, weighted_average and separate_wavelengths")
            return None, None

        # model_copy = copy.deepcopy(self.model)
        # model_copy2 = theano.clone(self.model)

        # map_soln, model = optimise_model(self, x, y, yerr)

    def optimise_model_sim(self,datasets,plot):
        """ Run the simultaneous optimisation process using PyMC3

        Parameters
        ----------
            datasets : dict
                Dictionary containing the data for each wavelength to fit
            plot : boolean
                Boolean whether to plot the results
        """
        xs,ys,yerrs = [],[],[]

        # THANKS MATHILDE/LIONEL!!
        firstrun = True

        with self.model as model:

            def init_prior(prior):
                if prior['dist'] == "Uniform":
                    if 'observed' in prior.keys():
                        return pm.Uniform(prior['name'], lower=prior['lower'], upper=prior['upper'],
                                          observed=prior['observed']), prior['observed']
                    else:
                        return pm.Uniform(prior['name'], lower=prior['lower'], upper=prior['upper'],
                                          testval=np.array(prior['testval'])), prior['testval']
                if prior['dist'] == "Normal":
                    if 'observed' in prior.keys():
                        return pm.Normal(prior['name'], mu=prior['mu'], sigma=prior['sigma'],
                                         observed=prior['observed']), prior['observed']
                    else:
                        return pm.Normal(prior['name'], mu=prior['mu'], sigma=prior['sigma']), prior['mu']

            for n, (name, (x, y, yerr, r_ratio, mean, ldc)) in enumerate(datasets.items()):
                if len(x) > 0:
                    # We define the per-instrument parameters in a submodel so that we don’t have to prefix the names manually
                    # with pm.Model(name=name, model=model):
                    # The limb darkening #It’s the same filter in this case
                    u = xo.QuadLimbDark(f'{name}_u', testval=ldc)
                    star = xo.LimbDarkLightCurve(u)

                    # The radius ratio
                    # depth = pm.Uniform('depth',lower=0,upper=1,testval=r_ratio**2)
                    # ror = star.get_ror_from_approx_transit_depth(depth, self.b)# pm.Deterministic('ror',depth)
                    ror_prior = self.r_prior.copy()
                    ror_prior['name'] = f'{name}_ror'
                    ror, init_ror = init_prior(ror_prior)
                    # ror = pm.Uniform(f'{name}_ror', lower=0.01, upper=0.5,
                    #                  testval=r_ratio)  # star.get_ror_from_approx_transit_depth(depth, b))
                    r_p = pm.Deterministic(f'{name}_r_p', ror * self.r_s)  # In solar radius
                    r = pm.Deterministic(f'{name}_r', r_p * 1 / R_sun)

                    # lightcurve mean
                    # mean = pm.Normal(f'{name}_mean', mu=mean, sigma=0.1)
                    mean_prior = self.mean_prior.copy()
                    mean_prior['name'] = f'{name}_mean'
                    mean,init_mean = init_prior(mean_prior)

                    # starry light-curve
                    light_curve = star.get_light_curve(orbit=self.orbit, r=r_p, t=x)
                    light_curves = pm.Deterministic(f'{name}_light_curves', pm.math.sum(light_curve, axis=-1) + mean)

                    if firstrun:
                        prior_checks = pm.sample_prior_predictive(samples=50, random_seed=RANDOM_SEED)
                        firstrun = False
                    # Systematics and final model
                    #             w = pm.Flat('w',shape=len(X))
                    #             systematics = pm.Deterministic('systematics',w@X)
                    # residuals = pm.Deterministic(“residuals”, y - transit)
                    # systematics=X
                    # mu = pm.Deterministic('mu', transit)  # +systematics)
                    # Likelihood function
                    pm.Normal(f'{name}_obs', mu=light_curves, sd=yerr, observed=y)

                    xs.append(x)
                    ys.append(y)
                    yerrs.append(yerr)
            # Maximum a posteriori
            # --------------------
            opt = pmx.optimize(start=model.test_point)
            opt = pmx.optimize(start=opt, vars=[ror, u, mean])
            # opt = pmx.optimize(start=opt, vars=[depth[1]])
            self.result = opt
            self.priors = prior_checks
            self.x = xs
            self.y = ys
            self.yerr = yerrs

        if plot:
            self.plot_fit()

        return opt, model

    def plot_fit(self):
        """ Plot the lightcurve compared to samples from both the priors and posteriors (only available after fitting)
        """
        sim_wavelengths = False

        # we need to treat the different opt methods slightly differently
        if 'light_curves' in self.result.keys():
            figsize = (12, 6)
            # if we have either used the weighted_average or separate_wavelengths:
            lc_model = [self.result["light_curves"]]
            x, y, yerr = [self.x.copy()], [self.y.copy()], [self.yerr.copy()]
            nrows = 1
            try:
                priors = [self.priors['light_curves']]
                mean_priors = [self.priors['mean']]
            except Exception as e:
                print(e)
        else:
            # if we have done simultaneous wavelength optimisation
            lc_model, priors, mean_priors = [], [], []
            nrows = 0
            sim_wavelengths = True
            for k, v in self.result.items():
                if "light_curves" in k:
                    lc_model.append(v)
                    nrows = nrows + 1
            try:
                for k, v in self.priors.items():
                    if "light_curves" in k:
                        priors.append(v)
                    if "mean" in k:
                        mean_priors.append(v)
            except Exception as e:
                print(e)
            x, y, yerr = self.x.copy(), self.y.copy(), self.yerr.copy()
            figsize = (12, 4*nrows)

        # set up plot
        _,ax = plt.subplots(ncols=2,nrows=nrows,sharex=True,sharey=True,figsize=figsize)

        for n in range(nrows):
            if nrows == 1:
                ax_prior = ax[0]
                ax_posterior = ax[1]
            else:
                ax_prior = ax[n,0]
                ax_posterior = ax[n,1]

            ax_prior.plot(x[n], y[n], ".k", ms=4)
            ax_posterior.plot(x[n], y[n], ".k", ms=4)

            # plot prior samples
            # ******************
            try:
                firstprior = True
                # loop over all lightcurves (for every wavelength):
                for prior, mean_prior in zip(priors, mean_priors):
                    # loop over every prior sample:
                    for lc,mean in zip(prior,mean_prior):
                        if firstprior:
                            ax_prior.plot(x[n], lc+mean, color="C1", lw=1, alpha=0.3, label="Prior Sample (n=50)")
                            firstprior = False
                        else:
                            ax_prior.plot(x[n], lc+mean, color="C1", lw=1, alpha=0.3)
                    # regenerating the orbit for each prior is SUPER SLOW!! And there's no need - each lc is already stored!

            except Exception as e:
                print(e)
            # ******************


            # plot initial parameter guess
            # ******************
            light_curve = generate_xo_orbit(self.init_period, self.init_t0, self.init_b, self.init_u, self.init_r, x[n]) + self.init_mean
            ax_prior.plot(x[n], light_curve, color="green", lw=1, alpha=1,linestyle='--',label="Initial Guess")
            # ******************


            # plot 50 chains sampled from the psterior (MCMC):
            # ******************
            try:
                firsttrace = True

                # plot the 16-84th posterior percentile region
                if sim_wavelengths:
                    trace = self.trace[f"wavelength_{n+1}_light_curves"]
                else:
                    trace = self.trace["light_curves"]

                q16, q50, q84 = np.percentile(trace, [16, 50, 84], axis=(0))

                ax_posterior.fill_between(x[n], q16.flatten(), q84.flatten(),color="C1", alpha=0.2, label="Posterior (16-84th percentile)")

                # plot 50 individual posterior samples
                for i in np.random.randint(len(self.trace) * self.trace.nchains, size=50):
                    if firsttrace:
                        # add the legend label to only the first prior line (to avoid 50 legend entries)
                        ax_posterior.plot(x[n], trace[i], color="C1", lw=1, alpha=0.3,
                                              label='Posterior Sample (n=50)')
                        firsttrace = False
                    else:
                        ax_posterior.plot(x[n], trace[i], color="C1", lw=1, alpha=0.3)
            except Exception as e:
                print(e)
            # ******************

            ax_prior.errorbar(x[n], y[n],yerr[n], c='k', alpha=0.2)
            ax_posterior.errorbar(x[n], y[n], yerr[n], c='k', alpha=0.2)
            ax_posterior.plot(x[n], lc_model[n], lw=1, label="model")
            ax_prior.set_xlim(x[n].min(), x[n].max())
            ax_posterior.set_ylim(y[n].min()-0.01, y[n].max()+0.01)
            ax_prior.set_ylabel("rel.\nflux")
            ax_prior.set_xlabel("time [days]")
            ax_posterior.set_xlabel("time [days]")
            # handles, labels = ax_prior.gca().get_legend_handles_labels()
            # by_label = OrderedDict(zip(labels, handles))
            # ax_prior.legend(by_label.values(), by_label.keys())
            # handles, labels = ax_posterior.gca().get_legend_handles_labels()
            # by_label = OrderedDict(zip(labels, handles))
            # ax_posterior.legend(by_label.values(), by_label.keys())
            ax_prior.legend(loc="upper right",fontsize=10,facecolor='white', frameon=True,framealpha=0.6)
            ax_posterior.legend(loc="upper right",fontsize=10, facecolor='white', frameon=True,framealpha=0.6)
            ax_prior.set_title("Data + Prior Models")
            ax_posterior.set_title("Data + Posterior Models")

        plt.tight_layout()
        plt.show()
        plt.close()

    def plot_simultaneous_fit(self):
        # we need to treat the different opt methods slightly differently
        sim_wavelengths = False
        if 'light_curves' in self.result.keys():
            lc_model = [self.result["light_curves"]]
            x,y,yerr = [self.x.copy()],[self.y.copy()],[self.yerr.copy()]
            nrows = 1
            # try:
            #     priors = [self.priors['light_curves']]
            #     mean_priors = [self.priors['mean']]
            # except Exception as e:
            #     print(e)
        else:
            lc_model, priors, mean_priors = [],[],[]
            nrows = 0
            sim_wavelengths = True
            for k, v in self.result.items():
                if "light_curves" in k:
                    lc_model.append(v)
                    nrows = nrows + 1
            # try:
            #     for k, v in self.priors.items():
            #         if "light_curves" in k:
            #             priors.append(v)
            #         if "mean" in k:
            #             mean_priors.append(v)
            # except Exception as e:
            #     print(e)
            x, y, yerr = self.x.copy(), self.y.copy(), self.yerr.copy()
            wavelengths = self.wavelength

        # set up plot
    #     _,ax = plt.subplots(ncols=2,nrows=nrows,sharex=True,sharey=True,figsize=(12,12))
        _,ax = plt.subplots(ncols=2,nrows=1,sharex=True,sharey=True,figsize=(12,12))

        offset, dy, miny, maxy = -1, 0.03, 100, -100
        firsttrace = True

        for n in range(nrows):
            ax_prior = ax[0]
            ax_posterior = ax[1]
            if nrows > 1:
                y[n] = y[n] - offset

            if firsttrace:
                ax_prior.plot(x[n], y[n], ".k", ms=4, label="data")
            else:
                ax_prior.plot(x[n], y[n], ".k", ms=4)

            ax_posterior.plot(x[n], y[n], ".k", ms=4)#, label="data")

            # plot initial parameter guess
            # ******************
            light_curve = generate_xo_orbit(self.init_period, self.init_t0, self.init_b, self.init_u, self.init_r, x[n]) + self.init_mean
            if firsttrace:
                ax_prior.plot(x[n], light_curve-offset, color="green", lw=1, alpha=1,linestyle='--',label="Initial Guess")
            else:
                ax_prior.plot(x[n], light_curve-offset, color="green", lw=1, alpha=1,linestyle='--')
            # ******************

            # plot 50 chains sampled from the posterior (MCMC):
            # ******************
            try:

                # plot the 16-84th posterior percentile region
                if sim_wavelengths:
                    trace = self.trace[f'wavelength_{n+1}_light_curves']
                else:
                    trace = self.trace["light_curves"]

                q16, q50, q84 = np.percentile(trace, [16, 50, 84], axis=(0))
                if firsttrace:
                    ax_posterior.fill_between(x[n], q16.flatten()-offset, q84.flatten()-offset,color="C1", alpha=0.2, label="Posterior (16-84th percentile)")
                else:
                    ax_posterior.fill_between(x[n], q16.flatten()-offset, q84.flatten()-offset,color="C1", alpha=0.2)

                # plot 50 individual posterior samples
                for i in np.random.randint(len(v.trace) * self.trace.nchains, size=50):
                    if firsttrace:
                        # add the legend label to only the first prior line (to avoid 50 legend entries)
                        ax_posterior.plot(x[n], trace[i]-offset, color="C1", lw=1, alpha=0.3,
                                     label='Posterior Sample (n=50)')
                        firsttrace = False
                    else:
                        ax_posterior.plot(x[n], trace[i]-offset, color="C1", lw=1, alpha=0.3)
            except Exception as e:
                print(e)
            # ******************

            if y[n].min() < miny:
                miny = y[n].min() - 0.01
            if y[n].max() > maxy:
                maxy = y[n].max() + 0.01

            ax_prior.errorbar(x[n], y[n],yerr[n], c='k', alpha=0.2)
            ax_posterior.errorbar(x[n], y[n], yerr[n], c='k', alpha=0.2)
            ax_posterior.plot(x[n], lc_model[n]-offset, lw=1, label="model")
            ax_prior.set_xlim(x[n].min(), x[n].max())
            offset = offset + dy

        ax_posterior.set_ylim(miny, maxy)
        ax_prior.set_ylabel("relative flux")
        ax_prior.set_xlabel("time [days]")
        ax_posterior.set_xlabel("time [days]")
        ax_prior.legend(loc="upper right",fontsize=10,facecolor='white',frameon=True, framealpha=0.6)
        ax_posterior.legend(loc="upper right",fontsize=10, facecolor='white',frameon=True, framealpha=0.6)
        ax_prior.set_title("Data + Prior Models")
        ax_posterior.set_title("Data + Posterior Models")
        plt.show()
        plt.close()


class chromatic_submodel(chromatic_model):

    def __init__(self,chromatic_model):
        self.model = chromatic_model.model
        self.r = chromatic_model.r
        self.P = chromatic_model.P
        self.t0 = chromatic_model.t0
        self.mean = chromatic_model.mean
        self.u_ld = chromatic_model.u_ld
        self.b = chromatic_model.b
        self.orbit = chromatic_model.orbit

def quicklook_priors(r,cm):
    """
    Quicklook plot of the priors compared to the averaged LC
    Parameters
    __________
        r : Rainbow object
            data rainbow object
        cm : chromatic_model object
            initialised chromatic_model object
    """
    light_curve = generate_xo_orbit(cm.init_period, cm.init_t0, cm.init_b, cm.init_u, cm.init_r, np.array(r.time))\
                  + cm.init_mean
    plt.plot(r.time, light_curve+1,'b')
    plt.plot(r.time, np.median(r.flux,axis=0),'k.')
    plt.show()
    plt.close()

def generate_xo_orbit(period,t0,b,u,r,x):
    """ Generate the Exoplanet orbit from orbital parameters

    Parameters
    ----------
        period : float
            Orbital period
        t0 : float
            Epoch of the transit (days)
        b : float
            Impact parameter
        u : list or np.array
            Quadratic limb-darkening coefficients
        r : float
            planetary radius in solar radii
        x : list or np.array
            x-axis (time)
    Returns
    ---------
        LimbDarkLightCurve
            Light curve (with quadratic limb-darkening) from Exoplanet
    """
    orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b)
    light_curve = xo.LimbDarkLightCurve([u[0], u[1]]).get_light_curve(
        orbit=orbit, r=r, t=list(x)).eval()
    return light_curve

def import_patricio_model():
    ''' Import spectral model
    Returns
    ---------
        model : PlanetarySpectrumModel
            Planetary spectrum model
        planet_params : dict
            Planetary parameters
        wavelength : np.array
            Wavelengths
        transmission : np.array
            Transmission values
    '''
    x = pickle.load(open('data_challenge_spectra_v01.pickle', 'rb'))
    # lets load a model
    planet = x['WASP39b_NIRSpec']
    planet_params = x['WASP39b_parameters']
    # print(planet_params)

    wavelength = planet['wl']
    transmission = planet['transmission']
    table = Table(dict(wavelength=planet['wl'], depth=np.sqrt(planet['transmission'])), meta=planet_params)

    # set up a new model spectrum
    model = PlanetarySpectrumModel(table=table, label='injected model')
    return model, planet_params, wavelength, transmission

def add_ld_coeffs(model, planet_params, wavelength,transmission,star_params,ld_eqn = 'quadratic',mode = "NIRSpec_Prism",plot=False):
    ''' Add the wavelength-dep limb-darkening coefficients to the transit model to inject

    Parameters
    ----------
        model : PlanetarySpectrumModel
            The spectral model of the injected planet
        planet_params : dict
            Synthetic planet parameters from user.
        wavelength : np.array or list
            The centres of the wavelength bins to calculate the LD coeffs over.
        transmission : np.array or list
            The transit depths as a function of wavelength.
        star_params : dict
            Stellar parameters from user. For now, must contain: M_H, teff, logg.
        ld_eqn : str
            (optional, default="quadratic")
            The equation used to calculate the limb-darkening coeffs, options "linear", "quadratic", "nonlinear" and
            "threeparam" (NOTE: There is a current issue open in ExoTiC to add a selection criteria so that only the
            desired coefficients are returned).
        mode : str
            (optional, default="NIRCam_F322W2")
            The instrument/wavelength band defined by ExoTiC-LD - several available for JWST.
    Returns
    ---------
        model : synthetic_planet
            Synthetic planet object to inject into Rainbow object.
    '''

    #  define where the zenodo LD files are stored (needed for ExoTiC)
    dirsen = '/Users/catrionamurray/Documents/Postdoc/CUBoulder/exotic-ld_data'
    #  calculate the LD coeffs for a series of wavelengths and transit depths
    model_ld = generate_spectrum_ld(wavelength, np.sqrt(transmission), star_params, planet_params, dirsen, mode=mode,
                                    ld_eqn=ld_eqn, ld_model='1D', plot_model=plot)

    return model_ld

def main():
    # set initial parameter estimates:
    init_t0 = 0.1
    init_period = 2  # 4.055259
    period_error = 2  # 0.000009
    init_b = 0.44
    init_r = 0.14
    init_mean = 0.0
    init_u = [1.3, -0.5]
    r_s = 1
    m_s = 1

    # create chromatic model:
    cm = chromatic_model()

    # init priors (use same distribution format as pyMC3):
    m_s_prior = cm.Normal('m_s', mu=1, sigma=0.05, observed=r_s)
    r_s_prior = cm.Normal('r_s', mu=1, sigma=0.05, observed=m_s)
    r_prior = cm.Uniform("r", lower=0.01, upper=0.3, testval=init_r)
    mean_prior = cm.Normal("mean", mu=init_mean, sigma=0.005)
    t0_prior = cm.Normal("t0", mu=init_t0, sigma=0.05)
    logP_prior = cm.Normal("logP", mu=np.log(init_period), sigma=np.log(period_error))

    # Initialise the (wavelength-independent) model:
    start_time = ttime.time()
    cm.initialise_model(r_s_prior, m_s_prior, r_prior, logP_prior, t0_prior, mean_prior, init_b=init_b, init_u=init_u)
    print("Initialising (static) model took --- %s seconds ---" % (ttime.time() - start_time))

    # load Patricio's spectrum model:
    model_staticld, planet_params, wavelength, transmission = import_patricio_model()

    # add wavelength-dep limb-darkening coeffs:
    model_ld = add_ld_coeffs(model_staticld, planet_params, wavelength, transmission,
                             star_params={"M_H": -0.03, "Teff": 5326.6, "logg": 4.38933})

    # return synthetic rainbow + rainbow with injected transit:
    bintime = 5
    binwave = 0.2
    r, i = inject_spectrum(model_ld, snr=1000, dt=bintime, res=50)

    # bin in time and wavelength (to speed up fitting)
    b_withouttransit = r.bin(
        dw=binwave * u.micron, dt=bintime * u.minute
    )
    b_withtransit = i.bin(
        dw=binwave * u.micron, dt=bintime * u.minute
    )

    result, model = cm.run(b_withtransit, optimisation='simultaneous', nwave=1, plot=False)
    cm.sample_posterior(plot=False)
    # print(cm.summarise())

# main()

# def plot_model_and_recovered(model, *trans_spect_file):
#     from astropy import units as u
#
#     print(trans_spect_file)
#
#     fig, ax = plt.subplots(nrows=4, figsize=(12, 6), sharex=True)
#     ax[0].plot(model.table['wavelength'] * u.Unit("um"), 100 * model.table['depth'], 'k', alpha=0.3, markersize=10)
#     ax[0].set_ylabel("(Rp/R*)^2 [%]")
#     plt.xlabel("Wavelength (microns)")
#     ax[1].set_ylabel("Depth Residuals [%]")
#     ax[2].set_ylabel("Limb-Darkening Coeff")
#     ax[3].set_ylabel("LD Residuals")
#
#     num_ld_coeffs = len(model.ld_coeffs[model.modemask == 0][0])
#     model_u = []
#     for ld in range(num_ld_coeffs):
#         plot_kw = dict(alpha=0.75, linewidth=2, label="LD Coeff " + str(ld))
#         extract_coeff = [l[ld] for l in model.ld_coeffs[model.modemask == 0]]
#         ax[2].plot(model.table['wavelength'][model.modemask == 0], extract_coeff, alpha=0.6)
#         model_u.append(extract_coeff)
#
#     for t in trans_spect_file:
#         resid, resid_u0, resid_u1 = [], [], []
#         trans_spect = pd.read_csv(t)
#         trans_spect['depth'] = trans_spect['r']  # **2
#         trans_spect['uncertainty'] = 2 * trans_spect['r'] * trans_spect['r_err']
#         halfbinwidths = [0.5 * (t - s) for s, t in
#                          zip(trans_spect['wavelength'], trans_spect['wavelength'][1:])] * u.Unit("um")
#
#         for d, w, u0, u1 in zip(trans_spect['depth'], trans_spect['wavelength'], trans_spect['u0'], trans_spect['u1']):
#             resid.append(100 * (d - model.table['depth'][find_nearest(model.table['wavelength'], w)]))
#             resid_u0.append(u0 - model_u[0][find_nearest(model.table['wavelength'][model.modemask == 0], w)])
#             resid_u1.append(u1 - model_u[1][find_nearest(model.table['wavelength'][model.modemask == 0], w)])
#
#         ax[0].plot(trans_spect['wavelength'] * u.Unit("um"), 100 * trans_spect['depth'], '.', markersize=5, label=t)
#         # plt.errorbar(trans_spect['wavelength'].values*u.Unit("um"),100*trans_spect['depth'],xerr=np.mean(halfbinwidths),yerr=100*trans_spect['uncertainty'],capsize=3,c='k',linestyle="None")
#         ax[0].errorbar(trans_spect['wavelength'].values * u.Unit("um"), 100 * trans_spect['depth'],
#                        yerr=100 * trans_spect['uncertainty'], capsize=3, color='k', alpha=0.4, linestyle="None")
#
#         ax[1].plot(trans_spect['wavelength'] * u.Unit("um"), resid, alpha=0.8, label=t)
#
#         ax[2].plot(trans_spect['wavelength'] * u.Unit("um"), trans_spect['u0'], '.', markersize=5, label=t)
#         ax[2].errorbar(trans_spect['wavelength'].values * u.Unit("um"), trans_spect['u0'], yerr=trans_spect['u0_err'],
#                        capsize=3, color='k', alpha=0.2, linestyle="None")
#         ax[2].plot(trans_spect['wavelength'] * u.Unit("um"), trans_spect['u1'], '.', markersize=5, label=t)
#         ax[2].errorbar(trans_spect['wavelength'].values * u.Unit("um"), trans_spect['u1'], yerr=trans_spect['u1_err'],
#                        capsize=3, color='k', alpha=0.2, linestyle="None")
#
#         ax[3].plot(trans_spect['wavelength'] * u.Unit("um"), resid_u0, alpha=0.8, label=t)
#         ax[3].plot(trans_spect['wavelength'] * u.Unit("um"), resid_u1, alpha=0.8, label=t)
#
#     ax[0].legend(fontsize=8)
#     ax[1].legend(fontsize=8)
#     ax[2].legend(fontsize=8)
#     ax[3].legend(fontsize=8)
#     plt.show()