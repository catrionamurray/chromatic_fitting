from src.inject_spectrum import *
# from src.plot.interactive_plots import *
from src.weighted_average_lc import *
from src.recover_transit import *
from src.utils import *
# import theano
from collections import OrderedDict
from astropy import units as u

R_sun = 696340000 * u.m
M_sun = 1.989 * 10 ** 30 * u.kg
R_earth = 6371000 * u.m

bintime=5 # minutes
binwave=0.1 # microns
RANDOM_SEED = 58

# r_prior = pm.Uniform("r", lower=0.01, upper=0.5, testval=np.array(init_r))
# mean_prior =  pm.Normal("mean", mu=init_mean, sigma=0.1)
# t0_prior = pm.Normal("t0", mu=init_t0, sigma=1.0)
# logP = pm.Normal("logP", mu=np.log(init_period), sigma=period_error)
# period = pm.Deterministic("period", pm.math.exp(logP))



class chromatic_model:
    def sample_posterior(self,map_soln,plot=True):
        # sample the posterior
        trace = sample(map_soln, self.model)
        self.trace = trace

        if plot:
            # plot the priors vs posterior again, now with posterior sample!
            self.plot_fit(prior_checks=self.priors)

    def cornerplot(self):
        # plot corner plot (posterior distribution for each variable)
        cornerplot(self.model, self.trace, self.P, self.r, self.t0, self.b, self.u_ld, self.mean)

    def summarise(self):
        # plot summary (mean, std etc.)
        return summarise(self.model, self.trace)

    def Normal(self,name,mu,sigma,observed=None):
        prior = {'name':name,'dist':"Normal",'mu':mu,'sigma':sigma}
        if observed is not None:
            prior['observed']=observed
        return prior

    def Uniform(self,name,testval,lower,upper,observed=None):
        prior = {"name":name,"dist":"Uniform", "testval":testval,"lower":lower, "upper":upper}
        if observed is not None:
            prior['observed']=observed
        return prior

    def initialise_model(self,r_s_prior, m_s_prior, r_prior, logP_prior, t0_prior,mean_prior, init_b, init_u, reinit=False):
        '''
        Create chromatic model (only the wavelength-independent part)
        Thoughts:
        - at the moment we have to reinitialise to reuse this model.
        - I hope there's a better way, but pyMC3/theano doesn't like you reassigning variable names.
        - I have tried creating a subclass of 'chromatic_model' but it also updates the master model
         when you optimise the submodel.
        - copy.deepclone() doesn't seem to work either

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
        '''

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

        with pm.Model() as model:

            def init_prior(prior):
                if prior['dist'] == "Uniform":
                    return pm.Uniform(prior['name'], lower=prior['lower'], upper=prior['upper'],
                                      testval=np.array(prior['testval'])), prior['testval']
                if prior['dist'] == "Normal":
                    return pm.Normal(prior['name'], mu=prior['mu'], sigma=prior['sigma']), prior['mu']

            # Initialise (wavelength-indep) priors:
            self.r_s, self.init_r_s = init_prior(r_s_prior)
            self.m_s, self.init_m_s = init_prior(m_s_prior)
            self.logP, self.init_logP = init_prior(logP_prior)
            self.t0, self.init_t0 = init_prior(t0_prior)
            self.P = pm.Deterministic("period", pm.math.exp(self.logP))
            self.init_period = math.exp(self.init_logP)

            self.b = xo.distributions.ImpactParameter("b", ror=self.init_r, testval=np.array(init_b))

            # Set up a Keplerian orbit for the planets
            self.orbit = xo.orbits.KeplerianOrbit(period=self.P, t0=self.t0, b=self.b, r_star=self.r_s, m_star=self.m_s)
            # pm.Deterministic('a', self.orbit.a)
            # pm.Deterministic('i', self.orbit.incl * 180 / np.pi)
            # pm.Deterministic('a/r_s', self.orbit.a / self.orbit.r_star)

            self.model = model

    def initialise_model_staticwavelength(self):
        # Update chromatic model (with the wavelength-dependent part)

        with self.model as model:
            def init_prior(prior):
                if prior['dist'] == "Uniform":
                    return pm.Uniform(prior['name'], lower=prior['lower'], upper=prior['upper'],
                                      testval=np.array(prior['testval'])), prior['testval']
                if prior['dist'] == "Normal":
                    return pm.Normal(prior['name'], mu=prior['mu'], sigma=prior['sigma']), prior['mu']

            self.r, self.init_r = init_prior(self.r_prior)
            self.mean, self.init_mean = init_prior(self.mean_prior)

            # The Kipping (2013) parameterization for quadratic limb darkening paramters
            self.u_ld = xo.distributions.QuadLimbDark("u", testval=np.array(self.init_u))


    def reinitialise(self):
        self.initialise_model(self.r_s_prior, self.m_s_prior, self.r_prior, self.logP_prior, self.t0_prior, self.mean_prior, self.init_b, self.init_u, reinit=True)

    def set_model(self,model):
        self.model = model

    def optimise_model(self,plot=True):
        # use our predefined model to optimise the parameters
        with self.model as model:
            # Compute the model light curve using starry
            light_curves = xo.LimbDarkLightCurve(self.u_ld[0], self.u_ld[1]).get_light_curve(
                orbit=self.orbit, r=self.r, t=list(self.x)
            )
            light_curve = pm.math.sum(light_curves, axis=-1) + self.mean

            # Here we track the value of the model light curve for plotting purposes
            pm.Deterministic("light_curves", light_curves)

            # The likelihood function assuming known Gaussian uncertainty
            pm.Normal("obs", mu=light_curve, sd=self.yerr, observed=self.y)
            # Draw 50 samples from the prior distributions (gives an idea of how good your priors are)
            prior_checks = pm.sample_prior_predictive(samples=50, random_seed=RANDOM_SEED)

            start_time = ttime.time()
            # Fit for the maximum a posteriori parameters given the actual lightcurve
            map_soln = pmx.optimize(start=model.test_point)
            print("Optimising model took --- %s seconds ---" % (ttime.time() - start_time))

            self.result = map_soln
            self.priors = prior_checks

            if plot:
                start_time = ttime.time()
                # plot the priors vs posterior BEFORE sampling (only shows priors + best-fit model)
                self.plot_fit(prior_checks)
                print("Plotting took --- %s seconds ---" % (ttime.time() - start_time))

        return map_soln, model


    def run(self, r, optimisation="weighted_average",plot=True):
        # r is a rainbow object
        flux, flux_error, time, wavelength = rainbow_to_vector(r)

        if len(time) < 100:
            endpoints = 3
        else:
            endpoints = 50

        if optimisation == "simultaneous":
            datasets = OrderedDict([])
            count = 1
            for rf, re, rw in zip(flux[:6], flux_error[:6], wavelength[:6]):
                x = np.array(time - min(time))[~np.isnan(rf)]
                if len(x) > 0:
                    re = re[~np.isnan(rf)]
                    y = np.array(rf[~np.isnan(rf)])
                    y = y / np.nanmedian(y[-endpoints:])
                    y = y - np.nanmedian(y[-endpoints:])
                    yerr = np.array(re)
                    datasets["wavelength_" + str(count)] = (x, y, yerr, self.init_r, self.init_mean, self.init_u)
                    count += 1

            map_soln, model = self.optimise_model_sim(datasets)


        elif optimisation == "weighted_average":
            # weighted_lc = np.nanmedian(flux,axis=0)
            # weighted_err = np.nanmedian(flux_error,axis=0)
            start_time = ttime.time()
            weighted_lc, weighted_err = weighted_avg_lc(time, flux, flux_error, wavelength,
                                                        wavelength_range=[np.min(wavelength), np.max(wavelength)])
            time = time[~np.isnan(weighted_lc)]
            weighted_err = weighted_err[~np.isnan(weighted_lc)]
            weighted_lc = weighted_lc[~np.isnan(weighted_lc)]
            x = time - min(time)
            y = weighted_lc - np.nanmedian(weighted_lc[:endpoints])
            yerr = weighted_err
            x = np.array([i.to_value() for i in x])
            print("Weighted Average LC took --- %s seconds ---" % (ttime.time() - start_time))

            self.x = x
            self.y = y
            self.yerr = yerr

            plt.plot(x, y, 'k.')
            plt.show()
            plt.close()

            start_time = ttime.time()
            self.initialise_model_staticwavelength()
            print("Initialising r/u/mean took --- %s seconds ---" % (ttime.time() - start_time))

            map_soln, model = self.optimise_model(plot=plot)

        elif optimisation == "separate_wavelengths":
            firstrun = True
            xs,ys, yerrs, models,results = [],[],[],[],[]

            for rf, re, rw in zip(flux, flux_error, wavelength):
                print("Wavelength: ",rw)
                try:

                    x = np.array(time - min(time))
                    y = np.array(rf)
                    y = y / np.nanmedian(y[-endpoints:])
                    y = y - np.nanmedian(y[-endpoints:])
                    yerr = np.array(re)

                    x = x[~np.isnan(y)]
                    yerr = yerr[~np.isnan(y)]
                    y = y[~np.isnan(y)]

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

                except Exception as e:
                    print(e)

            self.x = xs
            self.y = ys
            self.yerr = yerrs
            self.model = models
            self.result = results

        else:
            print("Unrecognised optimisation type, current options are: weighted_average and separate_wavelengths")
            return None, None

        # model_copy = copy.deepcopy(self.model)
        # model_copy2 = theano.clone(self.model)

        # map_soln, model = optimise_model(self, x, y, yerr)

        return map_soln, model

    def optimise_model_sim(self,datasets):

        # THANKS MATHILDE/LIONEL!!

        with self.Model as model:

            for n, (name, (x, y, yerr, r_ratio, mean, ldc)) in enumerate(datasets.items()):
                if len(x) > 0:
                    # We define the per-instrument parameters in a submodel so that we don’t have to prefix the names manually
                    with pm.Model(name=name, model=model):
                        # The limb darkening #It’s the same filter in this case
                        u = xo.QuadLimbDark('u', testval=ldc)
                        star = xo.LimbDarkLightCurve(u)

                        # The radius ratio
                        ror = pm.Uniform('ror', lower=0.01, upper=0.5,
                                         testval=r_ratio)  # star.get_ror_from_approx_transit_depth(depth, b))
                        r_p = pm.Deterministic('r_p', ror * self.r_s)  # In solar radius
                        r = pm.Deterministic('r', r_p * 1 / R_sun)

                        # lightcurve mean
                        mean = pm.Normal("mean", mu=mean, sigma=0.1)

                        # starry light-curve
                        light_curves = star.get_light_curve(orbit=self.orbit, r=r_p, t=self.x)
                        transit = pm.Deterministic('transit', pm.math.sum(light_curves, axis=-1) + mean)

                        # Systematics and final model
                        #             w = pm.Flat('w',shape=len(X))
                        #             systematics = pm.Deterministic('systematics',w@X)
                        # residuals = pm.Deterministic(“residuals”, y - transit)
                        # systematics=X
                        mu = pm.Deterministic('mu', transit)  # +systematics)
                        # Likelihood function
                        pm.Normal('obs', mu=mu, sd=yerr, observed=y)
            # Maximum a posteriori
            # --------------------
            opt = pmx.optimize(start=model.test_point)
            opt = pmx.optimize(start=opt, vars=[ror, u, mean])
            # opt = pmx.optimize(start=opt, vars=[depth[1]])

        return opt, model

    def plot_fit(self,prior_checks=[]):
        # set up plot
        _,ax = plt.subplots(ncols=2,nrows=1,sharex=True,sharey=True,figsize=(12,6))
        ax[0].plot(self.x, self.y, ".k", ms=4, label="data")
        ax[1].plot(self.x, self.y, ".k", ms=4, label="data")

        # plot prior samples
        # ******************
        try:
            firstprior = True
            for period,r,t0,b,u_ld,mean in zip(prior_checks["period"], prior_checks["r"],prior_checks["t0"],prior_checks["b"],prior_checks["u"],prior_checks["mean"]):
                # for each prior sample extract the transit model and plot
                light_curve = generate_xo_orbit(period, t0, b, u_ld, r, self.x) + mean
                if firstprior:
                    ax[0].plot(self.x, light_curve, color="C1", lw=1, alpha=0.5, label="Prior Sample (n=50)")
                    firstprior = False
                else:
                    ax[0].plot(self.x, light_curve, color="C1", lw=1, alpha=0.5)
        except Exception as e:
            print(e)
        # ******************


        # plot initial parameter guess
        # ******************
        light_curve = generate_xo_orbit(self.init_period, self.init_t0, self.init_b, self.init_u, self.init_r, self.x) + self.init_mean
        ax[0].plot(self.x, light_curve, color="green", lw=1, alpha=0.5,linestyle='--',label="Initial Guess")
        # ******************


        # plot 50 chains sampled from the psterior (MCMC):
        # ******************
        try:
            firsttrace = True

            # plot the 16-84th posterior percentile region
            q16, q50, q84 = np.percentile(self.trace["light_curves"], [16, 50, 84], axis=(0))
            ax[1].fill_between(self.x, q16.flatten(), q84.flatten(),color="C1", alpha=0.2, label="Posterior (16-84th percentile)")

            # plot 50 individual posterior samples
            for i in np.random.randint(len(self.trace) * self.trace.nchains, size=50):
                if firsttrace:
                    # add the legend label to only the first prior line (to avoid 50 legend entries)
                    ax[1].plot(self.x, self.trace['light_curves'][i], color="C1", lw=1, alpha=0.3,
                             label='Posterior Sample (n=50)')
                    firsttrace = False
                else:
                    ax[1].plot(self.x, self.trace['light_curves'][i], color="C1", lw=1, alpha=0.3)
        except Exception as e:
            print(e)
        # ******************


        ax[0].errorbar(self.x, self.y, self.yerr, c='k', alpha=0.2)
        ax[1].errorbar(self.x, self.y, self.yerr, c='k', alpha=0.2)
        ax[1].plot(self.x, self.result["light_curves"], lw=1, label="model")
        ax[0].set_xlim(self.x.min(), self.x.max())
        ax[1].set_ylim(self.y.min()-0.01, self.y.max()+0.01)
        ax[0].set_ylabel("relative flux")
        ax[0].set_xlabel("time [days]")
        ax[1].set_xlabel("time [days]")
        ax[0].legend(loc="upper right",fontsize=10,facecolor='white', framealpha=0.6)
        ax[1].legend(loc="upper right",fontsize=10, facecolor='white', framealpha=0.6)
        ax[0].set_title("Data + Prior Models")
        ax[1].set_title("Data + Posterior Models")
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

def generate_xo_orbit(period,t0,b,u,r,x):
    orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b)
    light_curve = xo.LimbDarkLightCurve([u[0], u[1]]).get_light_curve(
        orbit=orbit, r=r, t=list(x)).eval()
    return light_curve

#
# def plot_model_and_recovered(trans_spect_file,model):
#     trans_spect = pd.read_csv(trans_spect_file)
#     trans_spect['depth'] = trans_spect['r']  # **2
#     trans_spect['uncertainty'] = 2 * trans_spect['r'] * trans_spect['r_err']
#
#     fig,ax = plt.subplots(nrows=2,figsize=(12,6),sharex=True)
#     ax[0].plot(model.table['wavelength']*u.Unit("um"),100*model.table['depth'],'k',markersize=10)
#     ax[0].plot(trans_spect['wavelength']*u.Unit("um"),100*trans_spect['depth'],'.',color='orange',markersize=10)
#     # plt.errorbar(trans_spect['wavelength'].values*u.Unit("um"),100*trans_spect['depth'],xerr=np.mean(halfbinwidths),yerr=100*trans_spect['uncertainty'],capsize=3,c='k',linestyle="None")
#     ax[0].errorbar(trans_spect['wavelength'].values*u.Unit("um"),100*trans_spect['depth'],yerr=100*trans_spect['uncertainty'],capsize=3,c='orange',linestyle="None")
#     ax[0].set_ylabel("(Rp/R*)^2 [%]")
#     plt.xlabel("Wavelength (microns)")
#
#     num_ld_coeffs = len(model.ld_coeffs[model.modemask==0][0])
#     for ld in range(num_ld_coeffs):
#         plot_kw = dict(alpha=0.75, linewidth=2, label="LD Coeff " + str(ld))
#         extract_coeff = [l[ld] for l in model.ld_coeffs[model.modemask==0]]
#         ax[1].plot(model.table['wavelength'][model.modemask==0], extract_coeff)
#     ax[1].set_ylabel("Limb-Darkening Coeff")
#     ax[1].plot(trans_spect['wavelength']*u.Unit("um"),trans_spect['u0'],'.',c='blue',markersize=10)
#     ax[1].errorbar(trans_spect['wavelength'].values*u.Unit("um"),trans_spect['u0'],yerr=trans_spect['u0_err'],c='blue',capsize=3,linestyle="None")
#     ax[1].plot(trans_spect['wavelength']*u.Unit("um"),trans_spect['u1'],'.',c='orange',markersize=10)
#     ax[1].errorbar(trans_spect['wavelength'].values*u.Unit("um"),trans_spect['u1'],yerr=trans_spect['u1_err'],c='orange',capsize=3,linestyle="None")
#
# def multiple_transit_recover(summary,time,flux,flux_error,wavelength,bintime,binwave):
#     mcmcresults = summary['mean']
#     init_mean = mcmcresults['mean']
#     init_t0 = mcmcresults['t0']
#     init_period = mcmcresults['period']
#     period_error = summary['sd']['period']
#     init_b = mcmcresults['b']
#     init_r = mcmcresults['r']
#     init_u = [mcmcresults['u[0]'], mcmcresults['u[1]']]
#     fixed_var = ['t0', 'period', 'b']
#     trans_spec = {'wavelength': [], 'r': [], 'r_err': [], 'u0': [], 'u1': [], 'u0_err': [], 'u1_err': [], 'mean': [],
#                   'mean_err': [], 't0': [], 'period': [], 'b': []}
#
#     for rf, re, rw in zip(flux, flux_error, wavelength):
#         try:
#             print(rw)
#             if len(time)<100:
#                 endpoints = 3
#             else:
#                 endpoints = 50
#
#             x = np.array(time - min(time))
#             y = np.array(rf)
#             y = y / np.nanmedian(y[-endpoints:])
#             y = y - np.nanmedian(y[-endpoints:])
#             yerr = np.array(re)
#
#             yerr = yerr[~np.isnan(y)]
#             y = y[~np.isnan(y)]
#
#             map_soln, model, [period, r, t0, b, u, mean] = fit_transit(x, y, yerr, init_r, init_t0, init_period, init_b,
#                                                                        init_mean, init_u, period_error, fixed_var)
#             plot_fit(x, y, yerr, map_soln)
#             trace = sample(map_soln, model)
#             summary = summarise(model, trace, fixed_var)
#             print(summary, type(summary))
#             trans_spec['wavelength'].append(rw.to_value())
#             trans_spec['r'].append(summary['mean']['r'])
#             trans_spec['u0'].append(summary['mean']['u[0]'])
#             trans_spec['u1'].append(summary['mean']['u[1]'])
#             trans_spec['mean'].append(summary['mean']['mean'])
#             trans_spec['t0'].append(init_t0)
#             trans_spec['b'].append(init_b)
#             trans_spec['period'].append(init_period)
#             trans_spec['r_err'].append(summary['sd'][0])
#             trans_spec['u0_err'].append(summary['sd'][1])
#             trans_spec['u1_err'].append(summary['sd'][2])
#             trans_spec['mean_err'].append(summary['sd'][3])
#             print("r=", summary['mean']['r'], ", mean=", summary['mean']['mean'], ", u=", summary['mean']['u[0]'],
#                   summary['mean']['u[1]'])
#
#             print(trans_spec)
#             pd_ts = pd.DataFrame(trans_spec)
#             pd_ts.to_csv("transmission_spectrum_bt_"+str(bintime) +"_bw_"+str(binwave)+".csv", index=False)
#         except Exception as e:
#             print(e)
#
#     return trans_spec
#
# def single_transit_recover(time,weighted_lc,weighted_err,model_ld):
#     # set initial parameter estimates:
#     # dur = 0.1
#     init_t0 = 0.1
#     init_b = 0.44
#     init_r = 0.14
#     init_mean = 0.0
#     init_u = [1.3, -0.5]
#     # aR = semi_major_axis(init_period, 1, 1)
#     # est_period = (aR * dur * np.pi) / np.sqrt((1 + init_r) ** 2 - init_b ** 2)
#     # print(est_period)
#     init_period = 2  # 4.055259
#     period_error = 1  # 0.000009
#
#     # set up x, y, yerr vectors:
#     if len(time)<100:
#         endpoints = 3
#     else:
#         endpoints = 50
#     x = time - min(time)
#     y = weighted_lc - np.nanmedian(weighted_lc[-endpoints:])
#     yerr = weighted_err  # medflux*meddflux
#     x = np.array([i.to_value() for i in x])
#     print(np.shape(x), np.shape(y), np.shape(yerr))
#     print(type(x), type(y), type(yerr))
#
#     # fit MCMC transit
#     map_soln, model, [period, r, t0, b_ip, u_ld, mean] = fit_transit(x, y, yerr, init_r, init_t0, init_period, init_b,
#                                                                      init_mean, init_u, period_error)
#     # plot the fit
#     plot_fit(x, y, yerr, map_soln)
#     # sample the posterior
#     trace = sample(map_soln, model)
#     # plot corner plot (posterior distribution for each variable)
#     cornerplot(model, trace, period, r, t0, b_ip, u_ld, mean)
#     # plot summary (mean, std etc.)
#     summary = summarise(model, trace)
#     print(summary)
#     mcmcresults = summary['mean']
#
#     plt.figure(figsize=(8, 6), dpi=300)
#     ax = plt.gca()
#
#     num_ld_coeffs = len(model_ld.ld_coeffs[model_ld.modemask == 0][0])
#     # print(num_ld_coeffs, len(self.ld_coeffs[~np.isnan(self.ld_coeffs)][0]))
#
#     for ld in range(num_ld_coeffs):
#         plot_kw = dict(alpha=0.75, linewidth=2, label="LD Coeff " + str(ld))
#         plot_kw.update([])
#         extract_coeff = [l[ld] for l in model_ld.ld_coeffs[model_ld.modemask == 0]]
#         plt.plot(model_ld.table['wavelength'][model_ld.modemask == 0], extract_coeff)
#
#     plt.title(model_ld.ld_eqn + " Limb-Darkening")
#     plt.xlabel("Wavelength (micron)")
#     plt.ylabel("Limb-Darkening Coeff")
#
#     plt.axhline(mcmcresults['u[1]'], color='orange', label='u[1]')
#     plt.axhline(mcmcresults['u[0]'], color='blue', label='u[0]')
#
#     plt.legend()
#     plt.show()
#     plt.close()
#
#     return summary
#
# def main():
#     ## Import Patricio's multi-wavelength transit model:
#     x = pickle.load(open('data_challenge_spectra_v01.pickle', 'rb'))
#     x.keys()
#
#     # Load in planetary spectrum classes from ZBT defined[here](https: // github.com / ers - transit / ers - data - checkpoint - showcase / blob / main / features / playing - around -with-patricio - signals_Catriona_edits.ipynb)
#     # lets load a model
#     planet = x['WASP39b_NIRSpec']
#     planet_params = x['WASP39b_parameters']
#     print(planet_params)
#
#     wavelength = planet['wl']
#     transmission = planet['transmission']
#     table = Table(dict(wavelength=planet['wl'], depth=np.sqrt(planet['transmission'])), meta=planet_params)
#
#     # set up a new model spectrum
#     model = PlanetarySpectrumModel(table=table, label='injected model')
#
#     plt.plot(wavelength[1:], [t - s for s, t in zip(wavelength, wavelength[1:])])
#     plt.xlabel("Wavelength (microns)")
#     plt.ylabel("Wavelength Bin Spacing (microns)")
#     plt.show()
#     plt.close()
#
#     #  define where the zenodo LD files are stored (needed for ExoTiC)
#     dirsen = '/Users/catrionamurray/Documents/Postdoc/CUBoulder/exotic-ld_data'
#     #  define some stellar parameters
#     star_params = {"M_H": -0.03, "Teff": 5326.6, "logg": 4.38933}
#     mode = "NIRSpec_Prism"  # "NIRCam_F322W2" #
#     ld_eqn = 'quadratic'
#     p_params={}
#     # p_params =  {"t0": 0,"per": 2, "a": 10,"inc": 75, "ecc": 0, "w": 0}
#     #  calculate the LD coeffs for a series of wavelengths and transit depths
#     model_ld = generate_spectrum_ld(wavelength, np.sqrt(transmission), star_params, planet_params, dirsen, mode=mode,
#                                     ld_eqn=ld_eqn, ld_model='1D', plot_model=True)
#
#     # return synthetic rainbow + rainbow with injected transit
#     r, i = inject_spectrum(model_ld, snr=1000, dt=bintime, res=50, planet_params=p_params)
#
#     # b_withouttransit = r.bin(
#     #     dw=binwave * u.micron, dt=bintime * u.minute
#     # )
#     # b_withouttransit.imshow()
#     b_withtransit = i.bin(
#         dw=binwave * u.micron, dt= bintime * u.minute
#     )
#     # b_withtransit.imshow()
#
#     # ax = b_withtransit.bin(R=5).plot(plotkw=dict(alpha=0.1, markeredgecolor='none', linewidth=0))
#     # b_withtransit.bin(R=5, dt=10 * u.minute).plot(ax=ax)
#
#     flux, flux_error, time, wavelength = rainbow_to_vector(b_withtransit)
#
#     # Compute x^2 + y^2 across a 2D grid
#     x, y = np.meshgrid(time.to_value(), wavelength.to_value())
#     z = flux
#     x = x[~np.isnan(z)]
#     y = y[~np.isnan(z)]
#     z = z[~np.isnan(z)]
#
#     # alt_imshow(x, y, z, xlabel='Time (d)', ylabel='Wavelength (microns)', zlabel='Flux', ylog=False).display()
#
#     weighted_lc, weighted_err = weighted_avg_lc(time, flux, flux_error, wavelength,
#                                                 wavelength_range=[np.min(wavelength), np.max(wavelength)])
#     time = time[~np.isnan(weighted_lc)]
#     weighted_err = weighted_err[~np.isnan(weighted_lc)]
#     weighted_lc = weighted_lc[~np.isnan(weighted_lc)]
#
#     summary = single_transit_recover(time,weighted_lc,weighted_err,model_ld)
#     trans_spect = multiple_transit_recover(summary,time,flux,flux_error,wavelength,bintime,binwave)
