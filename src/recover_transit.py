from .imports import *
# from .pymc import *
import arviz as az
import pymc3 as pm
import pymc3_ext as pmx
# import pymc as pm
# import pymc_ext as pmx
import exoplanet as xo
import corner
# import theano
# import theano.tensor as tt
xo.utils.docs_setup()
# plot the samples from the gp posterior with samples and shading
# from pymc3.gp.util import plot_gp_dist

warnings.simplefilter(action="ignore", category=FutureWarning)
# %config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")

print(f"Running on PyMC v{pm.__version__}")
# print(f"Running on PyMC4 v{pm4.__version__}")
print(f"Running on ArviZ v{az.__version__}")
print(f"Running on Exoplanet v{xo.__version__}")


def fit_transit(x, y, yerr, init_r, init_t0, init_period, init_b, init_mean, init_u, period_error, fixed_var=[]):
    """
        Use PyMC3 and Exoplanet codes to fit a transit

        Parameters
        ----------
        x : np.array
            Orbital period (days).
        y : np.array
            Stellar mass (Solar masses).
        init_r : float
            Initial guess for planet-to-star radius ratio.
        init_t0 : float
            Initial guess for transit epoch.
        init_period : float
            Initial guess for orbital period.
        init_b : float
            Initial guess for impact parameter.
        init_mean : float
            Initial guess for the mean out-of-transit flux
        init_u : float
            Initial guess for the limb-darkening parameters.
        fixed_var : list (optional)
            List of variables that should remain constant in this fitting process.

        Returns
        ----------
        map_soln :
            The fit for the maximum a posteriori parameters given the simulated dataset.
        model : pm.Model
            The PyMC3 fitted model.
        [period, r, t0, b, u_ld, mean] : list
            List of distributions for each of the given params.

        """

    with pm.Model() as model:

        # The baseline flux
        if 'mean' in fixed_var:
            mean = pm.Normal('mean', mu=init_mean, sigma=0.1, observed=init_mean)
        else:
            mean = pm.Normal("mean", mu=init_mean, sigma=0.1)

        # The time of a reference transit for each planet
        if 't0' in fixed_var:
            t0 = pm.Normal("t0", mu=init_t0, sigma=1.0, observed=init_t0)
        else:
            t0 = pm.Normal("t0", mu=init_t0, sigma=1.0)

        # The log period; also tracking the period itself
        #         constrain the period as we only have 1 transit
        if 'period' in fixed_var:
            logP = pm.Normal("logP", mu=np.log(init_period), sigma=period_error, observed=np.log(init_period))
        else:
            logP = pm.Normal("logP", mu=np.log(init_period), sigma=period_error)
        period = pm.Deterministic("period", pm.math.exp(logP))

        # The Kipping (2013) parameterization for quadratic limb darkening paramters
        if 'u' in fixed_var:
            u_ld = xo.distributions.QuadLimbDark("u", testval=np.array(init_u), observed=np.array(init_u))
        else:
            u_ld = xo.distributions.QuadLimbDark("u", testval=np.array(init_u))

        #         r = pm.Uniform(
        #             "r", lower=init_r-0.00138, upper=init_r+0.001, testval=np.array(init_r)
        #         )
        if 'r' in fixed_var:
            r = pm.Uniform("r", lower=0.01, upper=0.5, testval=np.array(init_r), observed=np.array(init_r))
        else:
            r = pm.Uniform("r", lower=0.01, upper=0.5, testval=np.array(init_r))

        if 'b' in fixed_var:
            b = xo.distributions.ImpactParameter("b", ror=r, testval=np.array(init_b), observed=np.array(init_b))
        else:
            b = xo.distributions.ImpactParameter("b", ror=r, testval=np.array(init_b))

        # Set up a Keplerian orbit for the planets
        orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b)

        # Compute the model light curve using starry
        light_curves = xo.LimbDarkLightCurve(u_ld[0], u_ld[1]).get_light_curve(
            orbit=orbit, r=r, t=list(x)
        )
        light_curve = pm.math.sum(light_curves, axis=-1) + mean

        # Here we track the value of the model light curve for plotting
        # purposes
        pm.Deterministic("light_curves", light_curves)

        # ******************************************************************* #
        # On the folowing lines, we simulate the dataset that we will fit     #
        #                                                                     #
        # NOTE: if you are fitting real data, you shouldn't include this line #
        #       because you already have data!                                #
        # ******************************************************************* #
        #     y = pmx.eval_in_model(light_curve)
        #     y += yerr * np.random.randn(len(y))
        # ******************************************************************* #
        # End of fake data creation; you want to include the following lines  #
        # ******************************************************************* #

        # The likelihood function assuming known Gaussian uncertainty
        pm.Normal("obs", mu=light_curve, sd=yerr, observed=y)

        # Fit for the maximum a posteriori parameters given the simuated
        # dataset
        map_soln = pmx.optimize(start=model.test_point)

        return map_soln, model, [period, r, t0, b, u_ld, mean]


def plot_fit(x, y, yerr, map_soln,trace=[]):
    plt.plot(x, y, ".k", ms=4, label="data")
    plt.plot(x, map_soln["light_curves"], lw=1)

    # plot 50 chains from MCMC:
    if len(trace)>0:
        for i in np.random.randint(len(trace) * trace.nchains, size=50):
            plt.plot(x, trace['light_curves'][i], color="C1", lw=1, alpha=0.2)
    #         # Set up a Keplerian orbit for the planets
    #         orbit = xo.orbits.KeplerianOrbit(period=trace['period'][i], t0=trace['t0'][i], b=trace['b'][i])
    # #         # Compute the model light curve using starry
    #         light_curves = xo.LimbDarkLightCurve(trace['u[0]'][i], trace['u[1]']).get_light_curve(
    #             orbit=orbit, r=trace['r'], t=list(x)
    #         )
    #         light_curve = pm.math.sum(light_curves, axis=-1) + trace['mean'][i]
    #         plt.plot(x, light_curve, color="C1", lw=1, alpha=0.3)


    plt.errorbar(x, y, yerr, c='k', alpha=0.2)
    plt.xlim(x.min(), x.max())
    plt.ylabel("relative flux")
    plt.xlabel("time [days]")
    plt.legend(fontsize=10)
    _ = plt.title("map model")
    plt.show()
    plt.close()

def sample(map_soln, model, tune_steps=4000,draws=8000,cores=12,chains=4,target_accept=0.9):
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
    :return:
        The trace object from the MCMC sampling
    """
    np.random.seed(42)
    with model:
        trace = pmx.sample(
            tune=tune_steps,
            draws=draws,
            start=map_soln,
            cores=cores,
            chains=chains,
            target_accept=target_accept,
        )
        return trace


def summarise(model, trace, fixed_var=[], sim_wavelengths=False):
    # all_varnames = ["period", "r", "t0", "b", "u", "mean"]
    # nonfixed_varnames = [a for a in all_varnames if a not in fixed_var]

    with model:
        summary = az.summary(
            trace, round_to=7, fmt='wide'
        )
    return summary


def cornerplot(model, trace, period, r, t0, b, u, mean, fixed_var=[]):
    ''' Generate and display a corner plot from the posterior distributions
    Parameters
    ----------
        model : PyMC3 model
            The PyMC3 model
        trace : Arviz trace
            The samples taken from the posterior distributions
        period : float
            Orbital period
        r : float
            Radius ratio
        t0 : float
            Transit epoch
        b : ImpactParameter
            Impact Parameter
        u : list or np.array
            Quadratic limb-darkening coeffs
        mean : float
            Mean of out-of-transit lightcurve
        fixed_var : list
            (optional, default=[])
            List of variables to keep fixed (not optimised)
    '''
    all_varnames = ["period", "r", "t0", "b", "u", "mean"]
    nonfixed_varnames = [a for a in all_varnames if a not in fixed_var]
    all_vars = [period, r, t0, b, u, mean]
    to_del = []
    if 'period' in fixed_var:
        to_del.append(0)
    if 'r' in fixed_var:
        to_del.append(1)
    if 't0' in fixed_var:
        to_del.append(2)
    if 'b' in fixed_var:
        to_del.append(3)
    if 'u' in fixed_var:
        to_del.append(4)
    if 'mean' in fixed_var:
        to_del.append(5)
    print(np.delete(all_vars, to_del))
    nonfixed_vars = list(np.delete(all_vars, to_del))

    truth = dict(
        zip(
            nonfixed_varnames,
            pmx.eval_in_model(nonfixed_vars, model.test_point, model=model),
        )
    )
    _ = corner.corner(
        trace,
        var_names=nonfixed_varnames,
        truths=truth,
    )