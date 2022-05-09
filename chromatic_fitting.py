from src.inject_spectrum import *
# from src.plot.interactive_plots import *
from src.weighted_average_lc import *
from src.recover_transit import *
from src.utils import *
bintime=5 # minutes
binwave=0.1 # microns

def plot_model_and_recovered(trans_spect_file,model):
    trans_spect = pd.read_csv(trans_spect_file)
    trans_spect['depth'] = trans_spect['r']  # **2
    trans_spect['uncertainty'] = 2 * trans_spect['r'] * trans_spect['r_err']

    fig,ax = plt.subplots(nrows=2,figsize=(12,6),sharex=True)
    ax[0].plot(model.table['wavelength']*u.Unit("um"),100*model.table['depth'],'k',markersize=10)
    ax[0].plot(trans_spect['wavelength']*u.Unit("um"),100*trans_spect['depth'],'.',color='orange',markersize=10)
    # plt.errorbar(trans_spect['wavelength'].values*u.Unit("um"),100*trans_spect['depth'],xerr=np.mean(halfbinwidths),yerr=100*trans_spect['uncertainty'],capsize=3,c='k',linestyle="None")
    ax[0].errorbar(trans_spect['wavelength'].values*u.Unit("um"),100*trans_spect['depth'],yerr=100*trans_spect['uncertainty'],capsize=3,c='orange',linestyle="None")
    ax[0].set_ylabel("(Rp/R*)^2 [%]")
    plt.xlabel("Wavelength (microns)")

    num_ld_coeffs = len(model.ld_coeffs[model.modemask==0][0])
    for ld in range(num_ld_coeffs):
        plot_kw = dict(alpha=0.75, linewidth=2, label="LD Coeff " + str(ld))
        extract_coeff = [l[ld] for l in model.ld_coeffs[model.modemask==0]]
        ax[1].plot(model.table['wavelength'][model.modemask==0], extract_coeff)
    ax[1].set_ylabel("Limb-Darkening Coeff")
    ax[1].plot(trans_spect['wavelength']*u.Unit("um"),trans_spect['u0'],'.',c='blue',markersize=10)
    ax[1].errorbar(trans_spect['wavelength'].values*u.Unit("um"),trans_spect['u0'],yerr=trans_spect['u0_err'],c='blue',capsize=3,linestyle="None")
    ax[1].plot(trans_spect['wavelength']*u.Unit("um"),trans_spect['u1'],'.',c='orange',markersize=10)
    ax[1].errorbar(trans_spect['wavelength'].values*u.Unit("um"),trans_spect['u1'],yerr=trans_spect['u1_err'],c='orange',capsize=3,linestyle="None")

def multiple_transit_recover(summary,time,flux,flux_error,wavelength,bintime,binwave):
    mcmcresults = summary['mean']
    init_mean = mcmcresults['mean']
    init_t0 = mcmcresults['t0']
    init_period = mcmcresults['period']
    period_error = summary['sd']['period']
    init_b = mcmcresults['b']
    init_r = mcmcresults['r']
    init_u = [mcmcresults['u[0]'], mcmcresults['u[1]']]
    fixed_var = ['t0', 'period', 'b']
    trans_spec = {'wavelength': [], 'r': [], 'r_err': [], 'u0': [], 'u1': [], 'u0_err': [], 'u1_err': [], 'mean': [],
                  'mean_err': [], 't0': [], 'period': [], 'b': []}

    for rf, re, rw in zip(flux, flux_error, wavelength):
        try:
            print(rw)
            if len(time)<100:
                endpoints = 3
            else:
                endpoints = 50

            x = np.array(time - min(time))
            y = np.array(rf)
            y = y / np.nanmedian(y[-endpoints:])
            y = y - np.nanmedian(y[-endpoints:])
            yerr = np.array(re)

            yerr = yerr[~np.isnan(y)]
            y = y[~np.isnan(y)]

            map_soln, model, [period, r, t0, b, u, mean] = fit_transit(x, y, yerr, init_r, init_t0, init_period, init_b,
                                                                       init_mean, init_u, period_error, fixed_var)
            plot_fit(x, y, yerr, map_soln)
            trace = sample(map_soln, model)
            summary = summarise(model, trace, fixed_var)
            print(summary, type(summary))
            trans_spec['wavelength'].append(rw.to_value())
            trans_spec['r'].append(summary['mean']['r'])
            trans_spec['u0'].append(summary['mean']['u[0]'])
            trans_spec['u1'].append(summary['mean']['u[1]'])
            trans_spec['mean'].append(summary['mean']['mean'])
            trans_spec['t0'].append(init_t0)
            trans_spec['b'].append(init_b)
            trans_spec['period'].append(init_period)
            trans_spec['r_err'].append(summary['sd'][0])
            trans_spec['u0_err'].append(summary['sd'][1])
            trans_spec['u1_err'].append(summary['sd'][2])
            trans_spec['mean_err'].append(summary['sd'][3])
            print("r=", summary['mean']['r'], ", mean=", summary['mean']['mean'], ", u=", summary['mean']['u[0]'],
                  summary['mean']['u[1]'])

            print(trans_spec)
            pd_ts = pd.DataFrame(trans_spec)
            pd_ts.to_csv("transmission_spectrum_bt_"+str(bintime) +"_bw_"+str(binwave)+".csv", index=False)
        except Exception as e:
            print(e)

    return trans_spec

def single_transit_recover(time,weighted_lc,weighted_err,model_ld):
    # set initial parameter estimates:
    # dur = 0.1
    init_t0 = 0.1
    init_b = 0.44
    init_r = 0.14
    init_mean = 0.0
    init_u = [1.3, -0.5]
    # aR = semi_major_axis(init_period, 1, 1)
    # est_period = (aR * dur * np.pi) / np.sqrt((1 + init_r) ** 2 - init_b ** 2)
    # print(est_period)
    init_period = 2  # 4.055259
    period_error = 1  # 0.000009

    # set up x, y, yerr vectors:
    if len(time)<100:
        endpoints = 3
    else:
        endpoints = 50
    x = time - min(time)
    y = weighted_lc - np.nanmedian(weighted_lc[-endpoints:])
    yerr = weighted_err  # medflux*meddflux
    x = np.array([i.to_value() for i in x])
    print(np.shape(x), np.shape(y), np.shape(yerr))
    print(type(x), type(y), type(yerr))

    # fit MCMC transit
    map_soln, model, [period, r, t0, b_ip, u_ld, mean] = fit_transit(x, y, yerr, init_r, init_t0, init_period, init_b,
                                                                     init_mean, init_u, period_error)
    # plot the fit
    plot_fit(x, y, yerr, map_soln)
    # sample the posterior
    trace = sample(map_soln, model)
    # plot corner plot (posterior distribution for each variable)
    cornerplot(model, trace, period, r, t0, b_ip, u_ld, mean)
    # plot summary (mean, std etc.)
    summary = summarise(model, trace)
    print(summary)
    mcmcresults = summary['mean']

    plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.gca()

    num_ld_coeffs = len(model_ld.ld_coeffs[model_ld.modemask == 0][0])
    # print(num_ld_coeffs, len(self.ld_coeffs[~np.isnan(self.ld_coeffs)][0]))

    for ld in range(num_ld_coeffs):
        plot_kw = dict(alpha=0.75, linewidth=2, label="LD Coeff " + str(ld))
        plot_kw.update([])
        extract_coeff = [l[ld] for l in model_ld.ld_coeffs[model_ld.modemask == 0]]
        plt.plot(model_ld.table['wavelength'][model_ld.modemask == 0], extract_coeff)

    plt.title(model_ld.ld_eqn + " Limb-Darkening")
    plt.xlabel("Wavelength (micron)")
    plt.ylabel("Limb-Darkening Coeff")

    plt.axhline(mcmcresults['u[1]'], color='orange', label='u[1]')
    plt.axhline(mcmcresults['u[0]'], color='blue', label='u[0]')

    plt.legend()
    plt.show()
    plt.close()

    return summary

## Import Patricio's multi-wavelength transit model:
x = pickle.load(open('data_challenge_spectra_v01.pickle', 'rb'))
x.keys()

# Load in planetary spectrum classes from ZBT defined[here](https: // github.com / ers - transit / ers - data - checkpoint - showcase / blob / main / features / playing - around -with-patricio - signals_Catriona_edits.ipynb)
# lets load a model
planet = x['WASP39b_NIRSpec']
planet_params = x['WASP39b_parameters']
print(planet_params)

wavelength = planet['wl']
transmission = planet['transmission']
table = Table(dict(wavelength=planet['wl'], depth=np.sqrt(planet['transmission'])), meta=planet_params)

# set up a new model spectrum
model = PlanetarySpectrumModel(table=table, label='injected model')

plt.plot(wavelength[1:], [t - s for s, t in zip(wavelength, wavelength[1:])])
plt.xlabel("Wavelength (microns)")
plt.ylabel("Wavelength Bin Spacing (microns)")
plt.show()
plt.close()

#  define where the zenodo LD files are stored (needed for ExoTiC)
dirsen = '/Users/catrionamurray/Documents/Postdoc/CUBoulder/exotic-ld_data'
#  define some stellar parameters
star_params = {"M_H": -0.03, "Teff": 5326.6, "logg": 4.38933}
mode = "NIRSpec_Prism"  # "NIRCam_F322W2" #
ld_eqn = 'quadratic'
p_params={}
# p_params =  {"t0": 0,"per": 2, "a": 10,"inc": 75, "ecc": 0, "w": 0}
#  calculate the LD coeffs for a series of wavelengths and transit depths
model_ld = generate_spectrum_ld(wavelength, np.sqrt(transmission), star_params, planet_params, dirsen, mode=mode,
                                ld_eqn=ld_eqn, ld_model='1D', plot_model=True)

# return synthetic rainbow + rainbow with injected transit
r, i = inject_spectrum(model_ld, snr=1000, dt=bintime, res=50, planet_params=p_params)

# b_withouttransit = r.bin(
#     dw=binwave * u.micron, dt=bintime * u.minute
# )
# b_withouttransit.imshow()
b_withtransit = i.bin(
    dw=binwave * u.micron, dt= bintime * u.minute
)
# b_withtransit.imshow()

# ax = b_withtransit.bin(R=5).plot(plotkw=dict(alpha=0.1, markeredgecolor='none', linewidth=0))
# b_withtransit.bin(R=5, dt=10 * u.minute).plot(ax=ax)

flux, flux_error, time, wavelength = rainbow_to_vector(b_withtransit)

# Compute x^2 + y^2 across a 2D grid
x, y = np.meshgrid(time.to_value(), wavelength.to_value())
z = flux
x = x[~np.isnan(z)]
y = y[~np.isnan(z)]
z = z[~np.isnan(z)]

# alt_imshow(x, y, z, xlabel='Time (d)', ylabel='Wavelength (microns)', zlabel='Flux', ylog=False).display()

weighted_lc, weighted_err = weighted_avg_lc(time, flux, flux_error, wavelength,
                                            wavelength_range=[np.min(wavelength), np.max(wavelength)])
time = time[~np.isnan(weighted_lc)]
weighted_err = weighted_err[~np.isnan(weighted_lc)]
weighted_lc = weighted_lc[~np.isnan(weighted_lc)]

summary = single_transit_recover(time,weighted_lc,weighted_err,model_ld)
trans_spect = multiple_transit_recover(summary,time,flux,flux_error,wavelength,bintime,binwave)
