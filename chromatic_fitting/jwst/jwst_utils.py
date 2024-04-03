from ..imports import *
def get_spot_contrast(wavelengths, star_teff=3180, spot_teff=2900, spot_radii=[0.29, 0.16, 0.09],
                      logg=4.97, metallicity=0.0, visualize=False):
    phoenix = np.load("phoenix_photons_metallicity=0.0_R=100.npy", allow_pickle=True)
    pl = PHOENIXLibrary()
    S_star = pl.get_spectrum(temperature=star_teff, logg=logg, metallicity=metallicity,
                             wavelength=wavelengths, visualize=visualize)
    S_spot = pl.get_spectrum(temperature=spot_teff, logg=logg, metallicity=metallicity,
                             wavelength=wavelengths, visualize=visualize)

    f = 0
    for sr in spot_radii:
        f += sr ** 2
    print(f"Spot covering fraction = {f}")

    S_total = (f * S_spot[1]) + ((1 - f) * S_star[1])

    if visualize:
        plt.plot(S_star[0], S_star[1], label=f"Star (T={star_teff}K)")
        plt.plot(S_spot[0], S_spot[1], label=f"Spot (T={spot_teff}K)")
        plt.plot(wavelengths, S_total, 'k', alpha=0.3, label=f"Mixed Spectrum (f={f})")
        plt.plot(wavelengths, S_total, 'k.', label="Mixed Spectrum")

        plt.ylabel("Surface Flux [Photons / (s * m**2 * nm)]")
        plt.xlabel("Wavelength [micron]")
        plt.legend()

    contrast = (S_star[1] - S_spot[1]) / S_star[1]

    if visualize:
        plt.figure()
        plt.plot(wavelengths, contrast)
        plt.ylabel("Spot Contrast")
        plt.xlabel("Wavelength [micron]")

    return contrast