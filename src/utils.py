def rainbow_to_vector(r):
    # r: rainbow object
    rflux = r.fluxlike['flux']         # flux (MJy/sr)        : [n_wavelengths x n_integrations]
    rfluxe = r.fluxlike['uncertainty'] # flux error (MJy/sr)  : [n_wavelengths x n_integrations]
    rtime = r.timelike['time']         # time (BJD_TDB, days) : [n_integrations]
    rwavel = r.wavelike['wavelength']  # wavelength (microns) : [n_wavelengths]
    return rflux,rfluxe,rtime,rwavel