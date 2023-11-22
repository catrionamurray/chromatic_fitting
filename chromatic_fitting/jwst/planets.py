from ..imports import *
def toi3884():
    star_params = dict(
        # ydeg=30,  # degree of the map
        udeg=2,  # degree of the limb darkening
        inc=89.81,  # inclination in degrees
        amp=1.0,  # amplitude (a value prop. to luminosity)
        r=0.302 * u.R_sun,  #  radius in R_sun
        m=0.298 * u.M_sun,  # mass in M_sun
        prot=4.22,  # rotational period (d)
        u=[0.10, 0.1],  # limb darkening coefficients
    #     y=A_y,  # the spherical harmonic coefficients
    )

    planet_params = dict(
        porb=4.5445828, # orbital period of planet (d)
        mp=32.59 * u.M_earth, # mass of planet in M_earth
        rp=6.43 * u.R_earth, # radius of planet in R_earth
        ecc=0.06, # eccentricity
        omega=(-1.96*u.radian).to_value('degree'), # longitude of ascending node in degrees,
        epoch=2459556.51669
    )

    return star_params, planet_params


def kepler51():
    star_params = dict(
        udeg=2,  # degree of the limb darkening
        inc=80,  # inclination in degrees
        amp=1.0,  # amplitude (a value prop. to luminosity)
        r=0.881 * u.R_sun,  #  radius in R_sun
        m=0.985 * u.M_sun,  # mass in M_sun
        prot=8.222,  # rotational period (d)
        u=[0.247, 0.267],  # limb darkening coefficients
    #     y=A_y,  # the spherical harmonic coefficients
    )

    planet_params = dict(
        porb=130.1845, # orbital period of planet (d)
        mp=5.70 * u.M_earth, # mass of planet in M_earth
        rp=9.46 * u.R_earth, # radius of planet in R_earth
        ecc=0.0, # eccentricity
        omega=-12.6, # longitude of ascending node in degrees,
        epoch=2457778.75336,
        inc=89.91,
    )

    return star_params, planet_params