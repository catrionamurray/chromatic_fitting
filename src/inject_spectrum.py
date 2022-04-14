from .imports import *
from .spectrum import *
from chromatic import SimulatedRainbow

class star:
    def __init__(self, logg, M_H, Teff):
        self.logg = logg
        self.M_H = M_H
        self.Teff = Teff

class synthetic_planet(PlanetarySpectrumModel):

    def add_stellar_params(self,star_params):
        self.stellar = star(logg=star_params['logg'],M_H=star_params['M_H'],Teff=star_params['Teff'])

    # def generate_planet_params(ldc=[0.65, 0.28],limb_dark="quadratic", t0=0.0, p=1.0, R=1.0, M=1.0, i=0.5*math.pi, ecc=0, w=0):
    #     a = semi_major_axis(p,M,R)
    #
    #     planet_params = {
    #                 "t0": t0,
    #                 "per": p,
    #                 "a": a,
    #                 "inc": i,
    #                 "ecc": ecc,
    #                 "w": w,
    #                 "limb_dark": limb_dark,
    #                 "u": ldc,
    #             }

    def generate_ld_coeffs(self,mode,dirsen,ld_eqn,ld_model):
        # include an option to choose the linear, quadratic, 4-param or non-linear options
        # (NOTE: There is a current issue open in ExoTiC to add a selection criteria so that only the desired coefficients are returned.)
        ld_coeffs,mask = [],[]

        # For ExoTiC we need a range of wavelengths - create bins around each value (not ideal)
        for w in range(len(self.table["wavelength"])):
            w_now = self.table["wavelength"][w]

            if w > 0 and w < len(self.table["wavelength"]) - 1:
                w_last = self.table["wavelength"][w - 1]
                w_next = self.table["wavelength"][w + 1]

                wsdata = [w_now - (0.5 * (w_now - w_last)),
                          w_now + (0.5 * (w_next - w_now))]
            elif w == 0:
                w_next = self.table["wavelength"][w + 1]
                wsdata = [w_now, w_now + (0.5 * (w_next - w_now))]

            elif w == len(self.table["wavelength"]) - 1:
                w_last = self.table["wavelength"][w - 1]
                wsdata = [w_now - (0.5 * (w_now - w_last)), w_now]

            print("Wavelength range: ", wsdata)

            # * * * * * * * *
            # Use ExoTiC-LD to calculate wavelength-dependent LD coeffs
            result = limb_dark_fit(mode, np.array(wsdata) * 10000, self.stellar.M_H, self.stellar.Teff, self.stellar.logg, dirsen, ld_model=ld_model)
            # * * * * * * * *

            # If all zeros are returned then ignore (happens when we're outside the wavelength range defined by 'mode')
            if np.all(np.array(result) == 0.0):
                print("Outside " + mode + " wavelength range!\n")
                mask.append(1)
                ld_coeffs.append(np.nan)
            else:
                if ld_eqn == "linear":
                    print("Using linear LD equation\n")
                    ld_coeffs.append(result[0])
                    mask.append(0)

                elif ld_eqn == "quadratic":
                    print("Using quadratic LD equation\n")
                    ld_coeffs.append(result[9:11])
                    mask.append(0)

                elif ld_eqn == "nonlinear":
                    print("Using non-linear LD equation\n")
                    ld_coeffs.append(result[1:5])
                    mask.append(0)

                elif ld_eqn == "threeparam":
                    print("Using three-paramter LD equation\n")
                    ld_coeffs.append(result[5:9])
                    mask.append(0)

                else:
                    print("No valid LD equation method chosen!\n")

        self.modemask = mask
        self.ld_coeffs = ld_coeffs

def semi_major_axis(per,M,R):
    per_s = per * 24 * 60 * 60 * u.s
    R_sun = R * 696340000 * u.m
    M_sun = M * 1.989e30 * u.kg

    a = ((per_s)**2 * G * M_sun/ (4 * math.pi**2))**(1./3) # in units of m
    a_radii = a / R_sun

    return a_radii

def main(wavelength, depth, star_params,planet_params,dirsen,snr=100,dt=1,res=50,mode='NIRCam_F322W2',ld_eqn='quadratic',ld_model='1D',plot_model=True):
    # M_H,teff,logg in star_params
    table = Table(dict(wavelength=wavelength, depth=depth), meta=planet_params)
    model = synthetic_planet(table=table, label='injected model')
    model.add_stellar_params(star_params)
    model.generate_ld_coeffs(mode,dirsen,ld_eqn,ld_model)

    # plot model provided
    if plot_model==True:
        ax = model.plot()
        plt.legend(frameon=False)
        plt.show()
        plt.close()

    modemask = model.modemask

    r = SimulatedRainbow(
        signal_to_noise=snr,
        dt=dt * u.minute,
        wavelength=model.table["wavelength"][modemask==0] * u.micron,
        R=res
    )
    i = r.inject_transit(
        planet_radius=np.array(model.table['depth'][modemask==0]),
        planet_params= {"limb_dark": ld_eqn, 'u':model.ld_coeffs[modemask==0]} #planet_params
    )

    return r,i

