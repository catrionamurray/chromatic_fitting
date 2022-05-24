import altair as alt
from ..utils import *
from ..imports import *
import warnings
alt.data_transformers.disable_max_rows()


# Convert this grid to columnar data expected by Altair
def alt_imshow(r, quantity='Flux', ylog=False):
    xlabel = "Time (h)"
    ylabel = "Wavelength (microns)"

    if quantity.lower() == "flux":
        z = "Flux"  # rflux
    elif quantity.lower() == "uncertainty":
        z = "Flux Error"  # rfluxe
    elif quantity.lower() == "error":
        z = "Flux Error"  # rfluxe
    elif quantity.lower() == "flux_error":
        z = "Flux Error"  # rfluxe
    else:
        print("Unrecognised Quantity!")
        return

    source = rainbow_to_df(r)[[xlabel, ylabel, z]]
    if len(source) > 10000:
        warnings.warn(">10,000 data points - interactive plot will lag! You should try binning first!")
        if not ylog:
            warnings.warn("It looks like you might have unbinned data - you may want to use ylog=True!")

    if ylog:
        source[ylabel] = np.log10(source[ylabel])
        source = source.rename(columns={ylabel: f"log({ylabel})"})
        ylabel = f"log({ylabel})"

    brush = alt.selection(type='interval', encodings=['y'])

    spectrum = alt.Chart(source, width=280, height=230).mark_rect(clip=True, width=250 / len(r.timelike['time']),
                                                                  height=350 / len(r.wavelike['wavelength'])).encode(
        x=alt.X(f'{xlabel}:Q', scale=alt.Scale(zero=False)),
        y=alt.Y(f'{ylabel}:Q', scale=alt.Scale(zero=False, domain=[round(np.max(source[ylabel]), 2),
                                                                   round(np.min(source[ylabel]), 2)])),
        color=alt.Color(f'{z}:Q', scale=alt.Scale(scheme='viridis', zero=False,
                                                  domain=[np.min(source[z]), np.max(source[z])])),
        tooltip=[f'{xlabel}', f'{ylabel}', f'{z}']
    )

    # gray background with selection
    background = spectrum.encode(
        color=alt.value('#ddd')
    ).add_selection(brush)

    # highlights on the transformed data
    highlight = spectrum.transform_filter(brush)

    spectrum_int = alt.layer(
        background,
        highlight,
        data=source
    )

    lightcurve = alt.Chart(source, width=280, height=230, title=f"Mean {z} for Wavelength Range").mark_point(
        filled=True, clip=True, size=20, color='black').encode(
        x=alt.X(f'{xlabel}:Q',
                scale=alt.Scale(zero=False)),
        y=alt.Y(f'mean({z}):Q',
                scale=alt.Scale(zero=False, domain=[np.mean(source[z]).min() - 0.01, np.mean(source[z]).max() + 0.01]),
                title='Mean ' + z)
    ).transform_filter(
        brush
    )

    return spectrum_int | lightcurve
