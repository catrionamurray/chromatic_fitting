import altair as alt
from ..utils import *
from ..imports import *
import warnings
alt.data_transformers.disable_max_rows()

# Convert this grid to columnar data expected by Altair
def alt_imshow(r, quantity='Flux', ylog=False, timeformat='h'):
    """ Display interactive spectrum plot for chromatic Rainbow with a wavelength-averaged 2D quantity
    defined by the user. The user can interact with the 3D spectrum to choose the wavelength range over
    which the average is calculated.

    Parameters
    ----------
        r : Rainbow object
            chromatic Rainbow object to plot
        quantity : str
            (optional, default='Flux)
            The quantity on the z-axis
        ylog : boolean
            (optional, default=False)
            Boolean for whether to take log10 of the y-axis data
        timeformat : str
            (optional, default='hours')
            The time format to use (seconds, minutes, hours, days etc.)
    """

    # preset the x and y axes as Time (in units defined by the user) and Wavelength
    xlabel = f"Time ({timeformat})"
    ylabel = "Wavelength (microns)"

    # allow the user to plot flux or uncertainty
    if quantity.lower() == "flux":
        z = "Flux"
    elif quantity.lower() == "uncertainty":
        z = "Flux Error"
    elif quantity.lower() == "error":
        z = "Flux Error"
    elif quantity.lower() == "flux_error":
        z = "Flux Error"
    else:
        # if the quantity is not one of the predefined values:
        warnings.warn("Unrecognised Quantity!")
        return

    # convert rainbow object to pandas dataframe
    source = rainbow_to_df(r, timeformat=timeformat)[[xlabel, ylabel, z]]

    # if there are >10,000 data points Altair will be very laggy/slow. This is probably unbinned, therefore
    # encourage the user to bin the Rainbow before calling this function in future/
    if len(source) > 10000:
        warnings.warn(">10,000 data points - interactive plot will lag! You should try binning the Rainbow first!")
        if not ylog:
            warnings.warn("It looks like you might have unbinned data - you may want to use ylog=True!")

    # The unbinned Rainbow is sometimes in log scale, therefore plotting will be ugly with uniform axis spacing
    # ylog tells the function to bin the y-axis data
    if ylog:
        source[ylabel] = np.log10(source[ylabel])
        source = source.rename(columns={ylabel: f"log({ylabel})"})
        ylabel = f"log({ylabel})"

    # Add interactive part
    brush = alt.selection(type='interval', encodings=['y'])

    # Define the 3D spectrum plot
    spectrum = alt.Chart(source, width=280, height=230).mark_rect(clip=False, width=280 / len(r.timelike['time']),
                                                                  height=230 / len(r.wavelike['wavelength'])).encode(
        x=alt.X(f'{xlabel}:Q', scale=alt.Scale(zero=False, nice=False, domain=[np.min(source[xlabel]),
                                                                               np.max(source[xlabel])])),
        y=alt.Y(f'{ylabel}:Q', scale=alt.Scale(zero=False, nice=False, domain=[np.max(source[ylabel]),
                                                                               np.min(source[ylabel])])),
        fill=alt.Color(f'{z}:Q', scale=alt.Scale(scheme='viridis', zero=False, domain=[np.min(source[z]),
                                                                                       np.max(source[z])])),
        tooltip=[f'{xlabel}', f'{ylabel}', f'{z}']
    )

    # gray out the background with selection
    background = spectrum.encode(
        color=alt.value('#ddd')
    ).add_selection(brush)

    # highlights on the transformed data
    highlight = spectrum.transform_filter(brush)

    # Layer the various plotting parts
    spectrum_int = alt.layer(
        background,
        highlight,
        data=source
    )

    # Add the 2D averaged lightcurve (or uncertainty)
    lightcurve = alt.Chart(source, width=280, height=230, title=f"Mean {z} for Wavelength Range").mark_point(
        filled=True, size=20, color='black').encode(
        x=alt.X(f'{xlabel}:Q',
                scale=alt.Scale(zero=False, nice=False,
                                domain=[np.min(source[xlabel])-(0.02*np.abs(np.min(source[xlabel]))),
                                        1.02*np.max(source[xlabel])])),
        y=alt.Y(f'mean({z}):Q',
                scale=alt.Scale(zero=False, domain=[np.mean(source[z]).min() - 0.01, np.mean(source[z]).max() + 0.01]),
                title='Mean ' + z)
    ).transform_filter(
        brush
    )

    # display the interactive Altair plot
    (spectrum_int | lightcurve).display()
