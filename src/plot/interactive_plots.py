import altair as alt
from ..imports import *
alt.data_transformers.disable_max_rows()


# Convert this grid to columnar data expected by Altair
def alt_imshow(x, y, z, xlabel, ylabel, zlabel, ylog=True):
    source = pd.DataFrame({'x': x.ravel(),
                           'y': y.ravel(),
                           'z': z.ravel()})

    brush = alt.selection(type='interval', encodings=['y'])

    if ylog == False:
        spectrum = alt.Chart(source, width=280, height=230).mark_rect(clip=True, width=10, height=15).encode(
            x=alt.X('x:Q', scale=alt.Scale(zero=False), title=xlabel),
            y=alt.Y('y:Q', scale=alt.Scale(zero=False, domain=[round(np.min(y), 2), round(np.max(y), 2)]),
                    title=ylabel),
            color=alt.Color('z:Q', title=zlabel, scale=alt.Scale(zero=False, domain=[np.min(z), np.max(z)])),
            tooltip=['x', 'y', 'z']
        )
    else:
        spectrum = alt.Chart(source, width=280, height=230).mark_rect(clip=True, width=200 / len(x),
                                                                      height=300 / len(y)).encode(
            x=alt.X('x:Q', scale=alt.Scale(zero=False), title=xlabel),
            y=alt.Y('y:Q', scale=alt.Scale(zero=False, base=10, type='log'), title=ylabel),
            color=alt.Color('z:Q', title=zlabel, scale=alt.Scale(zero=False, domain=[np.min(z), np.max(z)])),
            tooltip=['x', 'y', 'z']
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

    lightcurve = alt.Chart(source, width=280, height=230, title="Mean Lightcurve for Wavelength Range").mark_point(
        filled=True, clip=True, size=20, color='black').encode(
        x=alt.X('x:Q',
                scale=alt.Scale(zero=False), title=xlabel),
        y=alt.Y('mean(z):Q',
                scale=alt.Scale(zero=False, domain=[0.96, 1.01]), title='Mean ' + zlabel)
    ).transform_filter(
        brush
    )

    return spectrum_int | lightcurve
