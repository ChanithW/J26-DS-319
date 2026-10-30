"""Component 1 · Flare Detection & Energy Prediction (Streamlit app).

Run (from the sub01 folder):  streamlit run app/Home.py
Needs  sub01/models/  with best_v2.keras, prep_settings.json, app_settings.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pipeline as pl

MODEL_DIR = Path(__file__).resolve().parent.parent / 'models'

st.set_page_config(page_title='Flare Detection & Energy Prediction', page_icon='🌟', layout='wide')


@st.cache_resource
def assets():
    return pl.load_assets(str(MODEL_DIR))


model, prep, levels = assets()

st.title('🌟 Flare Detection & Energy Prediction')
st.info('Upload a star\'s light curve and give its parameters. A two-head 1D-CNN predicts the probability '
        'that each spike is a flare, and for each flare predicts its energy (erg) and type.')

# ------------------------------------------------------------------ inputs
st.subheader('1 · Light curve')
up = st.file_uploader('CSV with columns time, flux, flux_err (TESS 2-minute data)', type='csv')

st.subheader('2 · Star parameters')
c1, c2, c3 = st.columns(3)
teff = c1.number_input('Teff (K)', 2000.0, 10000.0, 3500.0, 50.0)
radius = c2.number_input('Radius (R☉)', 0.05, 5.0, 0.50, 0.01, format='%.3f')
contam = c3.number_input('Contamination ratio', 0.0, 10.0, 0.0, 0.001, format='%.4f')

if not st.button('🔍 Detect flares & predict energy', type='primary', disabled=up is None):
    st.stop()

try:
    df = pl.read_light_curve(up)
except ValueError as e:
    st.error(str(e))
    st.stop()

with st.spinner('Running the model…'):
    res, info = pl.run(df, teff, radius, contam, model, prep, levels)

# ------------------------------------------------------------------ confirmation bar
st.success(f"✔ {info['n_windows']:,} windows checked · {info['n_points']:,} points over {info['days']:.1f} days · "
           f"star type {info['star_type']} · inputs: Teff {teff:.0f} K, radius {radius} R☉, "
           f"contamination {contam}")
for w in info['warnings']:
    st.warning(w)

conf = res[res.level == 'confirmed']
m1, m2, m3, m4 = st.columns(4)
m1.metric('Confirmed flares', len(conf))
m2.metric('Candidate flares', int((res.level == 'candidate').sum()))
m3.metric('Low-SNR flares (SNR < 5)', int(res.low_snr.sum()))
if len(conf):
    top = conf.loc[conf.log10_energy.idxmax()]
    m4.metric('Strongest flare', f'{top.energy_erg:.1e} erg', top.type, delta_color='off')
else:
    m4.metric('Strongest flare', '–')

tc = st.columns(4)
for col, ty in zip(tc, pl.TYPES):
    col.metric(ty, int((conf.type == ty).sum()))

# ------------------------------------------------------------------ chart
st.subheader('Light curve · flares coloured by type')
t, f = df.time.values, df.flux.values / np.median(df.flux.values)
fig, ax = plt.subplots(figsize=(15, 4))
gaps = np.where(np.diff(t) > 10 / 1440)[0] + 1                 # break the line at data gaps
for seg_t, seg_f in zip(np.split(t, gaps), np.split(f, gaps)):
    ax.plot(seg_t, seg_f, lw=0.5, color='#5B7FA6')
for r in res.itertuples():
    y = f[np.argmin(np.abs(t - r.peak_time))]
    ax.scatter(r.peak_time, y, s=70, zorder=3, linewidth=2, edgecolor=pl.COLOURS[r.type],
               color=pl.COLOURS[r.type] if r.level == 'confirmed' else 'none')
for ty in pl.TYPES:
    ax.scatter([], [], color=pl.COLOURS[ty], label=ty)
ax.scatter([], [], color='none', edgecolor='grey', linewidth=2, label='hollow = candidate')
ax.legend(loc='upper right', fontsize=9, ncol=5)
ax.set_xlabel('Time (BTJD)'); ax.set_ylabel('Relative flux')
st.pyplot(fig)

# ------------------------------------------------------------------ table
st.subheader('Detected flares')
if len(res) == 0:
    st.write('No flares found at the confirmed or candidate level.')
else:
    show = res.copy()
    show['energy_erg'] = show.energy_erg.map(lambda e: f'{e:.1e}')
    st.dataframe(show, hide_index=True)

    pick = st.selectbox('Zoom on a flare', show.flare_id)
    r = res[res.flare_id == pick].iloc[0]
    m = np.abs(t - r.peak_time) < 0.15
    fz, az = plt.subplots(figsize=(8, 3))
    az.plot(t[m], f[m], '.', ms=3, color='#5B7FA6')
    az.axvline(r.peak_time, color=pl.COLOURS[r.type])
    az.set_title(f'Flare {pick} · {r.level} · {r.type} · {r.energy_erg:.1e} erg')
    az.set_xlabel('Time (BTJD)'); az.set_ylabel('Relative flux')
    st.pyplot(fz)

# ------------------------------------------------------------------ downloads / hand-offs
st.subheader('Downloads')
d1, d2, d3 = st.columns(3)
d1.download_button('⬇ Flare table (CSV)', res.to_csv(index=False), 'flares.csv', 'text/csv')
d2.download_button('→ Flare cut-outs for Component 2 (CME)',
                   pl.cutouts_for_c2(res, df, star_id=up.name.rsplit('.', 1)[0]).to_csv(index=False),
                   'flare_cutouts_c2.csv', 'text/csv')
d3.download_button('→ Strongest flare for Component 4 (habitability)',
                   json.dumps(pl.summary_for_c4(res, teff, radius), indent=2),
                   'strongest_flare_c4.json', 'application/json')

with st.expander('How the flare levels are decided'):
    st.write(f"**Confirmed:** probability ≥ {levels['confirmed']['threshold']} and {levels['confirmed']['rule']}. "
             f"**Candidate:** probability ≥ {levels['candidate']['threshold']} and {levels['candidate']['rule']}. "
             'Both were tuned on the full light curves of validation stars (about 1 false alarm per sector '
             'at the confirmed level). Peaks of one complex flare are joined when the flux stays above '
             'normal between them. Energy types: Solar-like < 10³², Large 10³²–10³³, '
             'Superflare 10³³–10³⁴, Extreme superflare > 10³⁴ erg.')
