"""
Visual theme for the Sub-topic 3 Streamlit page: deep-space background with a
twinkling star field, background stars that flare, and a hero star erupting
flare loops. Presentation only - pure CSS/SVG, no effect on predictions.
Animations are switched off for users who prefer reduced motion.
"""
from __future__ import annotations

import streamlit as st

# Palette (kept in one place so the plot in ui.py can reuse it)
SPACE_BG = "#04050d"                 # page background (near-black blue)
PANEL_BG = "rgba(18, 21, 43, 0.72)"  # translucent panels (forms, cards)
STAR_WHITE = "#e8ebff"               # main text
MUTED_TEXT = "#9aa3d4"               # secondary text, axis ticks
FLARE_YELLOW = "#ffd36e"             # highlights, metric values, plot point
FLARE_ORANGE = "#ff8a2b"             # buttons, interval bars
FLARE_RED = "#ff4e1f"                # gradient ends, third flare loop
GRID_LINE = "#2a2f5a"                # plot grid and axis lines

# All page styling + the fixed background elements. Doubled braces {{ }} are
# literal CSS braces inside this f-string; single braces insert the palette above.
# Background stars use repeated radial-gradient "dots" (no images needed).
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&display=swap');

/* ---------- deep-space background with a static star field ---------- */
[data-testid="stApp"] {{
  background-color: {SPACE_BG};
  background-image:
    radial-gradient(ellipse at 15% 10%, rgba(88, 60, 180, 0.22), transparent 45%),
    radial-gradient(ellipse at 85% 85%, rgba(255, 110, 40, 0.14), transparent 50%),
    radial-gradient(ellipse at 70% 30%, rgba(40, 90, 200, 0.12), transparent 40%),
    radial-gradient(1px 1px at 25px 35px, #fff, transparent),
    radial-gradient(1px 1px at 120px 80px, rgba(255,255,255,.75), transparent),
    radial-gradient(1.5px 1.5px at 190px 150px, #fff, transparent),
    radial-gradient(1px 1px at 60px 170px, rgba(255,220,180,.8), transparent),
    radial-gradient(1px 1px at 230px 40px, rgba(190,210,255,.8), transparent),
    radial-gradient(1px 1px at 160px 220px, rgba(255,255,255,.6), transparent);
  background-size: 100% 100%, 100% 100%, 100% 100%,
                   260px 260px, 260px 260px, 260px 260px, 260px 260px, 260px 260px, 260px 260px;
  background-attachment: fixed;
}}
[data-testid="stHeader"] {{ background: transparent; }}

/* ---------- twinkling layer + flaring background stars (fixed, behind content) ---------- */
.space-twinkle {{
  position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-image:
    radial-gradient(1.5px 1.5px at 50px 90px, #fff, transparent),
    radial-gradient(1px 1px at 300px 200px, #cfe0ff, transparent),
    radial-gradient(2px 2px at 420px 60px, #fff, transparent),
    radial-gradient(1px 1px at 150px 330px, #ffe2c0, transparent);
  background-size: 470px 400px;
  animation: twinkle 5s ease-in-out infinite alternate;
}}
@keyframes twinkle {{ from {{ opacity: .25; }} to {{ opacity: 1; }} }}

.bg-star {{
  position: fixed; pointer-events: none; z-index: 0; border-radius: 50%;
  background: radial-gradient(circle, #fff 0%, {FLARE_YELLOW} 35%, rgba(255,138,43,0) 70%);
}}
.bg-star::after {{                         /* the flare burst */
  content: ""; position: absolute; inset: -220%; border-radius: 50%;
  background: radial-gradient(circle, rgba(255,211,110,.55) 0%, rgba(255,78,31,.25) 35%, transparent 65%);
  opacity: 0; animation: bg-flare 9s ease-out infinite;
}}
.bg-star.s1 {{ width: 10px; height: 10px; top: 14%; right: 8%; }}
.bg-star.s2 {{ width: 7px;  height: 7px;  top: 62%; left: 5%; }}
.bg-star.s2::after {{ animation-delay: 3s; }}
.bg-star.s3 {{ width: 6px;  height: 6px;  bottom: 9%; right: 18%; }}
.bg-star.s3::after {{ animation-delay: 6s; }}
@keyframes bg-flare {{
  0%, 70% {{ opacity: 0; transform: scale(.3); }}
  76%     {{ opacity: 1; transform: scale(1); }}
  100%    {{ opacity: 0; transform: scale(1.6); }}
}}

.shooting-star {{
  position: fixed; top: 18%; left: -10%; width: 140px; height: 2px; z-index: 0; pointer-events: none;
  background: linear-gradient(90deg, rgba(255,255,255,0), #fff);
  transform: rotate(18deg); opacity: 0;
  animation: shoot 14s linear infinite 4s;
}}
@keyframes shoot {{
  0%   {{ opacity: 0; transform: translate(0, 0) rotate(18deg); }}
  2%   {{ opacity: 1; }}
  8%   {{ opacity: 0; transform: translate(70vw, 23vw) rotate(18deg); }}
  100% {{ opacity: 0; transform: translate(70vw, 23vw) rotate(18deg); }}
}}

/* page content above the background layers */
[data-testid="stMain"], .block-container {{ position: relative; z-index: 1; }}

/* ---------- hero: star with erupting flares ---------- */
.flare-hero {{
  display: flex; align-items: center; gap: 1.4rem; margin: .2rem 0 1.2rem;
  padding: 1.1rem 1.3rem; border-radius: 18px;
  background: linear-gradient(135deg, rgba(30,22,60,.75), rgba(12,14,32,.75));
  border: 1px solid rgba(255,138,43,.28);
  box-shadow: 0 0 40px rgba(255,110,40,.12), inset 0 0 30px rgba(110,80,220,.10);
}}
.flare-hero > svg {{ flex: 0 0 auto; width: 165px; height: 165px; overflow: visible; }}
.flare-hero h1 {{
  font-family: 'Space Grotesk', system-ui, sans-serif; font-weight: 700;
  font-size: clamp(1.6rem, 3.6vw, 2.35rem); line-height: 1.12; margin: 0 0 .35rem; padding: 0;
  background: linear-gradient(90deg, #ffffff, {FLARE_YELLOW} 55%, {FLARE_ORANGE});
  -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.flare-hero h1 .stHeaderActionElements,
.flare-hero h1 [data-testid="stHeaderActionElements"] {{ display: none !important; }}  /* Streamlit's anchor link */
.flare-hero p {{ margin: 0; color: {MUTED_TEXT}; font-size: .95rem; line-height: 1.45; }}
.flare-hero .tag {{
  display: inline-block; margin-bottom: .45rem; padding: .12rem .6rem; border-radius: 999px;
  font-size: .72rem; letter-spacing: .08em; text-transform: uppercase;
  color: {FLARE_YELLOW}; border: 1px solid rgba(255,211,110,.4); background: rgba(255,138,43,.08);
}}
@media (max-width: 640px) {{
  .flare-hero {{ flex-direction: column; text-align: center; }}
  .flare-hero > svg {{ width: 120px; height: 120px; }}
}}

.hero-glow   {{ transform-origin: 100px 100px; animation: glow 3.2s ease-in-out infinite; }}
.hero-core   {{ transform-origin: 100px 100px; animation: boil 6s ease-in-out infinite; }}
.hero-rays   {{ transform-origin: 100px 100px; animation: spin 40s linear infinite; }}
.flare-loop  {{ fill: none; stroke-linecap: round; stroke-dasharray: 160; stroke-dashoffset: 160;
                filter: drop-shadow(0 0 4px {FLARE_ORANGE}) drop-shadow(0 0 10px rgba(255,170,60,.7));
                animation: erupt 4.5s ease-out infinite; }}
.flare-loop.l2 {{ animation-delay: 1.5s; }}
.flare-loop.l3 {{ animation-delay: 3s; }}
.flare-burst {{ transform-origin: 100px 100px; animation: burst 4.5s ease-out infinite; }}
@keyframes glow  {{ 0%,100% {{ opacity: .55; transform: scale(.95); }} 50% {{ opacity: 1; transform: scale(1.06); }} }}
@keyframes boil  {{ 0%,100% {{ transform: scale(1); }} 50% {{ transform: scale(1.03); }} }}
@keyframes spin  {{ to {{ transform: rotate(360deg); }} }}
@keyframes erupt {{
  0%   {{ stroke-dashoffset: 160; opacity: 0; }}
  10%  {{ opacity: 1; }}
  55%  {{ stroke-dashoffset: 0; opacity: 1; }}
  100% {{ stroke-dashoffset: -160; opacity: 0; }}
}}
@keyframes burst {{ 0%,60% {{ opacity: 0; transform: scale(.6); }} 68% {{ opacity: .9; }} 100% {{ opacity: 0; transform: scale(1.35); }} }}

/* ---------- widgets ---------- */
h2, h3, [data-testid="stHeading"] * {{ font-family: 'Space Grotesk', system-ui, sans-serif; }}
[data-testid="stForm"], [data-testid="stExpander"] details, [data-testid="stFileUploader"] section {{
  background: {PANEL_BG}; border: 1px solid rgba(140,150,255,.18); border-radius: 14px;
  backdrop-filter: blur(4px);
}}
[data-testid="stMetric"] {{
  background: {PANEL_BG}; border: 1px solid rgba(255,138,43,.32); border-radius: 14px;
  padding: .8rem 1rem; box-shadow: 0 0 22px rgba(255,110,40,.12);
}}
[data-testid="stMetricValue"] {{ color: {FLARE_YELLOW}; text-shadow: 0 0 14px rgba(255,170,60,.45); }}
[data-testid="stMetricLabel"] * {{ color: {MUTED_TEXT}; }}
[data-testid="stTabs"] button[role="tab"] p {{ font-family: 'Space Grotesk', system-ui, sans-serif; font-size: 1rem; }}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {{ color: {FLARE_YELLOW}; }}
[data-baseweb="tab-highlight"] {{ background: linear-gradient(90deg, {FLARE_YELLOW}, {FLARE_RED}); }}
[data-testid="stDataFrame"] {{ border: 1px solid rgba(140,150,255,.18); border-radius: 10px; }}

button[kind^="primary"], [data-testid^="stBaseButton-primary"] {{
  background: linear-gradient(90deg, {FLARE_ORANGE}, {FLARE_RED}) !important;
  border: none !important; color: #fff !important; font-weight: 600;
  box-shadow: 0 0 18px rgba(255,110,40,.45); transition: box-shadow .2s, transform .2s;
}}
button[kind^="primary"]:hover, [data-testid^="stBaseButton-primary"]:hover {{
  box-shadow: 0 0 30px rgba(255,170,60,.75); transform: translateY(-1px);
}}
[data-testid="stDownloadButton"] button {{
  border: 1px solid rgba(255,211,110,.5); color: {FLARE_YELLOW};
}}

@media (prefers-reduced-motion: reduce) {{
  .space-twinkle, .bg-star::after, .shooting-star, .hero-glow, .hero-core,
  .hero-rays, .flare-loop, .flare-burst {{ animation: none !important; }}
  .flare-loop {{ stroke-dashoffset: 0; }}
}}
</style>
<div class="space-twinkle"></div>
<div class="bg-star s1"></div><div class="bg-star s2"></div><div class="bg-star s3"></div>
<div class="shooting-star"></div>
"""

# Hero star: glow, rotating rays, an orange-white core and three flare loops
_HERO_SVG = f"""
<svg viewBox="0 0 200 200" role="img" aria-label="Star releasing flares">
  <defs>
    <radialGradient id="s3core" cx="45%" cy="42%" r="60%">
      <stop offset="0%" stop-color="#ffffff"/>
      <stop offset="35%" stop-color="{FLARE_YELLOW}"/>
      <stop offset="75%" stop-color="{FLARE_ORANGE}"/>
      <stop offset="100%" stop-color="{FLARE_RED}"/>
    </radialGradient>
    <radialGradient id="s3glow">
      <stop offset="0%" stop-color="rgba(255,170,60,0.75)"/>
      <stop offset="100%" stop-color="rgba(255,90,30,0)"/>
    </radialGradient>
    <radialGradient id="s3burst">
      <stop offset="0%" stop-color="rgba(255,255,255,0.9)"/>
      <stop offset="40%" stop-color="rgba(255,211,110,0.5)"/>
      <stop offset="100%" stop-color="rgba(255,78,31,0)"/>
    </radialGradient>
  </defs>
  <!-- pulsing outer glow (corona) -->
  <circle class="hero-glow" cx="100" cy="100" r="92" fill="url(#s3glow)"/>
  <!-- eight slowly rotating light rays -->
  <g class="hero-rays" stroke="{FLARE_YELLOW}" stroke-opacity=".35" stroke-width="2">
    <line x1="100" y1="8" x2="100" y2="38"/><line x1="100" y1="162" x2="100" y2="192"/>
    <line x1="8" y1="100" x2="38" y2="100"/><line x1="162" y1="100" x2="192" y2="100"/>
    <line x1="35" y1="35" x2="55" y2="55"/><line x1="145" y1="145" x2="165" y2="165"/>
    <line x1="165" y1="35" x2="145" y2="55"/><line x1="35" y1="165" x2="55" y2="145"/>
  </g>
  <!-- bright flash where a flare erupts -->
  <circle class="flare-burst" cx="138" cy="66" r="34" fill="url(#s3burst)"/>
  <!-- the star's surface -->
  <circle class="hero-core" cx="100" cy="100" r="46" fill="url(#s3core)"/>
  <!-- three flare (coronal) loops, drawn and erased in turn by the 'erupt' animation -->
  <path class="flare-loop l1" d="M118 62 C 140 10, 196 30, 146 84" stroke="{FLARE_YELLOW}" stroke-width="7"/>
  <path class="flare-loop l2" d="M58 128 C 8 150, 22 196, 84 146" stroke="{FLARE_ORANGE}" stroke-width="6"/>
  <path class="flare-loop l3" d="M140 128 C 190 140, 186 190, 124 146" stroke="{FLARE_RED}" stroke-width="5"/>
</svg>
"""


def apply_space_theme() -> None:
    """Inject the background, animations and widget styling (call once per page run)."""
    # unsafe_allow_html lets Streamlit render our own <style> and <div> tags.
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, tag: str = "Sub-topic 3") -> None:
    """Page header: animated flaring star next to the title.

    Args:
        title: heading text (HTML entities such as &amp; are allowed).
        subtitle: one-line description under the heading.
        tag: small label shown above the heading.
    """
    # Layout: [ star SVG | tag / title / subtitle ], styled by .flare-hero in _CSS.
    st.markdown(
        f'<div class="flare-hero">{_HERO_SVG}<div><span class="tag">{tag}</span>'
        f"<h1>{title}</h1><p>{subtitle}</p></div></div>",
        unsafe_allow_html=True,
    )
