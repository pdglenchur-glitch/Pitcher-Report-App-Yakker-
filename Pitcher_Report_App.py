import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import io
import re
from datetime import datetime


# Streamlit Page UI

st.set_page_config(layout="wide", page_title="Pitching Report Generator")
st.title("⚾ Pitching Report Builder – Streamlit Edition")
st.write("Upload multiple CSVs → choose pitcher → filter by date → generate full scouting report.")


#Extract date from filename

def extract_date(filename):
    match = re.search(r"(\d{1,2})_(\d{1,2})_(\d{2,4})", filename)
    if match:
        month, day, year = match.groups()
        if len(year) == 2:
            year = "20" + year
        return datetime(int(year), int(month), int(day))
    return None


#File Upload

uploaded_files = st.file_uploader(
    "Upload one or more CSV files (multi-session tracking supported)",
    type=["csv"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.stop()

df_list = []
file_dates = []

for file in uploaded_files:
    temp = pd.read_csv(file)
    session_date = extract_date(file.name)
    temp["SourceFile"] = file.name
    temp["SessionDate"] = session_date
    df_list.append(temp)
    file_dates.append(session_date)

df = pd.concat(df_list, ignore_index=True)
st.success(f"Loaded {len(uploaded_files)} files with {len(df):,} total pitches.")


#Season Date Range

valid_dates = [d for d in file_dates if d is not None]
season_range = (
    f"{min(valid_dates):%Y-%m-%d} → {max(valid_dates):%Y-%m-%d}"
    if valid_dates else "Unknown Date Range"
)


#Date Filter

available_dates = sorted(df["SessionDate"].dropna().unique())
selected_dates = st.multiselect(
    "Filter by session date(s):",
    options=available_dates,
    default=available_dates
)
df = df[df["SessionDate"].isin(selected_dates)]


#Pitcher Selection

pitchers = sorted(df["Pitcher"].dropna().unique())
selected_pitcher = st.selectbox("Select a pitcher:", pitchers)

p = df[df["Pitcher"] == selected_pitcher].copy()
if p.empty:
    st.warning("No data for this pitcher.")
    st.stop()


p = p[p["TaggedPitchType"].notna()]
p = p[p["RelSpeed"].notna()]
p = p.sort_values("PitchNo")
p["PitchNo"] = range(1, len(p) + 1)


# Batted ball data

p["IsBIP"] = p["ExitSpeed"].notna()
p["GB"] = p["IsBIP"] & p["Angle"].between(-90, 10, inclusive="both")
p["LD"] = p["IsBIP"] & p["Angle"].between(10, 25, inclusive="both")
p["FB"] = p["IsBIP"] & p["Angle"].between(25, 90, inclusive="both")

total_bip = p["IsBIP"].sum()
overall_gb_pct = p["GB"].sum() / total_bip * 100 if total_bip > 0 else np.nan
overall_ld_pct = p["LD"].sum() / total_bip * 100 if total_bip > 0 else np.nan
overall_fb_pct = p["FB"].sum() / total_bip * 100 if total_bip > 0 else np.nan


# Summary Stats

total_pitches = len(p)
max_velo = p["RelSpeed"].max()


# PZR Calculation

margin_in = 2.85
margin_ft = margin_in / 12
sx_left, sx_right = -8.5 - margin_in, 8.5 + margin_in
sz_bot, sz_top = 1.5 - margin_ft, 3.5 + margin_ft

def compute_pzr(sub):
    px = sub["PlateLocSide"] * 12
    pz = sub["PlateLocHeight"]
    return ((px >= sx_left) & (px <= sx_right) &
            (pz >= sz_bot) & (pz <= sz_top)).mean() * 100

overall_pzr = compute_pzr(p)


#Summary Table

summary_labels = ["Pitches", "GB%", "LD%", "FB%", "Max Velo", "PZR%"]
summary_values = [
    f"{total_pitches}",
    f"{overall_gb_pct:.1f}%" if not np.isnan(overall_gb_pct) else "–",
    f"{overall_ld_pct:.1f}%" if not np.isnan(overall_ld_pct) else "–",
    f"{overall_fb_pct:.1f}%" if not np.isnan(overall_fb_pct) else "–",
    f"{max_velo:.1f}",
    f"{overall_pzr:.1f}%"
]


#Color Scheme

pitch_types = p["TaggedPitchType"].unique().tolist()
palette = sns.color_palette("husl", len(pitch_types))
colors = dict(zip(pitch_types, palette))


#Figure Layout

fig = plt.figure(figsize=(17, 13), dpi=150)
gs = gridspec.GridSpec(
    7, 4,
    height_ratios=[0.8, 1.2, 3.0, 2.2, 2.2, 0.45, 2.8],
    hspace=1.05,
    wspace=1.05
)


#Header

ax_header = fig.add_subplot(gs[0,:])
ax_header.axis("off")
ax_header.text(0, 0.65, selected_pitcher, fontsize=26, weight="bold")
ax_header.text(0.78, 0.60, f"Outing Summary\n{season_range}",
               fontsize=14, ha="right")


#Summary Boxes

ax_sum = fig.add_subplot(gs[1,:])
ax_sum.axis("off")

num_boxes = len(summary_labels)
for i, (lab, val) in enumerate(zip(summary_labels, summary_values)):
    x_left = i / num_boxes
    width = 1 / num_boxes
    ax_sum.text(x_left + width / 2, 0.70, lab, fontsize=11, ha="center", weight="bold")
    ax_sum.add_patch(plt.Rectangle((x_left + 0.06, 0.15),
                                   width - 0.12, 0.45,
                                   fill=False, linewidth=1))
    ax_sum.text(x_left + width / 2, 0.32, val, fontsize=12, ha="center")

# Movement Plot
ax_mvmt = fig.add_subplot(gs[2:5, 0:2])
for pt in pitch_types:
    sub = p[p["TaggedPitchType"] == pt]
    ax_mvmt.scatter(
        sub["HorzBreak"], sub["InducedVertBreak"],
        s=65, edgecolor="black", color=colors[pt], alpha=0.9, label=pt
    )

ax_mvmt.axvline(0, color="gray", linestyle="--", linewidth=0.8)
ax_mvmt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax_mvmt.set_title("Pitch Movement", fontsize=16, weight="bold")
ax_mvmt.set_xlabel("Horizontal Break (in)")
ax_mvmt.set_ylabel("Induced Vertical Break (in)")
ax_mvmt.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
ax_mvmt.grid(alpha=0.25, linestyle="--")
ax_mvmt.set_xlim(-25, 25)
ax_mvmt.set_ylim(-25, 25)

# Velo over time
ax_velo = fig.add_subplot(gs[2:3, 2:4])
for pt in pitch_types:
    sub = p[p["TaggedPitchType"] == pt]
    ax_velo.plot(
        sub["PitchNo"], sub["RelSpeed"],
        marker="o", markersize=6, linewidth=2,
        color=colors[pt], label=pt
    )

ax_velo.set_title("Velocity Over Time", fontsize=14, weight="bold")
ax_velo.set_xlabel("Pitch #")
ax_velo.set_ylabel("Velocity (mph)")
ax_velo.grid(alpha=0.3)
ax_velo.legend(frameon=False, bbox_to_anchor=(1.1, 1), loc="upper left")

# Strike zone map
ax_heat = fig.add_subplot(gs[3:5, 2:4])

x_min, x_max = -20, 20
y_min, y_max = 0, 5.5

for pt in pitch_types:
    sub = p[p["TaggedPitchType"] == pt]
    px = sub["PlateLocSide"].values * 12
    pz = sub["PlateLocHeight"].values
    ax_heat.scatter(
        np.clip(px, x_min, x_max),
        np.clip(pz, y_min, y_max),
        s=55, color=colors[pt],
        edgecolor="black", linewidth=0.7, alpha=0.9
    )

ax_heat.axvline(-8.5, color="black", linestyle="--", linewidth=0.8)
ax_heat.axvline(8.5,  color="black", linestyle="--", linewidth=0.8)
ax_heat.axhline(1.5, color="black", linestyle="--", linewidth=0.8)
ax_heat.axhline(3.5, color="black", linestyle="--", linewidth=0.8)

ax_heat.set_xlim(-20, 20)
ax_heat.set_ylim(0, 5.5)
ax_heat.set_title("Strike Zone Heatmap", fontsize=14, weight="bold")
ax_heat.set_xlabel("Plate Side (inches)")
ax_heat.set_ylabel("Plate Height (ft)")
ax_heat.grid(alpha=0.25, linestyle="--")


# Pitch type summary table
ax_tbl = fig.add_subplot(gs[6,:])
ax_tbl.axis("off")

def summarize(sub):
    bip = sub["IsBIP"].sum()
    gb = sub["GB"].sum()
    ld = sub["LD"].sum()
    fb = sub["FB"].sum()

    return [
        len(sub),
        sub["RelSpeed"].mean(),
        sub["InducedVertBreak"].mean(),
        sub["HorzBreak"].mean(),
        sub["SpinRate"].mean(),
        sub["HorzApprAngle"].mean(),
        sub["VertApprAngle"].mean(),
        compute_pzr(sub),
        gb / bip * 100 if bip > 0 else np.nan,
        ld / bip * 100 if bip > 0 else np.nan,
        fb / bip * 100 if bip > 0 else np.nan,
        sub.loc[sub["IsBIP"], "ExitSpeed"].mean(),
        gb / fb if fb > 0 else np.nan
    ]

rows, row_labels = [], []
for pt in pitch_types:
    sub = p[p["TaggedPitchType"] == pt]
    rows.append(summarize(sub))
    row_labels.append(pt)

rows = np.array(rows)

col_labels = [
    "Count", "Velo", "IVB", "HB", "Spin", "HAA", "VAA", "PZR%",
    "GB%", "LD%", "FB%", "Avg EV", "GB/FB"
]

table_str = [[
    f"{r[0]:.0f}", f"{r[1]:.1f}", f"{r[2]:.1f}", f"{r[3]:.1f}",
    f"{r[4]:.0f}", f"{r[5]:.2f}", f"{r[6]:.2f}", f"{r[7]:.1f}%",
    f"{r[8]:.1f}%" if not np.isnan(r[8]) else "–",
    f"{r[9]:.1f}%" if not np.isnan(r[9]) else "–",
    f"{r[10]:.1f}%" if not np.isnan(r[10]) else "–",
    f"{r[11]:.1f}" if not np.isnan(r[11]) else "–",
    f"{r[12]:.2f}" if not np.isnan(r[12]) else "–"
] for r in rows]

tbl = ax_tbl.table(
    cellText=table_str,
    rowLabels=row_labels,
    colLabels=col_labels,
    loc="center",
    cellLoc="center"
)

tbl.scale(1.2, 1.4)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
ax_tbl.set_title("Pitch Type Summary", fontsize=16, weight="bold", pad=18)

# OUTPUT
st.pyplot(fig)

buffer = io.BytesIO()
fig.savefig(buffer, format="png", bbox_inches="tight")
buffer.seek(0)

st.download_button(
    "Download Pitching Report as PNG",
    buffer,
    f"{selected_pitcher.replace(' ','_')}_report.png",
    "image/png"
)



