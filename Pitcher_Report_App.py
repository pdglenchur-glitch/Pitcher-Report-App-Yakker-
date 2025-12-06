import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import io
import re
from datetime import datetime

# Streamlit Page Setup
st.set_page_config(layout="wide", page_title="Pitching Report Generator")
st.title("⚾ Pitching Report Builder – Streamlit Edition")
st.write("Upload multiple CSVs → choose pitcher → filter by date → generate full scouting report.")

# Helper: Extract date from filename (MM_DD_YY or MM_DD_YYYY)
def extract_date(filename):
    match = re.search(r"(\d{1,2})_(\d{1,2})_(\d{2,4})", filename)
    if match:
        month, day, year = match.groups()
        if len(year) == 2:
            year = "20" + year
        return datetime(int(year), int(month), int(day))
    return None

# File Upload
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

# Season Date Range
valid_dates = [d for d in file_dates if d is not None]
if valid_dates:
    start_date = min(valid_dates)
    end_date = max(valid_dates)
    season_range = f"{start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}"
else:
    season_range = "Unknown Date Range"

# Date Filter UI (Before Pitcher Filter)
available_dates = sorted(df["SessionDate"].dropna().unique())

selected_dates = st.multiselect(
    "Filter by session date(s):",
    options=available_dates,
    default=available_dates
)

df = df[df["SessionDate"].isin(selected_dates)]

# Pitcher Selection
pitchers = sorted(df["Pitcher"].dropna().unique().tolist())
selected_pitcher = st.selectbox("Select a pitcher:", pitchers)

p = df[df["Pitcher"] == selected_pitcher].copy()

if p.empty:
    st.warning("No data for this pitcher in the selected date range.")
    st.stop()

# Clean & Reorder
p = p[p["TaggedPitchType"].notna()]
p = p[p["RelSpeed"].notna()]
p = p.sort_values("PitchNo")
p["PitchNo"] = range(1, len(p) + 1)

# SUMMARY STATS
total_pitches = len(p)
avg_velo = p["RelSpeed"].mean()
max_velo = p["RelSpeed"].max()
min_velo = p["RelSpeed"].min()

# PZR CALCULATION FUNCTION
margin_in = 2.85
margin_ft = margin_in / 12

# Base strike zone
sx_left  = -8.5 - margin_in
sx_right =  8.5 + margin_in
sz_bot   = 1.5 - margin_ft
sz_top   = 3.5 + margin_ft

def compute_pzr(sub):
    px = sub["PlateLocSide"] * 12   # convert feet → inches
    pz = sub["PlateLocHeight"]
    pzr_mask = (
        (px >= sx_left) &
        (px <= sx_right) &
        (pz >= sz_bot) &
        (pz <= sz_top)
    )
    return pzr_mask.mean() * 100

overall_pzr = compute_pzr(p)

# SUMMARY BOX ROW (WITH PZR ADDED)
summary_labels = ["Pitches", "Avg Velo", "Max Velo", "Min Velo", "PZR%"]
summary_values = [
    f"{total_pitches}",
    f"{avg_velo:.1f}",
    f"{max_velo:.1f}",
    f"{min_velo:.1f}",
    f"{overall_pzr:.1f}%"
]

# PITCH COLORS
pitch_types = p["TaggedPitchType"].unique().tolist()
palette = sns.color_palette("husl", n_colors=len(pitch_types))
colors = {pt: palette[i] for i, pt in enumerate(pitch_types)}

# FIGURE LAYOUT
fig = plt.figure(figsize=(17, 13), dpi=150)
gs = gridspec.GridSpec(
    7, 4,
    height_ratios=[0.8, 1.2, 3.0, 2.2, 2.2, 0.45, 2.8],
    hspace=1.05,
    wspace=1.05
)

# HEADER
ax_header = fig.add_subplot(gs[0,:])
ax_header.axis("off")
ax_header.text(0, 0.65, f"{selected_pitcher}", fontsize=26, weight="bold")
ax_header.text(0.78, 0.60, f"Outing Summary\n{season_range}",
               fontsize=14, ha="right")

# SUMMARY BOXES
ax_sum = fig.add_subplot(gs[1,:])
ax_sum.axis("off")

num_boxes = len(summary_labels)
for i, (lab, val) in enumerate(zip(summary_labels, summary_values)):
    x_left = i / num_boxes
    width = 1 / num_boxes

    ax_sum.text(x_left + width / 2, 0.70, lab, fontsize=11, ha="center", weight="bold")
    ax_sum.add_patch(plt.Rectangle((x_left + 0.06, 0.15), width - 0.12, 0.45,
                                   fill=False, linewidth=1))
    ax_sum.text(x_left + width / 2, 0.32, val, fontsize=12, ha="center")

# PITCH MOVEMENT
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

# Lock movement plot axes (prevents shifting)
ax_mvmt.set_xlim(-25, 25)
ax_mvmt.set_ylim(-25, 25)


# VELOCITY OVER TIME
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

# STRIKE ZONE HEATMAP
ax_heat = fig.add_subplot(gs[3:5, 2:4])

# Fixed boundaries
x_min, x_max = -20, 20
y_min, y_max = 0, 5.5

# Scatter points with clamping for out-of-bounds pitches
for pt in pitch_types:
    sub = p[p["TaggedPitchType"] == pt]

    px = sub["PlateLocSide"].values * 12   # convert to inches
    pz = sub["PlateLocHeight"].values

    # Clamp to boundaries
    px_clamped = np.clip(px, x_min, x_max)
    pz_clamped = np.clip(pz, y_min, y_max)

    ax_heat.scatter(
        px_clamped,
        pz_clamped,
        s=55,                       # your preferred size
        color=colors[pt],
        edgecolor="black",
        linewidth=0.7,
        alpha=0.9
    )
    
#Strike Zone Boundaries
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

# PITCH TYPE SUMMARY TABLE (WITH PZR%)
ax_tbl = fig.add_subplot(gs[6,:])
ax_tbl.axis("off")

def summarize(sub):
    return [
        len(sub),
        sub["RelSpeed"].mean(),
        sub["InducedVertBreak"].mean(),
        sub["HorzBreak"].mean(),
        sub["SpinRate"].mean(),
        sub["HorzApprAngle"].mean(),
        sub["VertApprAngle"].mean(),
        compute_pzr(sub)               # <-- NEW PZR COLUMN
    ]

rows = []
row_labels = []

for pt in pitch_types:
    sub = p[p["TaggedPitchType"] == pt]
    rows.append(summarize(sub))
    row_labels.append(pt)

rows = np.array(rows)

col_labels = ["Count", "Velo", "IVB", "HB", "Spin", "HAA", "VAA", "PZR%"]

table_str = [
    [
        f"{r[0]:.0f}",
        f"{r[1]:.1f}",
        f"{r[2]:.1f}",
        f"{r[3]:.1f}",
        f"{r[4]:.0f}" if not np.isnan(r[4]) else "–",
        f"{r[5]:.2f}" if not np.isnan(r[5]) else "–",
        f"{r[6]:.2f}" if not np.isnan(r[6]) else "–",
        f"{r[7]:.1f}%"               # <-- NEW PZR COLUMN DISPLAY
    ]
    for r in rows
]

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

# Streamlit Output
st.pyplot(fig)

# Download Button
buffer = io.BytesIO()
fig.savefig(buffer, format="png", bbox_inches="tight")
buffer.seek(0)

st.download_button(
    label="Download Pitching Report as PNG",
    data=buffer,
    file_name=f"{selected_pitcher.replace(' ','_')}_report.png",
    mime="image/png"
)
