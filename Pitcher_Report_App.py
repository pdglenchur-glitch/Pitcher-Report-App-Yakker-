import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import io
import re
from datetime import datetime
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(layout="wide", page_title="Pitching + Hitting Report Builder")
st.title("⚾ Pitching + Hitting Report Builder")
st.write("Upload CSV(s) → filter by date → pick ANY pitcher or hitter in the file.")

# -----------------------------
# Helpers
# -----------------------------
def extract_date(filename: str):
    match = re.search(r"(\d{1,2})[_-](\d{1,2})[_-](\d{2,4})", filename)
    if match:
        month, day, year = match.groups()
        if len(year) == 2:
            year = "20" + year
        try:
            return datetime(int(year), int(month), int(day))
        except ValueError:
            return None
    return None

def safe_mean(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce")
    return float(s.mean()) if s.notna().any() else np.nan

def pct(numer, denom):
    return float(numer) / float(denom) * 100.0 if denom and denom > 0 else np.nan

def format_pct(x):
    return "–" if pd.isna(x) else f"{x:.1f}%"

def format_num(x, d=1):
    return "–" if pd.isna(x) else f"{x:.{d}f}"

def require_cols(df: pd.DataFrame, cols: list[str], label: str = ""):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns {label}: {missing}")
        st.stop()

def add_foul_flag(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not obj_cols:
        df["RowHasFoul"] = False
        return df
    combined = (
        df[obj_cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )
    df["RowHasFoul"] = combined.str.contains(r"\bfoul\b", case=False, regex=True)
    return df

def add_zone_flags(df: pd.DataFrame) -> pd.DataFrame:
    margin_in = 2.85
    margin_ft = margin_in / 12
    sx_left, sx_right = -8.5 - margin_in, 8.5 + margin_in
    sz_bot,  sz_top   = 1.5  - margin_ft, 3.5 + margin_ft
    px = pd.to_numeric(df["PlateLocSide"],   errors="coerce") * 12
    pz = pd.to_numeric(df["PlateLocHeight"], errors="coerce")
    df["InZone"]       = (px >= sx_left) & (px <= sx_right) & (pz >= sz_bot) & (pz <= sz_top)
    df["ZoneLocKnown"] = px.notna() & pz.notna()
    return df

def add_batted_ball_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Priority:
      1. HitType column, recognised value → use directly
         GroundBall→GB, LineDrive→LD, FlyBall/Popup→FB, Throwdown→excluded
      2. No HitType → fall back to launch angle (non-foul BIP with ExitSpeed)
         GB: -90–10°, LD: 10–25°, FB: 25–90°
      3. Neither → not a valid BIP
    IsBIP = GB | LD | FB
    """
    HT_GB    = {"GroundBall"}
    HT_LD    = {"LineDrive"}
    HT_FB    = {"FlyBall", "Popup"}
    HT_VALID = HT_GB | HT_LD | HT_FB

    has_hittype = "HitType" in df.columns
    df["GB"] = False
    df["LD"] = False
    df["FB"] = False

    for idx in df.index:
        ht = df.at[idx, "HitType"] if has_hittype else np.nan
        if pd.notna(ht) and str(ht) in HT_VALID:
            ht_str = str(ht)
            df.at[idx, "GB"] = ht_str in HT_GB
            df.at[idx, "LD"] = ht_str in HT_LD
            df.at[idx, "FB"] = ht_str in HT_FB
            continue
        if df.at[idx, "RowHasFoul"]:
            continue
        if pd.isna(df.at[idx, "ExitSpeed"]):
            continue
        angle = pd.to_numeric(df.at[idx, "Angle"], errors="coerce")
        if pd.isna(angle):
            continue
        if   -90 <= angle <= 10:  df.at[idx, "GB"] = True
        elif  10 < angle  <= 25:  df.at[idx, "LD"] = True
        elif  25 < angle  <= 90:  df.at[idx, "FB"] = True

    df["IsBIP"] = df["GB"] | df["LD"] | df["FB"]
    return df

def resolve_batter_side(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds BatterSideResolved column. Switch hitters are remapped to the
    handedness they actually bat from against this pitcher:
        Switch vs RHP  →  Left   (switch hitter bats left vs righty)
        Switch vs LHP  →  Right  (switch hitter bats right vs lefty)
    Non-switch batters keep their original BatterSide value.
    Rows missing BatterSide or PitcherThrows are left as NaN.
    """
    if "BatterSide" not in df.columns:
        df["BatterSideResolved"] = np.nan
        return df

    pitcher_throws = df["PitcherThrows"] if "PitcherThrows" in df.columns else pd.Series(np.nan, index=df.index)

    def _resolve(row):
        side   = row["BatterSide"]
        throws = pitcher_throws.loc[row.name] if hasattr(pitcher_throws, "loc") else np.nan
        if pd.isna(side):
            return np.nan
        if side != "Switch":
            return side
        # Switch hitter: bat opposite of pitcher's arm
        if throws == "Right":
            return "Left"
        if throws == "Left":
            return "Right"
        return np.nan   # PitcherThrows unknown — can't resolve

    df["BatterSideResolved"] = df.apply(_resolve, axis=1)
    return df

def fit_table_fontsize(table, ax, min_fontsize=6, start_fontsize=9):
    fig      = ax.get_figure()
    renderer = fig.canvas.get_renderer()
    fontsize = start_fontsize
    while fontsize >= min_fontsize:
        table.set_fontsize(fontsize)
        fig.canvas.draw()
        overflow = False
        for (row, col), cell in table.get_celld().items():
            bbox     = cell.get_window_extent(renderer)
            txt_bbox = cell.get_text().get_window_extent(renderer)
            if txt_bbox.width > bbox.width - 4:
                overflow = True
                break
        if not overflow:
            break
        fontsize -= 0.5
    return fontsize

# -----------------------------
# Upload
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload one or more CSV files",
    type=["csv"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.stop()

df_list    = []
file_dates = []

for f in uploaded_files:
    temp = pd.read_csv(f)
    session_date = extract_date(f.name)
    temp["SourceFile"]  = f.name
    temp["SessionDate"] = session_date
    # Normalise name columns to prevent duplicate entries from whitespace.
    # Cast to string first so .str accessor never raises on mixed/numeric dtypes.
    for col in ["Pitcher", "Batter"]:
        if col in temp.columns:
            temp[col] = (
                temp[col]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .replace("nan", np.nan)   # restore genuine NaNs cast to "nan"
            )
    df_list.append(temp)
    file_dates.append(session_date)

df = pd.concat(df_list, ignore_index=True)
st.success(f"Loaded {len(uploaded_files)} file(s) with {len(df):,} total rows.")

needed = [
    "Pitcher", "Batter",
    "TaggedPitchType", "PitchCall", "PitchNo",
    "RelSpeed", "HorzBreak", "InducedVertBreak",
    "PlateLocSide", "PlateLocHeight",
    "ExitSpeed", "Angle"
]
require_cols(df, needed, label="for this app")

df = add_foul_flag(df)
df = add_zone_flags(df)
df = add_batted_ball_flags(df)
df = resolve_batter_side(df)   # Switch hitters remapped here globally

# -----------------------------
# Date filter
# -----------------------------
valid_dates  = [d for d in file_dates if d is not None]
season_range = (
    f"{min(valid_dates):%Y-%m-%d} → {max(valid_dates):%Y-%m-%d}"
    if valid_dates else "Unknown Date Range"
)

available_dates = sorted(df["SessionDate"].dropna().unique())
if available_dates:
    selected_dates = st.multiselect(
        "Filter by session date(s):",
        options=available_dates,
        default=available_dates
    )
    df = df[df["SessionDate"].isin(selected_dates)]
else:
    st.info("No session dates detected in filenames; skipping date filter.")

# -----------------------------
# Tabs
# -----------------------------
tab_pitcher, tab_hitter = st.tabs(["🎯 Pitcher Report", "🏏 Hitter Dashboard"])

# =========================================================
# PITCHER REPORT
# =========================================================
with tab_pitcher:
    st.subheader("Pitcher Report Generator (All Players)")

    pitchers = sorted(df["Pitcher"].dropna().unique())
    if not pitchers:
        st.warning("No pitchers found in the dataset.")
        st.stop()

    selected_pitcher = st.selectbox("Select a pitcher:", pitchers)

    p = df[df["Pitcher"] == selected_pitcher].copy()
    if p.empty:
        st.warning("No data for this pitcher.")
        st.stop()

    p = p[p["TaggedPitchType"].notna()]
    p = p[p["RelSpeed"].notna()]
    p = p.sort_values("PitchNo").reset_index(drop=True)
    p["PitchNo"] = range(1, len(p) + 1)

    p["IsCalledStrike"]   = p["PitchCall"].eq("StrikeCalled")
    p["IsSwingingStrike"] = p["PitchCall"].eq("StrikeSwinging")

    all_pitch_types = p["TaggedPitchType"].dropna().unique().tolist()

    # -------------------------------------------------------
    # Pitch type filter
    # -------------------------------------------------------
    st.markdown("**Select pitch types to include in the report:**")
    selected_pitch_types = st.multiselect(
        "Pitch types shown on plots and summary table:",
        options=all_pitch_types,
        default=all_pitch_types,
        help="Deselect a pitch type to remove it from all charts and tables in the PNG."
    )

    if not selected_pitch_types:
        st.warning("Please select at least one pitch type.")
        st.stop()

    p_plot = p[p["TaggedPitchType"].isin(selected_pitch_types)].copy()
    p_plot = p_plot.sort_values("PitchNo").reset_index(drop=True)
    p_plot["PitchNo"] = range(1, len(p_plot) + 1)

    total_bip      = int(p_plot["IsBIP"].sum())
    overall_gb_pct = pct(int(p_plot["GB"].sum()), total_bip)
    overall_ld_pct = pct(int(p_plot["LD"].sum()), total_bip)
    overall_fb_pct = pct(int(p_plot["FB"].sum()), total_bip)
    total_pitches  = len(p_plot)
    max_velo       = float(pd.to_numeric(p_plot["RelSpeed"], errors="coerce").max())
    overall_pzr    = pct(int(p_plot["InZone"].sum()), len(p_plot))

    summary_labels = ["Pitches", "GB%", "LD%", "FB%", "Max Velo", "PZR%"]
    summary_values = [
        f"{total_pitches}",
        format_pct(overall_gb_pct),
        format_pct(overall_ld_pct),
        format_pct(overall_fb_pct),
        format_num(max_velo, 1),
        format_pct(overall_pzr),
    ]

    pitch_types = selected_pitch_types
    palette     = sns.color_palette("husl", len(pitch_types)) if pitch_types else []
    colors      = dict(zip(pitch_types, palette))

    # ------------------------------------------------------------------
    # Build per-pitch-type summary rows (left table)
    # ------------------------------------------------------------------
    for opt in ["SpinRate", "HorzApprAngle", "VertApprAngle"]:
        if opt not in p_plot.columns:
            p_plot[opt] = np.nan

    def summarize_pitch_type(sub):
        bip = int(sub["IsBIP"].sum())
        gb  = int(sub["GB"].sum())
        ld  = int(sub["LD"].sum())
        fb  = int(sub["FB"].sum())
        avg_ev = safe_mean(sub.loc[sub["IsBIP"], "ExitSpeed"])
        pzr    = pct(int(sub["InZone"].sum()), len(sub))
        csw    = pct(
            int(sub["IsCalledStrike"].sum()) + int(sub["IsSwingingStrike"].sum()),
            len(sub)
        )
        return [
            len(sub),
            safe_mean(sub["RelSpeed"]),
            safe_mean(sub["InducedVertBreak"]),
            safe_mean(sub["HorzBreak"]),
            safe_mean(sub["SpinRate"]),
            safe_mean(sub["HorzApprAngle"]),
            safe_mean(sub["VertApprAngle"]),
            pzr, csw,
            pct(gb, bip), pct(ld, bip), pct(fb, bip),
            avg_ev,
            (gb / fb) if fb > 0 else np.nan
        ]

    rows, row_labels = [], []
    for pt in pitch_types:
        sub = p_plot[p_plot["TaggedPitchType"] == pt]
        rows.append(summarize_pitch_type(sub))
        row_labels.append(pt)

    col_labels_left = [
        "Count", "Velo", "IVB", "HB", "Spin", "HAA", "VAA", "PZR%",
        "CSW%", "GB%", "LD%", "FB%", "Avg EV", "GB/FB"
    ]

    table_str_left = []
    for r in rows:
        table_str_left.append([
            f"{r[0]:.0f}",
            format_num(r[1], 1),  format_num(r[2], 1),  format_num(r[3], 1),
            format_num(r[4], 0),  format_num(r[5], 2),  format_num(r[6], 2),
            format_pct(r[7]),     format_pct(r[8]),
            format_pct(r[9]),     format_pct(r[10]),     format_pct(r[11]),
            format_num(r[12], 1), format_num(r[13], 2),
        ])

    # ------------------------------------------------------------------
    # Build handedness split table (right table)
    # Uses BatterSideResolved — switch hitters are already remapped to
    # Left (vs RHP) or Right (vs LHP), so only "Right" and "Left" groups
    # appear here. No "vs S" column will ever be generated.
    # ------------------------------------------------------------------
    hand_col_p = "BatterSideResolved"
    hands_present_p = []
    if hand_col_p in p_plot.columns:
        hands_present_p = sorted(
            p_plot[hand_col_p].dropna().unique(),
            key=lambda x: {"Right": 0, "Left": 1}.get(x, 9)
        )

    hand_label_map_p = {"Right": "vs R", "Left": "vs L"}

    def pitcher_hand_split(sub: pd.DataFrame) -> dict:
        result = {}
        for hnd in hands_present_p:
            g     = sub[sub[hand_col_p] == hnd] if hand_col_p in sub.columns else pd.DataFrame()
            label = hand_label_map_p.get(hnd, hnd)
            if g.empty:
                result[f"{label}\nPZR%"]   = np.nan
                result[f"{label}\nCSW%"]   = np.nan
                result[f"{label}\nGB%"]    = np.nan
                result[f"{label}\nLD%"]    = np.nan
                result[f"{label}\nFB%"]    = np.nan
                result[f"{label}\nAvg EV"] = np.nan
                result[f"{label}\nGB/FB"]  = np.nan
            else:
                bip_g = int(g["IsBIP"].sum())
                gb_g  = int(g["GB"].sum())
                ld_g  = int(g["LD"].sum())
                fb_g  = int(g["FB"].sum())
                result[f"{label}\nPZR%"]   = pct(int(g["InZone"].sum()), len(g))
                result[f"{label}\nCSW%"]   = pct(
                    int(g["IsCalledStrike"].sum()) + int(g["IsSwingingStrike"].sum()),
                    len(g)
                )
                result[f"{label}\nGB%"]    = pct(gb_g, bip_g)
                result[f"{label}\nLD%"]    = pct(ld_g, bip_g)
                result[f"{label}\nFB%"]    = pct(fb_g, bip_g)
                result[f"{label}\nAvg EV"] = safe_mean(g.loc[g["IsBIP"], "ExitSpeed"])
                result[f"{label}\nGB/FB"]  = (gb_g / fb_g) if fb_g > 0 else np.nan
        return result

    hand_rows_p = []
    for pt in pitch_types:
        sub = p_plot[p_plot["TaggedPitchType"] == pt]
        hand_rows_p.append({"Pitch Type": pt, **pitcher_hand_split(sub)})

    hand_df_p = pd.DataFrame(hand_rows_p)

    hand_stat_cols_p = []
    for hnd in hands_present_p:
        label = hand_label_map_p.get(hnd, hnd)
        hand_stat_cols_p += [
            f"{label}\nPZR%", f"{label}\nCSW%",
            f"{label}\nGB%",  f"{label}\nLD%", f"{label}\nFB%",
            f"{label}\nAvg EV", f"{label}\nGB/FB"
        ]
    hand_display_cols_p = ["Pitch Type"] + hand_stat_cols_p

    hand_df_p_fmt = hand_df_p[hand_display_cols_p].copy()
    for col in hand_stat_cols_p:
        if any(m in col for m in ["PZR%", "CSW%", "GB%", "LD%", "FB%"]):
            hand_df_p_fmt[col] = hand_df_p_fmt[col].map(
                lambda x: "–" if pd.isna(x) else f"{x:.1f}%")
        elif "Avg EV" in col:
            hand_df_p_fmt[col] = hand_df_p_fmt[col].map(
                lambda x: "–" if pd.isna(x) else f"{x:.1f}")
        else:  # GB/FB
            hand_df_p_fmt[col] = hand_df_p_fmt[col].map(
                lambda x: "–" if pd.isna(x) else f"{x:.2f}")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(22, 17), dpi=150)
    gs = gridspec.GridSpec(
        7, 4,
        height_ratios=[0.8, 1.2, 3.0, 2.2, 2.2, 0.45, 4.8],
        hspace=1.05,
        wspace=1.05
    )

    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis("off")
    ax_header.text(0, 0.65, selected_pitcher, fontsize=26, weight="bold")
    ax_header.text(0.78, 0.60, f"Outing Summary\n{season_range}", fontsize=14, ha="right")

    ax_sum = fig.add_subplot(gs[1, :])
    ax_sum.axis("off")
    num_boxes = len(summary_labels)
    for i, (lab, val) in enumerate(zip(summary_labels, summary_values)):
        x_left = i / num_boxes
        width  = 1 / num_boxes
        ax_sum.text(x_left + width / 2, 0.70, lab, fontsize=11, ha="center", weight="bold")
        ax_sum.add_patch(plt.Rectangle((x_left + 0.06, 0.15),
                                       width - 0.12, 0.45,
                                       fill=False, linewidth=1))
        ax_sum.text(x_left + width / 2, 0.32, val, fontsize=12, ha="center")

    ax_mvmt = fig.add_subplot(gs[2:5, 0:2])
    for pt in pitch_types:
        sub = p_plot[p_plot["TaggedPitchType"] == pt]
        ax_mvmt.scatter(
            pd.to_numeric(sub["HorzBreak"],        errors="coerce"),
            pd.to_numeric(sub["InducedVertBreak"], errors="coerce"),
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

    ax_velo = fig.add_subplot(gs[2:3, 2:4])
    for pt in pitch_types:
        sub = p_plot[p_plot["TaggedPitchType"] == pt]
        ax_velo.plot(
            sub["PitchNo"],
            pd.to_numeric(sub["RelSpeed"], errors="coerce"),
            marker="o", markersize=6, linewidth=2,
            color=colors[pt], label=pt
        )
    ax_velo.set_title("Velocity Over Time", fontsize=14, weight="bold")
    ax_velo.set_xlabel("Pitch #")
    ax_velo.set_ylabel("Velocity (mph)")
    ax_velo.grid(alpha=0.3)
    ax_velo.legend(frameon=False, bbox_to_anchor=(1.1, 1), loc="upper left")

    ax_heat = fig.add_subplot(gs[3:5, 2:4])
    x_min, x_max = -20, 20
    y_min, y_max =   0, 5.5
    for pt in pitch_types:
        sub = p_plot[p_plot["TaggedPitchType"] == pt]
        px  = pd.to_numeric(sub["PlateLocSide"],   errors="coerce").values * 12
        pz  = pd.to_numeric(sub["PlateLocHeight"],  errors="coerce").values
        ax_heat.scatter(
            np.clip(px, x_min, x_max), np.clip(pz, y_min, y_max),
            s=55, color=colors[pt], edgecolor="black", linewidth=0.7, alpha=0.9
        )
    ax_heat.axvline(-8.5, color="black", linestyle="--", linewidth=0.8)
    ax_heat.axvline( 8.5, color="black", linestyle="--", linewidth=0.8)
    ax_heat.axhline( 1.5, color="black", linestyle="--", linewidth=0.8)
    ax_heat.axhline( 3.5, color="black", linestyle="--", linewidth=0.8)
    ax_heat.set_xlim(-20, 20);  ax_heat.set_ylim(0, 5.5)
    ax_heat.set_title("Strike Zone Heatmap", fontsize=14, weight="bold")
    ax_heat.set_xlabel("Plate Side (inches)")
    ax_heat.set_ylabel("Plate Height (ft)")
    ax_heat.grid(alpha=0.25, linestyle="--")

    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=gs[6, :],
        wspace=0.08,
        width_ratios=[1.0, 1.0]
    )
    ax_tbl_left  = fig.add_subplot(gs_bot[0, 0])
    ax_tbl_right = fig.add_subplot(gs_bot[0, 1])
    ax_tbl_left.axis("off")
    ax_tbl_right.axis("off")

    TABLE_FONTSIZE = 10
    TABLE_TOP      = 0.78
    TABLE_SCALE_Y  = 1.6

    ax_tbl_left.text(
        0.5, 0.97, "Pitch Type Summary",
        ha="center", va="top", fontsize=12, weight="bold",
        transform=ax_tbl_left.transAxes
    )
    ax_tbl_right.text(
        0.5, 0.97, "Split by Batter Handedness",
        ha="center", va="top", fontsize=12, weight="bold",
        transform=ax_tbl_right.transAxes
    )

    tbl_left = ax_tbl_left.table(
        cellText=table_str_left,
        rowLabels=row_labels,
        colLabels=col_labels_left,
        loc="center",
        cellLoc="center",
        bbox=[0.0, 0.0, 1.0, TABLE_TOP]
    )
    tbl_left.auto_set_font_size(False)
    tbl_left.auto_set_column_width(col=list(range(len(col_labels_left))))
    tbl_left.set_fontsize(TABLE_FONTSIZE)
    tbl_left.scale(1.0, TABLE_SCALE_Y)

    if hand_df_p_fmt.empty or not hand_stat_cols_p:
        ax_tbl_right.text(
            0.5, 0.45, "No batter handedness\ndata available.",
            ha="center", va="center", fontsize=9,
            transform=ax_tbl_right.transAxes
        )
    else:
        tbl_right = ax_tbl_right.table(
            cellText=hand_df_p_fmt.values.tolist(),
            colLabels=list(hand_df_p_fmt.columns),
            loc="center",
            cellLoc="center",
            bbox=[0.0, 0.0, 1.0, TABLE_TOP]
        )
        tbl_right.auto_set_font_size(False)
        tbl_right.auto_set_column_width(col=list(range(len(hand_display_cols_p))))
        tbl_right.set_fontsize(TABLE_FONTSIZE)
        tbl_right.scale(1.0, TABLE_SCALE_Y)

    st.pyplot(fig)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)

    st.download_button(
        "Download Pitching Report as PNG",
        buffer,
        f"{selected_pitcher.replace(' ', '_')}_report.png",
        "image/png"
    )

# =========================================================
# HITTER DASHBOARD
# =========================================================
with tab_hitter:
    st.subheader("Hitter Whiff% + Exit Velocity (EV) + Chase% (All Players)")

    SWING_CALLS   = {"StrikeSwinging", "Foul", "InPlay"}
    WHIFF_CALLS   = {"StrikeSwinging"}
    CONTACT_CALLS = {"Foul", "InPlay"}

    hitters = sorted(df["Batter"].dropna().unique())
    if not hitters:
        st.warning("No hitters found in the dataset.")
        st.stop()

    selected_hitter = st.selectbox("Select a hitter:", hitters)

    h = df[df["Batter"] == selected_hitter].copy()
    if h.empty:
        st.warning("No data for this hitter.")
        st.stop()

    h = h[h["TaggedPitchType"].notna()]
    h = h[h["PitchCall"].notna()]

    h["IsSwing"]   = h["PitchCall"].isin(SWING_CALLS)
    h["IsWhiff"]   = h["PitchCall"].isin(WHIFF_CALLS)
    h["IsContact"] = h["PitchCall"].isin(CONTACT_CALLS)

    h["IsBIP"]      = h["ExitSpeed"].notna() & (~h["RowHasFoul"])
    h["IsChase"]    = h["IsSwing"] & (h["ZoneLocKnown"]) & (~h["InZone"])
    h["IsZSwing"]   = h["IsSwing"]   & (h["ZoneLocKnown"]) & (h["InZone"])
    h["IsZContact"] = h["IsContact"] & (h["ZoneLocKnown"]) & (h["InZone"])

    total_pitches   = len(h)
    total_swings    = int(h["IsSwing"].sum())
    total_whiffs    = int(h["IsWhiff"].sum())
    total_chases    = int(h["IsChase"].sum())
    total_z_swings  = int(h["IsZSwing"].sum())
    total_z_contact = int(h["IsZContact"].sum())

    overall_whiff     = pct(total_whiffs,    total_swings)
    overall_chase     = pct(total_chases,    total_pitches)
    overall_z_contact = pct(total_z_contact, total_z_swings)

    exit_speed_all   = pd.to_numeric(h["ExitSpeed"], errors="coerce")
    angle_all        = pd.to_numeric(h["Angle"],     errors="coerce")
    max_ev_overall   = float(exit_speed_all.max())          if exit_speed_all.notna().any() else np.nan
    avg_launch_angle = float(angle_all.mean())              if angle_all.notna().any()      else np.nan
    ev_90th          = float(exit_speed_all.quantile(0.90)) if exit_speed_all.notna().any() else np.nan

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Overall Chase%",   format_pct(overall_chase))
    m2.metric("Z-Contact%",       format_pct(overall_z_contact))
    m3.metric("Max EV (Overall)", format_num(max_ev_overall, 1))
    m4.metric("Avg Launch Angle", format_num(avg_launch_angle, 1))
    m5.metric("90th % EV",        format_num(ev_90th, 1))

    st.caption(
        "Whiff% = Swinging Strikes / All Swings, where Swings = {StrikeSwinging, Foul, InPlay}. "
        "Chase% = (Swings on out-of-zone pitches) / (All pitches). "
        "Z-Contact% = (In-zone swings with contact) / (All in-zone swings), where contact = {Foul, InPlay}. "
        "Zone matches pitcher report PZR. EV excludes any row tagged 'foul' anywhere in the row only for BIP-specific metrics."
    )

    def hitter_group_summary(g: pd.DataFrame):
        pitches   = len(g)
        swings    = int(g["IsSwing"].sum())
        whiffs    = int(g["IsWhiff"].sum())
        chases    = int(g["IsChase"].sum())
        z_swings  = int(g["IsZSwing"].sum())
        z_contact = int(g["IsZContact"].sum())
        whiff_pct     = pct(whiffs,    swings)
        chase_pct     = pct(chases,    pitches)
        z_contact_pct = pct(z_contact, z_swings)
        bip    = int(g["IsBIP"].sum())
        avg_ev = safe_mean(g.loc[g["IsBIP"], "ExitSpeed"])
        avg_la = safe_mean(g.loc[g["IsBIP"], "Angle"])
        return pd.Series({
            "Pitches":    pitches, "Swings":  swings,  "Whiffs":     whiffs,
            "Whiff%":     whiff_pct, "Chases": chases,  "Chase%":     chase_pct,
            "Z-Contact%": z_contact_pct, "BIP": bip,
            "Avg EV":     avg_ev,  "Avg LA":  avg_la
        })

    by_pitch = (
        h.groupby("TaggedPitchType", dropna=True)
         .apply(hitter_group_summary)
         .reset_index()
         .rename(columns={"TaggedPitchType": "Pitch Type"})
         .sort_values("Pitches", ascending=False)
         .reset_index(drop=True)
    )

    show_tbl = by_pitch.copy()
    show_tbl["Whiff%"]     = show_tbl["Whiff%"].map(lambda x: "–" if pd.isna(x) else f"{x:.1f}%")
    show_tbl["Chase%"]     = show_tbl["Chase%"].map(lambda x: "–" if pd.isna(x) else f"{x:.1f}%")
    show_tbl["Z-Contact%"] = show_tbl["Z-Contact%"].map(lambda x: "–" if pd.isna(x) else f"{x:.1f}%")
    show_tbl["Avg EV"]     = show_tbl["Avg EV"].map(lambda x: "–" if pd.isna(x) else f"{x:.1f}")
    show_tbl["Avg LA"]     = show_tbl["Avg LA"].map(lambda x: "–" if pd.isna(x) else f"{x:.1f}")

    st.markdown("### By Pitch Type")
    st.dataframe(show_tbl, use_container_width=True)

    st.markdown("### Strike Zone Plot: Hard-Hit Swings (EV 90+), Chases, Whiffs, and Called Strikes")

    show_called_strikes = st.checkbox(
        "Include called strikes (taken strikes) on zone plot",
        value=True,
        help="When checked, pitches taken for a strike (in-zone, no swing) appear as diamonds on the plot."
    )

    hitter_pitch_types = [x for x in h["TaggedPitchType"].dropna().unique().tolist()]
    palette_h    = sns.color_palette("husl", len(hitter_pitch_types)) if hitter_pitch_types else []
    pitch_colors = dict(zip(hitter_pitch_types, palette_h))

    h["IsHardHit90"]        = h["IsSwing"] & h["IsBIP"] & (pd.to_numeric(h["ExitSpeed"], errors="coerce") >= 90)
    h["IsCalledStrikeZone"] = h["ZoneLocKnown"] & h["InZone"] & (~h["IsSwing"])
    h["IsWhiffChase"]       = h["IsWhiff"] & h["IsChase"]

    h["PlotEvent"] = np.where(
        h["IsWhiffChase"], "WhiffChase",
        np.where(h["IsWhiff"], "Whiff",
            np.where(h["IsChase"], "Chase",
                np.where(h["IsHardHit90"], "HardHit90",
                    np.where(h["IsCalledStrikeZone"] & show_called_strikes,
                             "CalledStrike", None)))))

    plot_df = h[h["PlotEvent"].notna()].copy()

    overall_row = pd.DataFrame([{
        "Pitch Type":  "Overall",
        "Pitches":     total_pitches,
        "Swings":      total_swings,
        "Whiffs":      total_whiffs,
        "Whiff%":      overall_whiff,
        "Chases":      total_chases,
        "Chase%":      overall_chase,
        "Z-Contact%":  overall_z_contact,
        "BIP":         int(h["IsBIP"].sum()),
        "Avg EV":      safe_mean(h.loc[h["IsBIP"], "ExitSpeed"]),
        "Avg LA":      safe_mean(h.loc[h["IsBIP"], "Angle"])
    }])

    summary_for_png = pd.concat([overall_row, by_pitch], ignore_index=True)

    desired_cols = [
        "Pitch Type", "Pitches", "Swings", "Whiffs", "Whiff%",
        "Chases", "Chase%", "Z-Contact%", "BIP", "Avg EV", "Avg LA"
    ]
    for c in desired_cols:
        if c not in summary_for_png.columns:
            summary_for_png[c] = np.nan
    summary_for_png = summary_for_png[desired_cols].copy()

    hand_col_h    = "PitcherThrows"
    hands_in_data = []
    if hand_col_h in h.columns:
        hands_in_data = sorted(
            h[hand_col_h].dropna().unique(),
            key=lambda x: {"Right": 0, "Left": 1}.get(x, 9)
        )

    def hand_split_metrics(subset: pd.DataFrame) -> dict:
        result = {}
        for hnd in hands_in_data:
            g     = subset[subset[hand_col_h] == hnd] if hand_col_h in subset.columns else pd.DataFrame()
            label = "vs R" if hnd == "Right" else "vs L"
            if g.empty:
                result[f"{label}\nWhiff%"] = np.nan
                result[f"{label}\nAvg EV"] = np.nan
                result[f"{label}\nAvg LA"] = np.nan
                result[f"{label}\nChase%"] = np.nan
            else:
                swings_h = int(g["IsSwing"].sum())
                whiffs_h = int(g["IsWhiff"].sum())
                result[f"{label}\nWhiff%"] = pct(whiffs_h, swings_h)
                result[f"{label}\nAvg EV"] = safe_mean(g.loc[g["IsBIP"], "ExitSpeed"])
                result[f"{label}\nAvg LA"] = safe_mean(g.loc[g["IsBIP"], "Angle"])
                result[f"{label}\nChase%"] = pct(int(g["IsChase"].sum()), len(g))
        return result

    hand_rows_h = [{"Pitch Type": "Overall", **hand_split_metrics(h)}]
    for pt in summary_for_png["Pitch Type"].iloc[1:]:
        sub_pt = h[h["TaggedPitchType"] == pt]
        hand_rows_h.append({"Pitch Type": pt, **hand_split_metrics(sub_pt)})

    hand_df_h = pd.DataFrame(hand_rows_h)

    hand_stat_cols_h = []
    for hnd in hands_in_data:
        label = "vs R" if hnd == "Right" else "vs L"
        hand_stat_cols_h += [
            f"{label}\nWhiff%",
            f"{label}\nAvg EV",
            f"{label}\nAvg LA",
            f"{label}\nChase%"
        ]
    hand_display_cols_h = ["Pitch Type"] + hand_stat_cols_h

    hand_df_h_fmt = hand_df_h[hand_display_cols_h].copy()
    for col in hand_stat_cols_h:
        if "Whiff%" in col or "Chase%" in col:
            hand_df_h_fmt[col] = hand_df_h_fmt[col].map(
                lambda x: "–" if pd.isna(x) else f"{x:.1f}%")
        else:
            hand_df_h_fmt[col] = hand_df_h_fmt[col].map(
                lambda x: "–" if pd.isna(x) else f"{x:.1f}")

    fig_combo = plt.figure(figsize=(14.0, 13.5), dpi=170)
    gs_combo  = gridspec.GridSpec(5, 1, height_ratios=[0.6, 0.95, 3.8, 0.20, 2.6], hspace=0.30)

    axh = fig_combo.add_subplot(gs_combo[0, 0])
    axk = fig_combo.add_subplot(gs_combo[1, 0])

    gs_zone_row = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_combo[2, 0],
        width_ratios=[0.05, 1.0, 0.38], wspace=0.05
    )
    axz    = fig_combo.add_subplot(gs_zone_row[0, 1])
    ax_leg = fig_combo.add_subplot(gs_zone_row[0, 2])
    ax_leg.axis("off")

    gs_tables_h = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_combo[4, 0],
        wspace=0.08, width_ratios=[1.15, 0.85]
    )
    ax_tbl_orig = fig_combo.add_subplot(gs_tables_h[0, 0])
    ax_tbl_hand = fig_combo.add_subplot(gs_tables_h[0, 1])

    axh.axis("off"); axk.axis("off")
    ax_tbl_orig.axis("off"); ax_tbl_hand.axis("off")

    axh.text(0, 0.65, selected_hitter, fontsize=26, weight="bold")
    axh.text(0.97, 0.60, f"Performance Summary\n{season_range}", fontsize=14, ha="right")

    hitter_kpi_labels = ["Chase%", "Z-Contact%", "Max EV", "Avg LA", "90th % EV"]
    hitter_kpi_values = [
        format_pct(overall_chase), format_pct(overall_z_contact),
        format_num(max_ev_overall, 1), format_num(avg_launch_angle, 1), format_num(ev_90th, 1)
    ]
    num_boxes = len(hitter_kpi_labels)
    for i, (lab, val) in enumerate(zip(hitter_kpi_labels, hitter_kpi_values)):
        x_left = i / num_boxes; width = 1 / num_boxes
        axk.text(x_left + width / 2, 0.72, lab, fontsize=11, ha="center", weight="bold")
        axk.add_patch(plt.Rectangle((x_left + 0.06, 0.18), width - 0.12, 0.42, fill=False, linewidth=1))
        axk.text(x_left + width / 2, 0.34, val, fontsize=12, ha="center")

    axz.axvline(-8.5, color="black", linestyle="--", linewidth=0.9)
    axz.axvline( 8.5, color="black", linestyle="--", linewidth=0.9)
    axz.axhline( 1.5, color="black", linestyle="--", linewidth=0.9)
    axz.axhline( 3.5, color="black", linestyle="--", linewidth=0.9)
    x_min, x_max = -20, 20; y_min, y_max = 0, 5.5
    marker_map = {"HardHit90": "x", "Chase": "o", "Whiff": "^", "WhiffChase": "s", "CalledStrike": "D"}

    if plot_df.empty:
        axz.text(0.5, 0.5, "No pitches meet the criteria for this hitter.",
                 ha="center", va="center", transform=axz.transAxes)
    else:
        plot_df["px_in"] = pd.to_numeric(plot_df["PlateLocSide"],  errors="coerce") * 12
        plot_df["pz_ft"] = pd.to_numeric(plot_df["PlateLocHeight"], errors="coerce")
        plot_df = plot_df[plot_df["px_in"].notna() & plot_df["pz_ft"].notna()].copy()
        for pt in hitter_pitch_types:
            sub_pt = plot_df[plot_df["TaggedPitchType"] == pt]
            if sub_pt.empty: continue
            for event, marker in marker_map.items():
                sub = sub_pt[sub_pt["PlotEvent"] == event]
                if sub.empty: continue
                if marker == "x": size, lw, edge = 95, 2.2, None
                else:             size, lw, edge = 65, 0.9, "black"
                axz.scatter(
                    np.clip(sub["px_in"].values, x_min, x_max),
                    np.clip(sub["pz_ft"].values, y_min, y_max),
                    s=size, marker=marker, color=pitch_colors.get(pt, "gray"),
                    edgecolor=edge, linewidth=lw, alpha=0.95
                )

    axz.set_xlim(x_min, x_max); axz.set_ylim(y_min, y_max)
    axz.set_title("Hitter Strike Zone Plot (Filtered Events)", fontsize=14, weight="bold")
    axz.set_xlabel("Plate Side (inches)"); axz.set_ylabel("Plate Height (ft)")
    axz.grid(alpha=0.25, linestyle="--")

    pitch_handles = [Patch(facecolor=pitch_colors[pt], edgecolor="black", label=pt)
                     for pt in hitter_pitch_types]
    leg1 = ax_leg.legend(handles=pitch_handles if pitch_handles else [],
                         title="Pitch Type (Color)", loc="upper left",
                         bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=9, title_fontsize=9)
    ax_leg.add_artist(leg1)

    event_handles = [
        Line2D([0], [0], marker="x", color="black", linestyle="None", markersize=10, label="Swing → EV 90+"),
        Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=8,  label="Chase"),
        Line2D([0], [0], marker="^", color="black", linestyle="None", markersize=8,  label="Whiff"),
        Line2D([0], [0], marker="s", color="black", linestyle="None", markersize=8,  label="Whiff + Chase"),
    ]
    if show_called_strikes:
        event_handles.append(Line2D([0], [0], marker="D", color="black", linestyle="None",
                                    markersize=8, label="Called Strike\n(In-Zone, No Swing)"))
    ax_leg.legend(handles=event_handles, title="Event (Shape)", loc="lower left",
                  bbox_to_anchor=(0.0, 0.0), frameon=False, fontsize=9, title_fontsize=9)

    table_display = summary_for_png.copy()
    table_display["Whiff%"]     = table_display["Whiff%"].map(lambda x: "–" if pd.isna(x) else f"{x:.1f}%")
    table_display["Chase%"]     = table_display["Chase%"].map(lambda x: "–" if pd.isna(x) else f"{x:.1f}%")
    table_display["Z-Contact%"] = table_display["Z-Contact%"].map(lambda x: "–" if pd.isna(x) else f"{x:.1f}%")
    table_display["Avg EV"]     = table_display["Avg EV"].map(lambda x: "–" if pd.isna(x) else f"{x:.1f}")
    table_display["Avg LA"]     = table_display["Avg LA"].map(lambda x: "–" if pd.isna(x) else f"{x:.1f}")
    for c in ["Pitches", "Swings", "Whiffs", "Chases", "BIP"]:
        table_display[c] = table_display[c].map(lambda v: "–" if pd.isna(v) else f"{int(v)}")

    ax_tbl_orig.text(0.5, 0.97, "Overall Summary",
                     ha="center", va="top", fontsize=11, weight="bold",
                     transform=ax_tbl_orig.transAxes)
    tbl_orig = ax_tbl_orig.table(
        cellText=table_display.values.tolist(),
        colLabels=list(table_display.columns),
        loc="center", cellLoc="center",
        bbox=[0.0, 0.0, 1.0, 0.82]
    )
    tbl_orig.auto_set_font_size(False)
    tbl_orig.auto_set_column_width(col=list(range(len(table_display.columns))))
    tbl_orig.set_fontsize(8.5)
    tbl_orig.scale(1.0, 1.3)

    ax_tbl_hand.text(0.5, 0.97, "Split by Pitcher Handedness",
                     ha="center", va="top", fontsize=11, weight="bold",
                     transform=ax_tbl_hand.transAxes)
    if hand_df_h_fmt.empty or not hand_stat_cols_h:
        ax_tbl_hand.text(0.5, 0.45, "No pitcher handedness\ndata available.",
                         ha="center", va="center", fontsize=9, transform=ax_tbl_hand.transAxes)
    else:
        tbl_hand = ax_tbl_hand.table(
            cellText=hand_df_h_fmt.values.tolist(),
            colLabels=list(hand_df_h_fmt.columns),
            loc="center", cellLoc="center",
            bbox=[0.0, 0.0, 1.0, 0.82]
        )
        tbl_hand.auto_set_font_size(False)
        tbl_hand.auto_set_column_width(col=list(range(len(hand_display_cols_h))))
        tbl_hand.set_fontsize(8.5)
        tbl_hand.scale(1.0, 1.3)

    st.pyplot(fig_combo)

    combo_buf = io.BytesIO()
    fig_combo.savefig(combo_buf, format="png", bbox_inches="tight")
    combo_buf.seek(0)

    st.download_button(
        "Download Hitter Heatmap + Summary Table (PNG)",
        data=combo_buf,
        file_name=f"{selected_hitter.replace(' ', '_')}_heatmap_summary.png",
        mime="image/png"
    )

    st.markdown("### Download")
    out = by_pitch.copy()
    out["Whiff%"]     = out["Whiff%"].round(3)
    out["Chase%"]     = out["Chase%"].round(3)
    out["Z-Contact%"] = out["Z-Contact%"].round(3)
    out["Avg EV"]     = out["Avg EV"].round(3)
    out["Avg LA"]     = out["Avg LA"].round(3)
    csv_bytes = out.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Hitter Summary (CSV)",
        data=csv_bytes,
        file_name=f"{selected_hitter.replace(' ', '_')}_hitter_summary.csv",
        mime="text/csv"
    )
