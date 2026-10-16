import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import io
import re
import difflib
from datetime import datetime
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# page setup
st.set_page_config(layout="wide", page_title="Pitching + Hitting Report Builder")
st.title("⚾ Pitching + Hitting Report Builder")
st.write("Upload CSV(s) → filter by date → pick ANY pitcher or hitter in the file.")

# helper: pull the date out of a file name
def extract_date(filename: str):
    match = re.search(r"(\d{1,2})[_.-](\d{1,2})[_.-](\d{2,4})", filename)
    if match:
        month, day, year = match.groups()
        if len(year) == 2:
            year = "20" + year
        try:
            return datetime(int(year), int(month), int(day))
        except ValueError:
            return None
    return None

# helper: turn a file name into a Home vs Away date label
def parse_matchup(filename: str) -> str:
    base = re.sub(r"\s*\(\d+\)\s*", " ", str(filename))
    base = re.sub(r"\.(csv|xlsx)$", "", base, flags=re.I).strip()
    dt = extract_date(base)
    date_str = f"{dt.month}/{dt.day}/{dt:%y}" if dt else ""

    m = re.search(r"[AP]M\s*-\s*(.+?)\s*@\s*(.+?)\s*$", base)
    if m:
        away = re.sub(r"\s*\[[^\]]*\]\s*$", "", m.group(1)).strip()
        home = re.sub(r"\s*\[[^\]]*\]\s*$", "", m.group(2)).strip()
        teams = [t for t in (away, home) if t and t.lower() != "null"]
        if len(teams) == 2:
            is_gu = lambda s: "georgetown" in s.lower() or s.lower() in ("gu", "gtown", "gu 2025", "gu 2026")
            first, second = (home, away) if (is_gu(home) or not is_gu(away)) else (away, home)
            return f"{first} vs {second} {date_str}".strip()

    # TrackMan report workbooks are already named like "UVA vs Georgetown - 9.27.26 ..."
    if re.search(r"\bvs\.?\s", base, flags=re.I):
        return base

    if date_str:
        return f"Session {date_str}"
    return base

# helper: one label covering every uploaded file
def matchup_label(source_files) -> str:
    labels = sorted({parse_matchup(s) for s in source_files if pd.notna(s)})
    if not labels:
        return "session"
    if len(labels) <= 3:
        return " + ".join(labels)
    return " + ".join(labels[:3]) + f" +{len(labels) - 3} more"

# helper: make text safe for a file name
def slugify(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_") or "session"

# helpers: small math and formatting utilities
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

# helper: flag any row that mentions a foul
def add_foul_flag(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
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

# helper: flag pitches inside the strike zone
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

# helper: label each ball in play as GB, LD, or FB
def add_batted_ball_flags(df: pd.DataFrame) -> pd.DataFrame:
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

# helper: point switch hitters to the side they bat from
def resolve_batter_side(df: pd.DataFrame) -> pd.DataFrame:
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

        if throws == "Right":
            return "Left"
        if throws == "Left":
            return "Right"
        return np.nan

    df["BatterSideResolved"] = df.apply(_resolve, axis=1)
    return df

# helper: read one upload and say which platform it came from
# .xlsx = TrackMan report workbook (Raw Data tab); .csv with yt_ columns = YakkerTech; other .csv = TrackMan export
def read_upload(f):
    if f.name.lower().endswith(".xlsx"):
        xl = pd.ExcelFile(f)
        if "Raw Data" not in xl.sheet_names:
            raise ValueError(f"{f.name}: no 'Raw Data' tab found, so it was skipped.")
        return pd.read_excel(xl, sheet_name="Raw Data"), "TrackMan"
    temp = pd.read_csv(f, low_memory=False)
    source = "YakkerTech" if any(str(c).startswith("yt_") for c in temp.columns) else "TrackMan"
    return temp, source

# helper: fall back to the Date column when the file name has no date
def date_from_column(temp: pd.DataFrame):
    if "Date" not in temp.columns:
        return None
    dates = pd.to_datetime(temp["Date"], errors="coerce").dropna()
    return dates.mode().iloc[0].to_pydatetime() if not dates.empty else None

# helper: "Pauley, Gibson" -> "Gibson Pauley", "Burnett," -> "Burnett"
def flip_name(name):
    if pd.isna(name) or "," not in str(name):
        return name
    last, first = [s.strip() for s in str(name).split(",", 1)]
    return f"{first} {last}".strip()

# TrackMan report columns computed with their own definitions (different zone, 95 mph hard-hit, etc.)
# The app recomputes all of these itself, so they're dropped to keep both platforms on one definition.
TM_HELPER_COLS = re.compile(r"^(Is[A-Z]\w*|InZone|CountsAsPA|TotalBases|PAKey)$|_Clean$")

TM_PITCH_TYPES = {
    "ChangeUp": "Changeup",
    "FourSeamFastBall": "Fastball",
    "Four-Seam": "Fastball",
    "TwoSeamFastBall": "Sinker",
}

# helper: put TrackMan column names and values into the YakkerTech format the app uses
def normalize_trackman(temp: pd.DataFrame) -> pd.DataFrame:
    temp = temp.drop(columns=[c for c in temp.columns if TM_HELPER_COLS.search(str(c))])
    obj_cols = temp.select_dtypes(include=["object", "string"]).columns
    temp[obj_cols] = temp[obj_cols].replace("Undefined", np.nan)
    for old, new in {"TaggedHitType": "HitType", "Half": "Top/Bottom"}.items():
        if old in temp.columns and new not in temp.columns:
            temp = temp.rename(columns={old: new})
    if "TaggedPitchType" in temp.columns:
        temp["TaggedPitchType"] = temp["TaggedPitchType"].replace(TM_PITCH_TYPES)
    return temp

# helper: one pitch-call vocabulary for every platform
# FoulBallNotFieldable / FoulBallFieldable / FoulTip -> Foul, BallinDirt / BallIntentional -> BallCalled
def normalize_pitch_calls(temp: pd.DataFrame) -> pd.DataFrame:
    if "PitchCall" in temp.columns:
        call = temp["PitchCall"].astype(str)
        temp["PitchCall"] = (
            temp["PitchCall"]
            .mask(call.str.startswith("Foul"), "Foul")
            .mask(call.str.startswith("Ball"), "BallCalled")
        )
    return temp

# readings outside these ranges are tracking dropouts (e.g. a 45 mph "fastball"),
# treated as missing on every platform
PLAUSIBLE_RANGES = {"RelSpeed": (60, 105), "Extension": (0, 10)}

def screen_outliers(df: pd.DataFrame):
    n_screened = 0
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        bad  = vals.notna() & ((vals < lo) | (vals > hi))
        n_screened += int(bad.sum())
        df[col] = vals.mask(bad)
    return df, n_screened

# helper: label which platform(s) a slice of data came from
def platform_label(frame: pd.DataFrame) -> str:
    sources = sorted(frame["Source"].dropna().unique()) if "Source" in frame.columns else []
    return " + ".join(sources) if sources else "–"

# Georgetown players, so last-name-only TrackMan entries can be matched even when no
# YakkerTech file is uploaded. Built from the 2027 fall YakkerTech CSVs (team labels
# Fall2026PracticeGT / Gtown fall2024 roster / Georgetown). Add new players here as "First Last".
GU_ROSTER = [
    "Addison Lee", "Amman Dewberry", "Andrew Jergins", "Ashton Arroyo", "Ashton Seymore",
    "Brady Holland", "Brandon Davis", "Braxton Templin", "Charlie Hendrix", "Christian Hamilton",
    "Collin Sullivan", "Connor Peek", "Dan Ockerman", "Davis Hanson", "Dixon Monk",
    "Dom Cafferillo", "Drew Rybinski", "Dylan Larkins", "Gibson Pauley", "J.w. Riggins",
    "Jack Volo", "Jordan Kahn", "Julius Pfau", "Kavi Caster", "Khalon Davis",
    "Luke Jackman", "Matthew Sapienza", "Max Andrews", "Max Whitmer", "Mike Poncini",
    "Nelson Hayes", "Roman Tozzi", "Rowin O'connor", "Sebastian Burnett", "Shane Keup",
    "Simon Tesfaye", "Travis Ilitch",
]

# team labels Georgetown shows up under (TrackMan code GEO_HOY; YakkerTech practice/roster names).
# Deliberately not plain "geo": George Mason (geo01) and GW (geo02) would match.
GU_TEAM_PATTERN = r"GEO_HOY|georgetown|gtown|practicegt|^gu\b"

ROLE_TEAM_COLS = [("Pitcher", "PitcherTeam"), ("Batter", "BatterTeam")]

def is_gu_team(team: pd.Series) -> pd.Series:
    return team.astype(str).str.contains(GU_TEAM_PATTERN, case=False, regex=True)

# helper: is this file a Georgetown intrasquad (GU is both the home and away team)?
def is_intrasquad(g: pd.DataFrame, src_file: str) -> bool:
    if "HomeTeam" in g.columns and "AwayTeam" in g.columns:
        home, away = g["HomeTeam"].dropna(), g["AwayTeam"].dropna()
        if not home.empty and not away.empty:
            return bool(is_gu_team(home).any() and is_gu_team(away).any())
    m = re.search(r"[AP]M\s*-\s*(.+?)\s*@\s*(.+?)\s*(\(\d+\))?\.\w+$", str(src_file))
    return bool(m) and all(re.search(GU_TEAM_PATTERN, t, flags=re.I) for t in m.group(1, 2))

# helper: which pitcher rows / batter rows belong to Georgetown players.
# Player team labels are unreliable (YakkerTech sometimes attaches the wrong team profile, e.g. a GU
# hitter tagged "Augusta", or an old school's profile), so in order:
#   1. GU intrasquad file              -> every pitcher and batter is GU
#   2. GU's batting half can be told   -> batters in that half and pitchers in the other half are GU,
#      decided per player by majority (from where most GU-labeled hitters bat). This is the only thing
#      that overrides the Georgetown list, so an opponent who shares a GU player's name stays out.
#   3. otherwise                       -> GU if the player is on the Georgetown list (roster + anyone
#      rule 1 or 2 confirmed as GU anywhere in the upload), whatever their team label says; players
#      not on the list go by the team label (unlabeled rows count if they're GU-labeled elsewhere)
def gu_row_masks(df: pd.DataFrame):
    labeled, label_names = {}, set()
    for col, team_col in ROLE_TEAM_COLS:
        team = df[team_col] if team_col in df.columns else pd.Series(np.nan, index=df.index)
        labeled[col] = is_gu_team(team)
        label_names |= set(df.loc[labeled[col], col].dropna())

    # rules 1 and 2: True / False where the game itself shows the side, NaN where it can't
    decided = {col: pd.Series(np.nan, index=df.index, dtype=object) for col, _ in ROLE_TEAM_COLS}
    half = df["Top/Bottom"] if "Top/Bottom" in df.columns else pd.Series(np.nan, index=df.index)
    for src_file, idx in df.groupby("SourceFile").groups.items():
        g = df.loc[idx]
        if is_intrasquad(g, src_file):
            for col, _ in ROLE_TEAM_COLS:
                named = g.index[g[col].notna()]
                decided[col].loc[named] = True
            continue
        h = half[idx]
        gu_halves = h[labeled["Batter"][idx] & h.notna()]
        if len(gu_halves) < 5:
            continue
        top_share = float((gu_halves == "Top").mean())
        if 0.25 < top_share < 0.75:
            continue
        gu_bat_half = "Top" if top_share >= 0.75 else "Bottom"
        # decide per player by majority, since a stray pitch can be stamped with the wrong half
        known = idx[h.notna()]
        for col, gu_side in [("Batter", True), ("Pitcher", False)]:
            in_gu_half = (half[known] == gu_bat_half) == gu_side
            share  = in_gu_half.groupby(df.loc[known, col]).mean()
            vote   = g[col].map(share)
            decide = vote.index[vote.notna()]
            decided[col].loc[decide] = (vote[decide] >= 0.5).values

    # the Georgetown list: roster + everyone the games themselves confirmed as GU
    gu_list = set(GU_ROSTER)
    for col, _ in ROLE_TEAM_COLS:
        gu_list |= set(df.loc[decided[col].eq(True), col].dropna())

    masks = []
    for col, team_col in ROLE_TEAM_COLS:
        team = df[team_col] if team_col in df.columns else pd.Series(np.nan, index=df.index)
        fallback = labeled[col] | df[col].isin(gu_list) | (team.isna() & df[col].isin(label_names))
        masks.append(decided[col].where(decided[col].notna(), fallback).astype(bool))
    return masks[0], masks[1]

# helper: one comparable key per team ("GU" for every Georgetown label, opponents without the [code])
def team_key(team) -> str:
    t = str(team)
    if re.search(GU_TEAM_PATTERN, t, flags=re.I):
        return "GU"
    return re.sub(r"\s*\[.*$", "", t).strip().lower()

# helper: when one short name has more than one player ID in the same file (raw TrackMan
# exports carry BatterId / PitcherId), tag each as "Davis [ID 1234]" so they match separately
ID_TAG = re.compile(r" \[ID [^\]]*\]$")

def base_name(name: str) -> str:
    return ID_TAG.sub("", str(name))

def split_by_ids(df: pd.DataFrame) -> pd.DataFrame:
    for col, id_col in [("Pitcher", "PitcherId"), ("Batter", "BatterId")]:
        if id_col not in df.columns:
            continue
        short = df[col].notna() & ~df[col].astype(str).str.contains(" ") & df[id_col].notna()
        sub   = df[short]
        if sub.empty:
            continue
        n_ids = sub.groupby(["SourceFile", col])[id_col].transform("nunique")
        idx   = sub.index[n_ids > 1]
        ids   = df.loc[idx, id_col].map(lambda v: str(int(v)) if isinstance(v, float) and v.is_integer() else str(v))
        df.loc[idx, col] = df.loc[idx, col] + " [ID " + ids + "]"
    return df

# helper: one entry per plate appearance (batters) or per inning (pitchers) for a name in one
# file, so two players sharing a short name in the same game can be split by hand
def split_units(g: pd.DataFrame, col: str, name: str):
    rows = g[g[col] == name].sort_values("PitchNo")
    keys = ["Inning", "Top/Bottom", "PAofInning"] if col == "Batter" else ["Inning", "Top/Bottom"]
    keys = [k for k in keys if k in rows.columns] or ["PitchNo"]

    def _fmt(v):
        return "?" if pd.isna(v) else (str(int(v)) if isinstance(v, (int, float)) and float(v).is_integer() else str(v))

    units = []
    for key, u in rows.groupby(keys, dropna=False, sort=False):
        parts = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        label = f"Inning {_fmt(parts.get('Inning'))}"
        if pd.notna(parts.get("Top/Bottom")):
            label += f" ({parts['Top/Bottom']})"
        if col == "Batter":
            if "PAofInning" in parts:
                label += f" · PA {_fmt(parts['PAofInning'])}"
            label += f" · vs {', '.join(u['Pitcher'].dropna().unique()) or '?'}"
            last = u.iloc[-1]
            result = next((last[c] for c in ["PlayResult", "KorBB", "PitchCall"]
                           if c in u.columns and pd.notna(last[c])), None)
            if result is not None:
                label += f" · {result}"
        else:
            label += f" · {len(u)} pitches · {u['Batter'].nunique()} batters"
        units.append((str(key), label, u.index))
    return units

# helper: table of names to review, one row per game file + name, since the same short
# name ("Davis") can be a different player in different games
KEEP_NAME  = "(keep as is)"
SPLIT_NAME = "More than one player…"
NAME_TABLE_COLS = ["File", "Name in file", "Role", "Reason", "Platform", "Match to"]

# helper: two full names that are probably one player, e.g. a nickname ("Dom" / "Dominick
# Cafferillo") or a tagging typo ("Drew Raybinski" / "Drew Rybinski")
def similar_names(a: str, b: str) -> bool:
    a_l, b_l = a.lower(), b.lower()
    if a_l == b_l:
        return False
    fa, la = a_l.split()[0], a_l.split()[-1]
    fb, lb = b_l.split()[0], b_l.split()[-1]
    if la == lb and fa != fb and min(len(fa), len(fb)) >= 2 and (fa.startswith(fb) or fb.startswith(fa)):
        return True
    # same first name, misspelled last name ("Drew Raybinish" / "Drew Rybinski")
    if fa == fb and la != lb and difflib.SequenceMatcher(None, la, lb).ratio() >= 0.75:
        return True
    sm = difflib.SequenceMatcher(None, a_l, b_l)
    return sm.real_quick_ratio() >= 0.88 and sm.quick_ratio() >= 0.88 and sm.ratio() >= 0.88

# helper: map each "possible duplicate" full name to the spelling it should merge into.
# Only names that share a team are compared, so two different players on different teams never pair up.
# The roster spelling wins, then the more common spelling in the upload, then alphabetical.
def find_duplicate_targets(full: list, name_teams: dict, name_counts: pd.Series) -> dict:
    def rank(x):
        return (x in GU_ROSTER, int(name_counts.get(x, 0)), [-ord(c) for c in x])

    by_team = {}
    for n in full:
        for t in (name_teams.get(n) or {"?"}):
            by_team.setdefault(t, []).append(n)

    # group chains of similar spellings ("Raybinish" ~ "Raybinski" ~ "Rybinski") into one cluster
    parent = {}
    def root(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for group in by_team.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if similar_names(a, b):
                    parent[root(a)] = root(b)

    clusters = {}
    for n in parent:
        clusters.setdefault(root(n), []).append(n)
    targets = {}
    for members in clusters.values():
        best = max(members, key=rank)
        targets.update({m: best for m in members if m != best})
    return targets

def build_name_table(df: pd.DataFrame, show_all_file=None, gu_only: bool = False):
    upload_full = {
        n for col in ["Pitcher", "Batter"] for n in df[col].dropna().unique() if " " in base_name(n)
    }
    full_names  = sorted(upload_full | set(GU_ROSTER))

    # every team each name shows up for, and how often it appears
    name_teams = {n: {"GU"} for n in GU_ROSTER}
    for col, team_col in ROLE_TEAM_COLS:
        if team_col in df.columns:
            pairs = df[[col, team_col]].dropna().drop_duplicates()
            for n, t in zip(pairs[col], pairs[team_col]):
                name_teams.setdefault(n, set()).add(team_key(t))
    name_counts = pd.concat([df["Pitcher"], df["Batter"]]).value_counts()
    dup_target  = find_duplicate_targets(full_names, name_teams, name_counts)

    gu_pitch, gu_bat = gu_row_masks(df)

    rows = []
    for src_file, g in df.groupby("SourceFile", sort=False):
        roles = {}
        for col, _ in ROLE_TEAM_COLS:
            for n in g[col].dropna().unique():
                roles.setdefault(n, set()).add(col)
        gu_names = set(g.loc[gu_pitch[g.index], "Pitcher"].dropna()) | set(g.loc[gu_bat[g.index], "Batter"].dropna())

        for n in sorted(roles):
            if gu_only and n not in gu_names:
                continue
            last_only = " " not in base_name(n)
            id_split  = n != base_name(n)
            # roster names are only suggested for players on a Georgetown team row
            candidates = (upload_full | set(GU_ROSTER)) if n in gu_names else upload_full
            # same name with different capitalization, e.g. "J.W. Riggins" vs "J.w. Riggins"
            # (roster spelling wins; otherwise the alphabetically first, so two names never swap)
            recased = [
                m for m in candidates
                if m != n and m.lower() == n.lower()
                and (m in GU_ROSTER or (n not in GU_ROSTER and m < n))
            ]
            dup_of = dup_target.get(n) if not (last_only or id_split) else None
            if not (last_only or recased or dup_of or src_file == show_all_file):
                continue
            last = base_name(n).split()[-1].lower()
            # collapse nickname/typo variants first, so "Cafferillo" sees one player, not two spellings
            same_last = {
                dup_target.get(m, m) for m in candidates
                if m != n and m.split()[-1].lower() == last
            }
            if recased:
                suggestion, reason = dup_target.get(recased[0], recased[0]), "capitalization"
            elif id_split:
                # two IDs share this name, so a single suggestion would merge them back together
                suggestion, reason = KEEP_NAME, "two player IDs"
            elif dup_of:
                suggestion, reason = dup_of, "possible same player"
            elif last_only:
                suggestion = next(iter(same_last)) if len(same_last) == 1 else KEEP_NAME
                reason     = "last name only"
            else:
                suggestion, reason = KEEP_NAME, ""
            rows.append({
                "File":         src_file,
                "Name in file": n,
                "Role":         " / ".join(sorted(roles[n])),
                "Reason":       reason,
                "Platform":     g["Source"].iat[0],
                "Match to":     suggestion,
            })
    return pd.DataFrame(rows, columns=NAME_TABLE_COLS), full_names

# helper: shrink a table font until it fits
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

# upload one or more YakkerTech / TrackMan files
uploaded_files = st.file_uploader(
    "Upload one or more files (YakkerTech CSV, TrackMan CSV, or TrackMan report .xlsx with a 'Raw Data' tab)",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.stop()

df_list      = []
file_dates   = []
file_sources = []

# read each file and tag it with its name, date, and platform
for f in uploaded_files:
    try:
        temp, source = read_upload(f)
    except ValueError as e:
        st.warning(str(e))
        continue

    if source == "TrackMan":
        temp = normalize_trackman(temp)
    temp = normalize_pitch_calls(temp)

    session_date = extract_date(f.name) or date_from_column(temp)
    temp["SourceFile"]  = f.name
    temp["SessionDate"] = session_date
    temp["Source"]      = source

    for col in ["Pitcher", "Batter"]:
        if col in temp.columns:
            temp[col] = (
                temp[col]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .replace("nan", np.nan)
                .map(flip_name)
            )
    df_list.append(temp)
    file_dates.append(session_date)
    file_sources.append(source)

if not df_list:
    st.stop()

df = pd.concat(df_list, ignore_index=True)
source_counts = pd.Series(file_sources).value_counts()
st.success(
    f"Loaded {len(df_list)} file(s) "
    f"({', '.join(f'{n} {s}' for s, n in source_counts.items())}) "
    f"with {len(df):,} total rows."
)

# stop early if a required column is missing
needed = [
    "Pitcher", "Batter",
    "TaggedPitchType", "PitchCall", "PitchNo",
    "RelSpeed", "HorzBreak", "InducedVertBreak",
    "PlateLocSide", "PlateLocHeight",
    "ExitSpeed", "Angle"
]
require_cols(df, needed, label="for this app")

# match last-name-only entries (common in TrackMan files) to full names found in the uploads
# Georgetown-only mode: limits name matching and the pitcher / hitter lists to GU players, so an
# opponent who shares a name with a GU player (or with another opponent) is never pooled in
gu_only = st.toggle(
    "Georgetown players only",
    value=False,
    help="Only review Georgetown names below, and only list Georgetown pitchers / hitters (using "
         "their Georgetown rows) in the report tabs. Opponents with the same name are left out."
)

df = split_by_ids(df)
names_preview, _ = build_name_table(df, gu_only=gu_only)

st.markdown(
    f"#### Name matching ({len(names_preview)} name(s) to review across "
    f"{names_preview['File'].nunique()} game file(s))"
)
st.caption(
    "Some files tag players by last name only (e.g. \"Burnett,\"). Open a game below and pick the full "
    "name each one should count as; the two are merged into one player everywhere in the app. Matches "
    "are set per game file, so the same short name can point to different players in different games. "
    "If two players share a name in the same game, pick \"More than one player…\" and assign each "
    "plate appearance (or inning, for pitchers). Click a box and type to search. Suggestions come "
    "from the Georgetown roster and full names found in the uploaded files."
)
show_all_file = None
if st.checkbox("Show every name in one game file (to merge any two spellings)", value=False):
    show_all_file = st.selectbox(
        "Game file:",
        list(df["SourceFile"].dropna().unique()),
        format_func=lambda s: f"{parse_matchup(s)}  ({s})",
    )
names_tbl, full_names = build_name_table(df, show_all_file, gu_only=gu_only)

name_map     = {}   # {game file: {name in file: full name}}
split_assign = []   # [(row index, column, full name)] for names split by plate appearance / inning
if names_tbl.empty:
    st.caption("Nothing to review.")

for src_file, file_rows in names_tbl.groupby("File", sort=False):
    # label is built from the suggestions only, so it doesn't change (and collapse) as picks are made
    n_open = int((file_rows["Match to"] == KEEP_NAME).sum())
    status = f"{n_open} without a suggestion" if n_open else "all suggested"
    with st.expander(
        f"{parse_matchup(src_file)} · {file_rows['Platform'].iat[0]} · "
        f"{len(file_rows)} name(s), {status}"
    ):
        st.caption(src_file)
        g = df[df["SourceFile"] == src_file]
        for _, r in file_rows.iterrows():
            name = r["Name in file"]
            options = [KEEP_NAME, SPLIT_NAME] + [n for n in full_names if n != name]
            c1, c2, c3 = st.columns([2, 2, 3])
            c1.write(name)
            c2.caption(f"{r['Role']} · {r['Reason']}" if r["Reason"] else r["Role"])
            choice = c3.selectbox(
                f"Same player as {name} in {src_file}",
                options,
                index=options.index(r["Match to"]) if r["Match to"] in options else 0,
                key=f"name_map::{src_file}::{name}",
                label_visibility="collapsed",
            )
            if choice == SPLIT_NAME:
                unit_options = [KEEP_NAME] + [n for n in full_names if n != name]
                for role in r["Role"].split(" / "):
                    for unit_key, unit_label, idx in split_units(g, role, name):
                        s1, s2 = st.columns([4, 3])
                        s1.caption(f"↳ {role}: {unit_label}")
                        pick = s2.selectbox(
                            f"{name} {role} {unit_label}",
                            unit_options,
                            key=f"split::{src_file}::{role}::{name}::{unit_key}",
                            label_visibility="collapsed",
                        )
                        if pick != KEEP_NAME:
                            split_assign.append((idx, role, pick))
            elif choice != KEEP_NAME:
                name_map.setdefault(src_file, {})[name] = choice

n_merges = sum(len(m) for m in name_map.values())
if n_merges or split_assign:
    st.success(
        f"{n_merges} name merge(s) applied across {len(name_map)} game file(s)"
        + (f", plus {len(split_assign)} split assignment(s)." if split_assign else ".")
    )

for src_file, file_map in name_map.items():
    in_file = df["SourceFile"] == src_file
    for col in ["Pitcher", "Batter"]:
        df.loc[in_file, col] = df.loc[in_file, col].replace(file_map)
for idx, col, full in split_assign:
    df.loc[idx, col] = full

# treat physically impossible tracking readings as missing (same rule for every platform)
df, n_screened = screen_outliers(df)
if n_screened:
    st.caption(
        f"{n_screened} tracking reading(s) outside plausible ranges were treated as missing "
        f"(velo outside {PLAUSIBLE_RANGES['RelSpeed'][0]}–{PLAUSIBLE_RANGES['RelSpeed'][1]} mph, "
        f"extension outside {PLAUSIBLE_RANGES['Extension'][0]}–{PLAUSIBLE_RANGES['Extension'][1]} ft)."
    )

# add every derived column the app needs
df = add_foul_flag(df)
df = add_zone_flags(df)
df = add_batted_ball_flags(df)
df = resolve_batter_side(df)

# figure out the overall date range
valid_dates  = [d for d in file_dates if d is not None]
season_range = (
    f"{min(valid_dates):%Y-%m-%d} → {max(valid_dates):%Y-%m-%d}"
    if valid_dates else "Unknown Date Range"
)

# let the user filter by session date
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

# let the user filter by platform when both are uploaded
available_platforms = sorted(df["Source"].dropna().unique())
if len(available_platforms) > 1:
    selected_platforms = st.multiselect(
        "Filter by platform:",
        options=available_platforms,
        default=available_platforms,
        help="Velo, spin, extension, etc. read slightly differently on each system."
    )
    df = df[df["Source"].isin(selected_platforms)]

# label used to tag and name the CSV exports
game_label = st.text_input(
    "Game / session label (tags every exported CSV and its filename):",
    value=matchup_label(df["SourceFile"].dropna().unique()),
    help="Auto-detected from the file name(s). Edit it if the opponent didn't come through."
)
label_slug = slugify(game_label)

# rows each tab draws from: with Georgetown-only on, a pitcher's GU pitching rows / a hitter's GU batting rows
gu_pitch_rows, gu_bat_rows = gu_row_masks(df)
pitch_df = df[gu_pitch_rows] if gu_only else df
bat_df   = df[gu_bat_rows]   if gu_only else df

# two tabs: pitcher report and hitter dashboard
tab_pitcher, tab_hitter = st.tabs(["🎯 Pitcher Report", "🏏 Hitter Dashboard"])

# PITCHER REPORT TAB
with tab_pitcher:
    st.subheader("Pitcher Report Generator (All Players)")

    pitchers = sorted(pitch_df["Pitcher"].dropna().unique())
    if not pitchers:
        st.warning("No pitchers found in the dataset.")
        st.stop()

    selected_pitcher = st.selectbox("Select a pitcher:", pitchers)

    p = pitch_df[pitch_df["Pitcher"] == selected_pitcher].copy()
    if p.empty:
        st.warning("No data for this pitcher.")
        st.stop()

    p = p[p["TaggedPitchType"].notna()]
    p = p[p["RelSpeed"].notna()]
    # keep sessions in order: date, then file, then pitch number within the file
    pitch_order = ["SessionDate", "SourceFile", "PitchNo"]
    p = p.sort_values(pitch_order).reset_index(drop=True)
    p["PitchNo"] = range(1, len(p) + 1)

    p["IsCalledStrike"]   = p["PitchCall"].eq("StrikeCalled")
    p["IsSwingingStrike"] = p["PitchCall"].eq("StrikeSwinging")

    all_pitch_types = p["TaggedPitchType"].dropna().unique().tolist()

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

    # keep only the selected pitch types
    p_plot = p[p["TaggedPitchType"].isin(selected_pitch_types)].copy()
    p_plot = p_plot.sort_values(pitch_order).reset_index(drop=True)
    p_plot["PitchNo"] = range(1, len(p_plot) + 1)

    # numbers for the summary strip at the top of the report
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

    for opt in ["SpinRate", "HorzApprAngle", "VertApprAngle"]:
        if opt not in p_plot.columns:
            p_plot[opt] = np.nan

    # pitch type summary table (left table on the PNG)
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

    # same stats split by batter handedness (right table on the PNG)
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
        else:
            hand_df_p_fmt[col] = hand_df_p_fmt[col].map(
                lambda x: "–" if pd.isna(x) else f"{x:.2f}")

    # draw the pitcher report figure
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
    ax_header.text(0.78, 0.05, f"Data: {platform_label(p_plot)}", fontsize=11, ha="right")

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

    # offer the figure as a PNG download
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)

    st.download_button(
        "Download Pitching Report as PNG",
        buffer,
        f"{selected_pitcher.replace(' ', '_')}_report.png",
        "image/png"
    )

    # build a CSV with exactly the numbers shown on the PNG
    def _type_max_velo(pt):
        v = pd.to_numeric(
            p_plot.loc[p_plot["TaggedPitchType"] == pt, "RelSpeed"], errors="coerce"
        )
        return float(v.max()) if v.notna().any() else np.nan

    report_df = pd.DataFrame(rows, columns=col_labels_left)
    report_df.insert(0, "Pitch Type", row_labels)
    report_df.insert(3, "Max Velo", [_type_max_velo(pt) for pt in row_labels])

    hand_num = hand_df_p.copy()
    hand_num.columns = [c.replace("\n", " ") for c in hand_num.columns]
    report_df = report_df.merge(hand_num, on="Pitch Type", how="left")

    overall_out = {c: np.nan for c in report_df.columns}
    overall_out["Pitch Type"] = "Overall"
    overall_out["Count"]    = total_pitches
    overall_out["Max Velo"] = max_velo
    overall_out["PZR%"]     = overall_pzr
    overall_out["GB%"]      = overall_gb_pct
    overall_out["LD%"]      = overall_ld_pct
    overall_out["FB%"]      = overall_fb_pct
    report_df = pd.concat([pd.DataFrame([overall_out]), report_df], ignore_index=True)

    round_map = {
        "Count": 0, "Velo": 1, "Max Velo": 1, "IVB": 1, "HB": 1, "Spin": 0,
        "HAA": 2, "VAA": 2, "PZR%": 1, "CSW%": 1, "GB%": 1, "LD%": 1, "FB%": 1,
        "Avg EV": 1, "GB/FB": 2,
    }
    for c in report_df.columns:
        if c == "Pitch Type":
            continue
        d = round_map.get(c, 2 if c.endswith("GB/FB") else 1)
        report_df[c] = pd.to_numeric(report_df[c], errors="coerce").round(d)
    report_df["Count"] = report_df["Count"].astype("Int64")
    report_df.insert(0, "Game", game_label)
    report_df.insert(1, "Player", selected_pitcher)
    report_df.insert(2, "Data", platform_label(p_plot))

    st.download_button(
        "Download Report Data (CSV)",
        report_df.to_csv(index=False).encode("utf-8"),
        f"{label_slug}_{slugify(selected_pitcher)}_pitching.csv",
        "text/csv"
    )

# HITTER DASHBOARD TAB
with tab_hitter:
    st.subheader("Hitter Whiff% + Exit Velocity (EV) + Chase% (All Players)")

    SWING_CALLS   = {"StrikeSwinging", "Foul", "InPlay"}
    WHIFF_CALLS   = {"StrikeSwinging"}
    CONTACT_CALLS = {"Foul", "InPlay"}

    hitters = sorted(bat_df["Batter"].dropna().unique())
    if not hitters:
        st.warning("No hitters found in the dataset.")
        st.stop()

    selected_hitter = st.selectbox("Select a hitter:", hitters)

    h = bat_df[bat_df["Batter"] == selected_hitter].copy()
    if h.empty:
        st.warning("No data for this hitter.")
        st.stop()

    h = h[h["TaggedPitchType"].notna()]
    h = h[h["PitchCall"].notna()]

    # per pitch swing, whiff, chase, and contact flags
    h["IsSwing"]   = h["PitchCall"].isin(SWING_CALLS)
    h["IsWhiff"]   = h["PitchCall"].isin(WHIFF_CALLS)
    h["IsContact"] = h["PitchCall"].isin(CONTACT_CALLS)

    h["IsBIP"]      = h["ExitSpeed"].notna() & (~h["RowHasFoul"])
    h["IsChase"]    = h["IsSwing"] & (h["ZoneLocKnown"]) & (~h["InZone"])
    h["IsZSwing"]   = h["IsSwing"]   & (h["ZoneLocKnown"]) & (h["InZone"])
    h["IsZContact"] = h["IsContact"] & (h["ZoneLocKnown"]) & (h["InZone"])

    # overall hitter totals
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

    # KPI row at the top of the tab
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

    # per pitch type summary table
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
    st.dataframe(show_tbl, width="stretch")

    # strike zone plot of notable pitches
    st.markdown("### Strike Zone Plot: Hard-Hit Swings (EV 90+), Chases, Whiffs, and Called Strikes")

    show_called_strikes = st.checkbox(
        "Include called strikes (taken strikes) on zone plot",
        value=True,
        help="When checked, pitches taken for a strike (in-zone, no swing) appear as diamonds on the plot."
    )

    hitter_pitch_types = [x for x in h["TaggedPitchType"].dropna().unique().tolist()]
    palette_h    = sns.color_palette("husl", len(hitter_pitch_types)) if hitter_pitch_types else []
    pitch_colors = dict(zip(hitter_pitch_types, palette_h))

    # tag each pitch with one plot event
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

    # assemble the overall and per pitch type table
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

    # split by pitcher handedness
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

    # draw the hitter figure
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
    axh.text(0.97, 0.25, f"Data: {platform_label(h)}", fontsize=11, ha="right")

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

    # offer the figure as a PNG download
    combo_buf = io.BytesIO()
    fig_combo.savefig(combo_buf, format="png", bbox_inches="tight")
    combo_buf.seek(0)

    st.download_button(
        "Download Hitter Heatmap + Summary Table (PNG)",
        data=combo_buf,
        file_name=f"{selected_hitter.replace(' ', '_')}_heatmap_summary.png",
        mime="image/png"
    )

    # build a CSV with exactly the numbers shown on the PNG
    st.markdown("### Download")
    hitter_report = summary_for_png.merge(
        hand_df_h.rename(columns=lambda c: c.replace("\n", " ")),
        on="Pitch Type", how="left"
    )
    for c in hitter_report.columns:
        if c == "Pitch Type":
            continue
        if c in ("Pitches", "Swings", "Whiffs", "Chases", "BIP"):
            hitter_report[c] = pd.to_numeric(hitter_report[c], errors="coerce").astype("Int64")
        else:
            hitter_report[c] = pd.to_numeric(hitter_report[c], errors="coerce").round(1)
    hitter_report.insert(0, "Game", game_label)
    hitter_report.insert(1, "Player", selected_hitter)
    hitter_report.insert(2, "Data", platform_label(h))

    st.download_button(
        "Download Report Data (CSV)",
        data=hitter_report.to_csv(index=False).encode("utf-8"),
        file_name=f"{label_slug}_{slugify(selected_hitter)}_hitting.csv",
        mime="text/csv"
    )
