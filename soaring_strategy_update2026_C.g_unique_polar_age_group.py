import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
matplotlib.use("Qt5Agg")
import math
import scipy.stats as st
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.optimize import curve_fit
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D

###################################
def peakdetect_scipy(signal, lookahead=1):
    signal = np.array(signal)
    distance = max(1, lookahead)
    max_idx, _ = find_peaks(signal, distance=distance)
    min_idx, _ = find_peaks(-signal, distance=distance)
    higher_peaks = [[i, signal[i]] for i in max_idx]
    lower_peaks  = [[i, signal[i]] for i in min_idx]
    return (higher_peaks, lower_peaks)

###################################

class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

setattr(Axes3D, 'arrow3D', _arrow3D)

### --- CONTROL FLAGS ---
full_track_control = False
do_plots = False
do_corrected_thermals = False
circles = False
do_last_points = False
daily_plots = False

interval1 = 5
interval2 = 20
bin_limit = 10
defined_threshold = 36

### --- LOAD DATA ---
df_main = pd.read_csv(
    r'C:\Users\alrodrigues\Documents\M2 AMU\STAGE M2\RODRIGUES_CIRCAETE\PYTHON\adaptive_soaring_strategy\prepared_data\Circaetus_gallicus2026-06-05.csv')

df_main = df_main.rename(columns={
    "X.m._MA": "X(m)_MA",
    "Y.m._MA": "Y(m)_MA",
    "Z.m._MA": "Z(m)_MA",
})
df_main["date_in_unix_original"] = df_main["date_in_unix"]

df_main["age_season"] = df_main["age_group"].astype(str) + "_" + df_main["season"].astype(str)

# ============================================================
# MODIFICATION 1 : Créer un dictionnaire individual -> age_group
# Nécessite une colonne "age_group" (ou "age") dans le CSV.
# ============================================================
AGE_COL = "age_group"   

if AGE_COL not in df_main.columns:
    raise ValueError(
        f"La colonne '{AGE_COL}' est absente du CSV. "
        f"Colonnes disponibles : {list(df_main.columns)}"
    )

# Dictionnaire : individual_id -> age_group
individual_to_age = (
    df_main[["individuals", AGE_COL]]
    .drop_duplicates(subset=["individuals"])
    .set_index("individuals")[AGE_COL]
    .to_dict()
)

# Palette de couleurs par groupe d'âge — cohérente avec R
AGE_PALETTE = ["#00B074", "#FF9F1C"]  # vert (adult), orange (juvenile)

# Correspondance valeur brute -> libellé d'affichage (cohérent avec les figures R)
AGE_DISPLAY_LABELS = {
    "adult": "≥ 5 yrs",
    "adulte": "≥ 5 yrs",
    "juvenile": "≤ 3 yrs",
    "juvénile": "≤ 3 yrs",
}

def get_age_label(age):
    return AGE_DISPLAY_LABELS.get(str(age).strip().lower(), str(age))

def is_adult(age):
    return str(age).strip().lower() in ("adult", "adulte")

# ============================================================

individuals = df_main["individuals"].unique()
individuals = [individual for individual in individuals if str(individual) != 'nan']

updated_all_individuals = pd.DataFrame()

basic_thermal_climbrate = []
basic_thermal_climbrate_err = []
basic_mean_horizontal_speed = []
basic_mean_horizontal_speed_err = []
# MODIFICATION 2 : stocker l'age_group associé à chaque point journalier
basic_age_group = []

horizontal_velocity_samples = []
vertical_velocity_samples = []
radius_samples = []
K_samples = []

raw_radius_samples = []
raw_K_samples = []
raw_Vxy_samples = []
raw_Vz_samples = []
name_indiviuals = []

for individual in individuals:

    df_ind = df_main.loc[(df_main.individuals == individual)]
    age_group_of_individual = individual_to_age.get(individual, "unknown")

    updated_all_days = pd.DataFrame()
    days = df_ind["only_date"].unique()
    days = [day for day in days if str(day) != 'nan']

    for day in days:

        daily_thermal_counts = []
        df = df_ind.loc[(df_ind.only_date == day)]

        df["to_discard"] = (abs(df["date_in_unix"].diff()) > 3.01)
        df["to_discard"] = df["to_discard"].where(df["to_discard"]).bfill(limit=5).fillna(0).astype(bool)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df.loc[df["to_discard"], numeric_cols] = np.nan

        df["heading"] = np.arctan2(df["Vx_MA"], df["Vy_MA"]) / np.pi * 180
        df.loc[(df.heading < 0), "heading"] = df["heading"] + 360
        df["K"] = np.abs(df["K"])
        df["R"] = 1 / df["K"]
        df["heading_thermal"] = df["heading"]
        df["time_delta"] = df["date_in_unix"].diff()

        if len(df) < 100:
            continue

        deltat = float("%.2f" % df["time_delta"].describe()["mean"])

        if np.isnan(deltat):
            continue

        threshold_value = defined_threshold / deltat

        df["thermal_check"] = 0
        df.loc[(abs(df['R']) < 1000) & (df["vertical_speed"] > 0) & (df["horizontal_speed"] > 3), 'thermal_check'] = 1

        rolling_window = int((3 / deltat))
        df["thermal_check"] = df["thermal_check"].rolling(rolling_window, center=True, win_type='gaussian').mean(std=1)
        df.loc[abs(df['thermal_check']) == 0, 'heading_thermal'] = np.nan

        thermals = df.loc[df['thermal_check'] > 0]

        if df["Vxy_MA"].describe()["mean"] < 1 or len(df) == 0:
            continue

        df["Original_X(m)_MA"], df["Original_Y(m)_MA"], df["Original_Z(m)_MA"] = df["X(m)_MA"], df["Y(m)_MA"], df["Z(m)_MA"]

        if full_track_control:
            points = np.array([df["X(m)_MA"], df["Y(m)_MA"], df["Z(m)_MA"]], dtype='object').T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = Line3DCollection(segments, cmap='jet')
            vario = df["vertical_speed"].dropna()
            lc.set_array(np.array(vario))
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.add_collection3d(lc)
            ax.set_xlim(df["X(m)_MA"].min() - 400, df["X(m)_MA"].max() + 400)
            ax.set_ylim(df["X(m)_MA"].min() - 400, df["X(m)_MA"].max() + 400)
            ax.set_zlim(df["Z(m)_MA"].min(), df["Z(m)_MA"].max() + 100)
            ax.plot(df["X(m)_MA"], df["Y(m)_MA"], 0, color="grey", zorder=0)
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
            ax.set_title(f"{individual}{day}")
            line = ax.add_collection(lc)
            fig.colorbar(line)
            plt.show()

        counter_number = 0
        is_found = True
        extract_series = pd.DataFrame()
        datapoints = []
        new_dataframe = pd.DataFrame()

        df["Xw"] = np.nan
        df["Yw"] = np.nan
        df["WindX"] = np.nan
        df["WindY"] = np.nan

        peak_counter_x = []
        peak_counter_y = []
        number_of_daily_thermals = []
        last_thermal_timestamps = []

        for idx in range(0, len(df['heading_thermal'])):
            heading_value = df.iloc[idx, df.columns.get_loc('heading_thermal')]

            if math.isnan(heading_value):
                if counter_number >= threshold_value:
                    print(f"thermal found, day:{day}, range: {((idx) - counter_number)}:{(idx - 1)}")
                    number_of_daily_thermals.append(day)
                    extract_series = df[((idx) - counter_number):(idx - 1)]
                    extract_series.reset_index(inplace=True)

                    if len(extract_series) < 12:
                        counter_number = 0
                        is_found = False
                        extract_series = pd.DataFrame()
                        continue

                    extract_series["merge_key"] = (
                        extract_series["HF.ID"].astype(str) + "_" +
                        extract_series["date_in_unix_original"].astype(str)
                    )
                    numeric_cols = extract_series.select_dtypes(include=[np.number]).columns
                    extract_series.loc[extract_series.index[-1:], numeric_cols] = np.nan
                    start_heading = extract_series['heading_thermal'][0]
                    last_start_heading = 0

                    extract_series["date_in_unix"] = extract_series["date_in_unix"] - extract_series["date_in_unix"][0]
                    extract_series["X_visualization"] = extract_series["X(m)_MA"]
                    extract_series["Y_visualization"] = extract_series["Y(m)_MA"]
                    extract_series["X(m)_MA"] = extract_series["X(m)_MA"] - extract_series["X(m)_MA"][0]
                    extract_series["Y(m)_MA"] = extract_series["Y(m)_MA"] - extract_series["Y(m)_MA"][0]

                    lookahead = int(3 / deltat)
                    peaks_x = peakdetect_scipy(extract_series["Vx_MA"], lookahead=lookahead)
                    peaks_y = peakdetect_scipy(extract_series["Vy_MA"], lookahead=lookahead)

                    higherPeaks_x = np.array(peaks_x[0])
                    lowerPeaks_x  = np.array(peaks_x[1])
                    higherPeaks_y = np.array(peaks_y[0])
                    lowerPeaks_y  = np.array(peaks_y[1])

                    if do_plots:
                        fig_first, ax_arrr = plt.subplots(2)
                        ax_arrr[0].plot(extract_series["date_in_unix"], extract_series["Vx_MA"])
                        if len(higherPeaks_x) > 0 and np.array(higherPeaks_x).ndim == 2:
                            ax_arrr[0].scatter((higherPeaks_x[:, 0] * deltat), higherPeaks_x[:, 1])
                        if len(lowerPeaks_x) > 0 and np.array(lowerPeaks_x).ndim == 2:
                            ax_arrr[0].scatter((lowerPeaks_x[:, 0] * deltat), lowerPeaks_x[:, 1])
                        ax_arrr[0].set_xlabel("time"); ax_arrr[0].set_ylabel("Vx")
                        ax_arrr[1].plot(extract_series["date_in_unix"], extract_series["Vy_MA"])
                        if len(higherPeaks_y) > 0 and np.array(higherPeaks_y).ndim == 2:
                            ax_arrr[1].scatter((higherPeaks_y[:, 0] * deltat), higherPeaks_y[:, 1])
                        if len(lowerPeaks_y) > 0 and np.array(lowerPeaks_y).ndim == 2:
                            ax_arrr[1].scatter((lowerPeaks_y[:, 0] * deltat), lowerPeaks_y[:, 1])
                        ax_arrr[1].set_xlabel("time"); ax_arrr[1].set_ylabel("Vy")
                        plt.show(block=True)

                    if len(lowerPeaks_x) > len(higherPeaks_x):
                        lowerPeaks_x = lowerPeaks_x[:-1, :]
                    if len(lowerPeaks_x) < len(higherPeaks_x):
                        higherPeaks_x = higherPeaks_x[1:, :]
                    if (len(higherPeaks_x) != len(lowerPeaks_x) or len(lowerPeaks_x) == 0):
                        counter_number = 0; is_found = False; extract_series = pd.DataFrame(); continue

                    differencePeaks_x = (higherPeaks_x + lowerPeaks_x) / 2
                    for index, value in enumerate(differencePeaks_x):
                        if index <= len(differencePeaks_x) - 2:
                            if np.absolute(differencePeaks_x[:, 0][index] - differencePeaks_x[:, 0][index + 1]) < 2:
                                row = np.where((differencePeaks_x == differencePeaks_x[index]).all(axis=1))
                                differencePeaks_x = np.delete(differencePeaks_x, row, axis=0)
                        else:
                            break

                    range_mid_x = np.arange(int(min(differencePeaks_x[:, 0]) + 1),
                                            int(max(differencePeaks_x[:, 0]) - 1))

                    if len(higherPeaks_y) > len(lowerPeaks_y):
                        higherPeaks_y = higherPeaks_y[:-1, :]
                    if len(higherPeaks_y) < len(lowerPeaks_y):
                        lowerPeaks_y = lowerPeaks_y[1:, :]
                    if (len(higherPeaks_y) != len(lowerPeaks_y) or len(lowerPeaks_y) == 0):
                        counter_number = 0; is_found = False; extract_series = pd.DataFrame(); continue

                    differencePeaks_y = (higherPeaks_y + lowerPeaks_y) / 2
                    for index, value in enumerate(differencePeaks_y):
                        if index <= len(differencePeaks_y) - 2:
                            if np.absolute(differencePeaks_y[:, 0][index] - differencePeaks_y[:, 0][index + 1]) < 2:
                                row = np.where((differencePeaks_y == differencePeaks_y[index]).all(axis=1))
                                differencePeaks_y = np.delete(differencePeaks_y, row, axis=0)
                        else:
                            break

                    range_mid_y = np.arange(int(min(differencePeaks_y[:, 0]) + 1),
                                            int(max(differencePeaks_y[:, 0]) - 1))

                    if len(range_mid_y) == 0 or len(range_mid_x) == 0:
                        counter_number = 0; is_found = False; extract_series = pd.DataFrame(); continue

                    if range_mid_y[0] >= range_mid_x[0] and range_mid_x[-1] <= range_mid_y[-1]:
                        shared_mid = np.arange(int(min(differencePeaks_y[:, 0]) + 1), int(max(differencePeaks_x[:, 0]) - 1))
                    elif range_mid_y[0] >= range_mid_x[0] and range_mid_x[-1] >= range_mid_y[-1]:
                        shared_mid = np.arange(int(min(differencePeaks_y[:, 0]) + 1), int(max(differencePeaks_y[:, 0]) - 1))
                    elif range_mid_y[0] <= range_mid_x[0] and range_mid_x[-1] <= range_mid_y[-1]:
                        shared_mid = np.arange(int(min(differencePeaks_x[:, 0]) + 1), int(max(differencePeaks_x[:, 0]) - 1))
                    elif range_mid_y[0] <= range_mid_x[0] and range_mid_x[-1] >= range_mid_y[-1]:
                        shared_mid = np.arange(int(min(differencePeaks_x[:, 0]) + 1), int(max(differencePeaks_y[:, 0]) - 1))
                    else:
                        counter_number = 0; is_found = False; extract_series = pd.DataFrame(); continue

                    if len(shared_mid) == 0:
                        counter_number = 0; is_found = False; extract_series = pd.DataFrame(); continue

                    if len(differencePeaks_x) > 3 and len(differencePeaks_y) > 3:
                        f2_x = interp1d(differencePeaks_x[:, 0], differencePeaks_x[:, 1], kind='cubic')
                        f2_y = interp1d(differencePeaks_y[:, 0], differencePeaks_y[:, 1], kind='cubic')
                    else:
                        f2_x = interp1d(differencePeaks_x[:, 0], differencePeaks_x[:, 1], kind='linear')
                        f2_y = interp1d(differencePeaks_y[:, 0], differencePeaks_y[:, 1], kind='linear')

                    for i in extract_series["date_in_unix"].index:
                        if i <= shared_mid[0]:
                            extract_series.loc[i, "Xw"] = 0.0
                            extract_series.loc[i, "WindX"] = 0.0
                        else:
                            if i <= shared_mid[-1]:
                                extract_series.loc[i, "Xw"] = extract_series.loc[i - 1, "Xw"] + deltat * f2_x(i)
                                extract_series.loc[i, "WindX"] = f2_x(i)
                            else:
                                extract_series.loc[i, "Xw"] = extract_series.loc[i - 1, "Xw"]
                                extract_series.loc[i, "WindX"] = extract_series.loc[i - 1, "WindX"]
                        if i <= shared_mid[0]:
                            extract_series.loc[i, "Yw"] = 0.0
                            extract_series.loc[i, "WindY"] = 0.0
                        else:
                            if i <= shared_mid[-1]:
                                extract_series.loc[i, "Yw"] = extract_series.loc[i - 1, "Yw"] + deltat * f2_y(i)
                                extract_series.loc[i, "WindY"] = f2_y(i)
                            else:
                                extract_series.loc[i, "Yw"] = extract_series.loc[i - 1, "Yw"]
                                extract_series.loc[i, "WindY"] = extract_series.loc[i - 1, "WindY"]

                    extract_series["x_update"] = extract_series["X(m)_MA"] - extract_series["Xw"]
                    extract_series["y_update"] = extract_series["Y(m)_MA"] - extract_series["Yw"]
                    extract_series["x_update"].apply(lambda x: float(x))
                    extract_series["y_update"].apply(lambda x: float(x))
                    extract_series["x_update"] = extract_series["x_update"][shared_mid[0]:shared_mid[-1]]
                    extract_series["y_update"] = extract_series["y_update"][shared_mid[0]:shared_mid[-1]]
                    extract_series["x_update_MA"] = extract_series["x_update"].rolling(5, center=True, win_type='gaussian').mean(std=1)
                    extract_series["y_update_MA"] = extract_series["y_update"].rolling(5, center=True, win_type='gaussian').mean(std=1)

                    if do_corrected_thermals:
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')
                        ax.set_xlim(10, 400); ax.set_ylim(10, 400)
                        ax.set_zlim(200, max(extract_series["Z(m)_MA"]))
                        ax.plot(extract_series["X(m)_MA"], extract_series["Y(m)_MA"], extract_series["Z(m)_MA"])
                        ax.plot(extract_series["x_update_MA"], extract_series["y_update_MA"], extract_series["Z(m)_MA"])
                        ax.set_xlabel("X(m)"); ax.set_ylabel("Y(m)"); ax.set_zlabel("Z(m)")
                        fig.tight_layout(); plt.show(block=False)

                    extract_series['Vx_update'] = extract_series['x_update_MA'].diff() / extract_series["date_in_unix"].diff()
                    extract_series['Vy_update'] = extract_series['y_update_MA'].diff() / extract_series["date_in_unix"].diff()
                    extract_series['Vx_update'] = extract_series['Vx_update'].apply(lambda x: float(x))
                    extract_series['Vy_update'] = extract_series['Vy_update'].apply(lambda x: float(x))
                    extract_series["Vxy_update"] = np.sqrt(((extract_series['Vx_update'] ** 2) + (extract_series['Vy_update'] ** 2)))
                    extract_series['AX_update'] = extract_series['Vx_update'].diff() / extract_series["date_in_unix"].diff()
                    extract_series['AY_update'] = extract_series['Vy_update'].diff() / extract_series["date_in_unix"].diff()
                    extract_series['K_update'] = ((extract_series['Vx_update'] * extract_series['AY_update']) - (
                        extract_series['Vy_update'] * extract_series['AX_update'])) / (
                        ((extract_series['Vx_update'] ** 2) + (extract_series['Vy_update'] ** 2)) ** 1.5)
                    extract_series['K_update'] = np.abs(extract_series['K_update'])
                    extract_series['R_update'] = 1 / extract_series['K_update']

                    new_dataframe = pd.concat([new_dataframe, extract_series])

                    thermal_last_timestamps = extract_series["date_in_unix_original"].iloc[-2]
                    last_thermal_timestamps.append(thermal_last_timestamps)

                    extract_series = extract_series.dropna()
                    extract_series.reset_index(inplace=True)

                    for series_idx in range(1, len(extract_series['heading'])):
                        if math.fabs(start_heading - extract_series['heading'][series_idx]) <= 35.0 and (
                                series_idx - last_start_heading >= 10 / deltat):
                            pattern_series = extract_series[last_start_heading: series_idx]

                            if circles:
                                fig_circles = plt.figure()
                                plt.plot(pattern_series["x_update"], pattern_series["y_update"])
                                plt.axis('equal'); plt.show(block=True)

                            horizontal_velocity_mean = np.mean(pattern_series["Vxy_update"])
                            vertical_velocity_mean   = np.mean(pattern_series["Vz"])
                            radius_mean = np.mean(pattern_series["R_update"])
                            K_mean      = np.mean(pattern_series["K_update"])

                            raw_radius_samples.append(pattern_series["R_update"].values)
                            raw_K_samples.append(pattern_series["K_update"].values)
                            raw_Vxy_samples.append(pattern_series["Vxy_update"].values)
                            raw_Vz_samples.append(pattern_series["Vz"].values)

                            horizontal_velocity_samples.append(horizontal_velocity_mean)
                            vertical_velocity_samples.append(vertical_velocity_mean)
                            radius_samples.append(radius_mean)
                            K_samples.append(K_mean)
                            datapoints.append(len(pattern_series))

                            last_start_heading = series_idx + 1
                            start_heading = extract_series['heading'][series_idx]

                counter_number = 0
                is_found = False
                extract_series = pd.DataFrame()
            else:
                counter_number += 1
                is_found = True

        if len(new_dataframe) < 10:
            continue

        wind_cols = ["merge_key", "date_in_unix_original", "Xw", "Yw", "WindX", "WindY",
                     "x_update", "y_update", "x_update_MA", "y_update_MA",
                     "Vx_update", "Vy_update", "Vxy_update", "K_update", "R_update",
                     "X_visualization", "Y_visualization"]
        wind_cols_existing = [c for c in wind_cols if c in new_dataframe.columns]

        if "WindX" not in wind_cols_existing or "WindY" not in wind_cols_existing:
            continue

        new_dataframe_dedup = new_dataframe[wind_cols_existing].drop_duplicates(subset=["merge_key"]).rename(columns={
            "WindX": "WindX_thermal",
            "WindY": "WindY_thermal",
            "Xw": "Xw_thermal",
            "Yw": "Yw_thermal",
        })

        df["merge_key"] = df["HF.ID"].astype(str) + "_" + df["date_in_unix_original"].astype(str)

        updated_df = df.merge(new_dataframe_dedup, on="merge_key", how="left")

        print("Exemples merge_key dans new_dataframe_dedup:")
        print(new_dataframe_dedup["merge_key"].head(10).tolist())
        print("Exemples merge_key dans df:")
        print(df["merge_key"].head(10).tolist())
        keys_thermal = set(new_dataframe_dedup["merge_key"])
        keys_df = set(df["merge_key"])
        print(f"Clés en commun: {len(keys_thermal & keys_df)} / {len(keys_thermal)}")
        print(f"HF.ID uniques dans df ce jour: {df['HF.ID'].unique()}")
        print(f"HF.ID uniques dans new_dataframe: {new_dataframe['HF.ID'].unique()}")

        updated_df.loc[updated_df["WindX_thermal"] == 0, "WindX_thermal"] = np.nan
        updated_df.loc[updated_df["WindY_thermal"] == 0, "WindY_thermal"] = np.nan

        average_wind_x = np.nanmean(updated_df["WindX_thermal"]) if updated_df["WindX_thermal"].notna().any() else 0.0
        average_wind_y = np.nanmean(updated_df["WindY_thermal"]) if updated_df["WindY_thermal"].notna().any() else 0.0

        updated_df["air_speed_x"]  = pd.to_numeric(updated_df["Vx_MA"], errors='coerce') - average_wind_x
        updated_df["air_speed_y"]  = pd.to_numeric(updated_df["Vy_MA"], errors='coerce') - average_wind_y
        updated_df["air_speed_xy"] = np.sqrt(updated_df["air_speed_x"] ** 2 + updated_df["air_speed_y"] ** 2)

        thermal = updated_df[updated_df["Xw_thermal"].notnull()]
        thermal = thermal[thermal["vertical_speed"] > 0]
        daily_thermal_counts.append((day, len(number_of_daily_thermals), len(thermal)))
        gliding_part = updated_df[updated_df["vertical_speed"] < 0]
        gliding_part = gliding_part[gliding_part["K"] < 0.001]

        print(f"ID {individual}  Jour {day}: thermal={len(thermal)} pts, gliding={len(gliding_part)} pts, "
              f"Xw_thermal matchés={updated_df['Xw_thermal'].notna().sum()}")
        print(f"  vertical_speed dans Xw_thermal non-nul: "
              f"{updated_df[updated_df['Xw_thermal'].notnull()]['vertical_speed'].describe()}")
        print("\n--- Thermiques par jour ---")
        for d, n_thermals, n_points in daily_thermal_counts:
            print(f"{d} : {n_thermals} thermiques détectés, {n_points} points thermiques")

        if len(thermal) == 0:
            continue

        basic_thermal_climbrate.append(np.mean(thermal["vertical_speed"]))
        if len(thermal["vertical_speed"]) > 1:
            basic_thermal_climbrate_err.append(st.sem(thermal["vertical_speed"], ddof=1))
        else:
            basic_thermal_climbrate_err.append(np.nan)

        basic_mean_horizontal_speed.append(np.mean(gliding_part["air_speed_xy"]))
        if len(gliding_part["air_speed_xy"]) > 1:
            basic_mean_horizontal_speed_err.append(st.sem(gliding_part["air_speed_xy"], ddof=1))
        else:
            basic_mean_horizontal_speed_err.append(np.nan)

        # MODIFICATION 3 : enregistrer le groupe d'âge pour ce point journalier
        basic_age_group.append(age_group_of_individual)

        if daily_plots:
            fig_work_parts = plt.figure()
            ax = fig_work_parts.add_subplot(111, projection='3d')

            gliding_part_plot = gliding_part.copy()
            gliding_part_plot["to_discard"] = (abs(gliding_part_plot["date_in_unix"].diff()) > 3.01)
            gliding_part_plot = gliding_part_plot.astype(object)
            gliding_part_plot[gliding_part_plot["to_discard"]] = np.nan

            thermal_plot = thermal.copy()
            thermal_plot["to_discard"] = (abs(thermal_plot["date_in_unix"].diff()) > 3.01)
            bool_cols = thermal_plot.select_dtypes(include='bool').columns
            thermal_plot[bool_cols] = thermal_plot[bool_cols].astype(object)
            thermal_plot.loc[thermal_plot["to_discard"], :] = np.nan

            ax.plot(df["Original_X(m)_MA"], df["Original_Y(m)_MA"], df["Original_Z(m)_MA"], color="Green")
            ax.plot(gliding_part_plot["X(m)_MA"], gliding_part_plot["Y(m)_MA"], gliding_part_plot["Z(m)_MA"], color="Blue", label="Gliding")
            ax.plot(thermal_plot["X_visualization"], thermal_plot["Y_visualization"], thermal_plot["Z(m)_MA"], color="Red", label="Thermals")

            first_valid_index = df["Original_X(m)_MA"].first_valid_index()
            x_metric_difference = (gliding_part["X(m)_MA"].max() - gliding_part["X(m)_MA"].min()) / 10
            y_metric_difference = (gliding_part["Y(m)_MA"].max() - gliding_part["Y(m)_MA"].min()) / 10

            ax.arrow3D(
                int(df["Original_X(m)_MA"][first_valid_index]),
                int(df["Original_Y(m)_MA"][first_valid_index]),
                int(df["Original_Z(m)_MA"].max()) + 100,
                average_wind_x * x_metric_difference,
                average_wind_y * y_metric_difference, 0,
                mutation_scale=20, arrowstyle="-|>", color="black", zorder=30, label="Wind"
            )
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
            ax.set_title(f"{individual}{day}")
            ax.legend()
            plt.tight_layout(); plt.show(block=True)

        ### daily csv
        daily_record = False
        output_path_daily = r"C:\Users\alrodrigues\Documents\M2 AMU\STAGE M2\RODRIGUES_CIRCAETE\PYTHON\adaptive_soaring_strategy\output_data\daily_results_unique_polar.csv"
        daily_headers = ["Species", "Individual", "Date", "n_thermals", "average_wind_x", "average_wind_y",
                         "n_thermal_points", "mean_vertical_speed_thermal", "sem_vertical_speed_thermal",
                         "n_gliding_points", "mean_airspeed_gliding", "sem_airspeed_gliding"]
        daily_csv_results = [
            updated_df["Species"].unique()[0], updated_df["individuals"].unique()[0],
            updated_df["only_date"].unique()[0], len(number_of_daily_thermals),
            average_wind_x, average_wind_y, len(thermal),
            np.mean(thermal["vertical_speed"]), st.sem(thermal["vertical_speed"], ddof=1),
            len(gliding_part), np.mean(gliding_part["air_speed_xy"]), st.sem(gliding_part["air_speed_xy"], ddof=1)
        ]
        if daily_record:
            import os
            file_exists = os.path.isfile(output_path_daily) and os.path.getsize(output_path_daily) > 0
            with open(output_path_daily, "a+", newline="") as daily_result_updater:
                csv_writer = csv.writer(daily_result_updater)
                if not file_exists:
                    csv_writer.writerow(daily_headers)
                csv_writer.writerow(daily_csv_results)

        ### thermal rate output
        thermal_rate_record = False
        output_path_thermal_rate = r"C:\Users\alrodrigues\Documents\M2 AMU\STAGE M2\RODRIGUES_CIRCAETE\PYTHON\adaptive_soaring_strategy\output_data\thermal_rate_unique_polar.csv"
        thermal_rate_headers = ["Species", "Individual", "Date", "track_start", "track_end", "n_thermals",
                                "hf_duration_hours", "n_thermals_per_hour", "mean_thermal_duration_sec",
                                "total_thermal_time_sec", "thermal_time_proportion"]

        hf_valid_deltas   = updated_df["time_delta"][updated_df["time_delta"] <= 4]
        hf_duration_seconds = hf_valid_deltas.sum()
        hf_duration_hours = hf_duration_seconds / 3600 if hf_duration_seconds > 0 else np.nan
        track_start = pd.to_datetime(updated_df["date_in_unix"].min(), unit='s').strftime('%Y-%m-%d %H:%M:%S')
        track_end   = pd.to_datetime(updated_df["date_in_unix"].max(), unit='s').strftime('%Y-%m-%d %H:%M:%S')
        n_thermals_per_hour = len(number_of_daily_thermals) / hf_duration_hours if hf_duration_hours > 0 else np.nan

        if len(new_dataframe) > 0 and "date_in_unix" in new_dataframe.columns:
            total_thermal_time   = len(new_dataframe) * deltat
            mean_thermal_duration = total_thermal_time / len(number_of_daily_thermals) if len(number_of_daily_thermals) > 0 else np.nan
            thermal_time_proportion = total_thermal_time / hf_duration_seconds if hf_duration_seconds > 0 else np.nan
        else:
            total_thermal_time = mean_thermal_duration = thermal_time_proportion = np.nan

        thermal_rate_results = [
            updated_df["Species"].unique()[0], updated_df["individuals"].unique()[0],
            updated_df["only_date"].unique()[0], track_start, track_end,
            len(number_of_daily_thermals), round(hf_duration_hours, 4), round(n_thermals_per_hour, 4),
            round(mean_thermal_duration, 2) if not np.isnan(mean_thermal_duration) else np.nan,
            round(total_thermal_time, 2) if not np.isnan(total_thermal_time) else np.nan,
            round(thermal_time_proportion, 4) if not np.isnan(thermal_time_proportion) else np.nan,
        ]
        if thermal_rate_record:
            import os
            file_exists = os.path.isfile(output_path_thermal_rate) and os.path.getsize(output_path_thermal_rate) > 0
            with open(output_path_thermal_rate, "a+", newline="") as thermal_rate_updater:
                csv_writer = csv.writer(thermal_rate_updater)
                if not file_exists:
                    csv_writer.writerow(thermal_rate_headers)
                csv_writer.writerow(thermal_rate_results)

        updated_all_days = pd.concat([updated_all_days, updated_df])

        if do_last_points:
            fig_circles = plt.figure()
            plt.scatter(updated_df["air_speed_xy"], updated_df["vertical_speed"])
            plt.scatter(updated_df["Vxy_MA"], updated_df["vertical_speed"])
            plt.title(f"{updated_df['Species'].unique()[0]}")
            plt.show(block=True)

        updated_all_individuals = pd.concat([updated_all_individuals, updated_all_days])


# ============================================================
# GLIDING POLAR (all individuals, all ages combined)
# ============================================================
gliding = updated_all_individuals.loc[(updated_all_individuals.vertical_speed < 0)]
gliding = gliding.loc[(gliding.K < 0.01)]

meanYValues = []
stdYValues   = []
semYValues   = []
countYValues = []
meanXValues  = []
medianYValues = []
modeYValues  = []

for increment in np.arange(interval1, interval2, 1):
    logicalIndexs = np.logical_and((gliding["air_speed_xy"] >= increment), gliding["air_speed_xy"] < (increment + 1))
    valuesY = np.extract(logicalIndexs, gliding["vertical_speed"])
    valuesX = np.extract(logicalIndexs, gliding["air_speed_xy"])
    if len(valuesY) >= bin_limit:
        meanYValues.append(np.mean(valuesY))
        medianYValues.append(np.median(valuesY))
        modeYValues.append(st.mode(valuesY)[0])
        stdYValues.append(np.std(valuesY, ddof=1))
        countYValues.append(len(valuesY))
        semYValues.append(st.sem(valuesY, ddof=1))
        meanXValues.append(increment)

x     = np.array(meanXValues)
y     = np.array(meanYValues)
y_err = np.array(semYValues)


def f(x, a, b, c):
    return (a * (x ** 2)) + (b * x) + c

def f_linear(x, a, b):
    return (a * x) + b


import scipy as sc

print(f"Nombre de jours avec données: {len(basic_thermal_climbrate)}")
print(f"basic_thermal_climbrate: {basic_thermal_climbrate}")
print(f"basic_mean_horizontal_speed: {basic_mean_horizontal_speed}")

constantsQuadErr, pcov = curve_fit(f, x, y)
a_mean, b_mean, c_mean = constantsQuadErr
a_err, b_err, c_err = np.sqrt(np.diagonal(pcov))

x_line     = np.arange(5, 25, 1)
y_line_mean = f(x_line, a_mean, b_mean, c_mean)

B1, B2, B3 = 1, 2, 3

tangent_x1_0 = np.sqrt((c_mean - 0) / a_mean)
tangent_y1_0 = (2 * c_mean) + b_mean * tangent_x1_0

tangent_x1_1 = np.sqrt((c_mean - B1) / a_mean)
tangent_y1_1 = (2 * a_mean * tangent_x1_1 + b_mean) * tangent_x1_1 + B1

tangent_x1_2 = np.sqrt((c_mean - B2) / a_mean)
tangent_y1_2 = (2 * a_mean * tangent_x1_2 + b_mean) * tangent_x1_2 + B2

tangent_x1_3 = np.sqrt((c_mean - B3) / a_mean)
tangent_y1_3 = (2 * a_mean * tangent_x1_3 + b_mean) * tangent_x1_3 + B3

optimal_thermal_x = [tangent_x1_0, tangent_x1_1, tangent_x1_2, tangent_x1_3]
optimal_thermal_y = [0, 1, 2, 3]

MacCreadyConstant, _ = sc.optimize.curve_fit(f_linear, optimal_thermal_y, optimal_thermal_x)
mac_cready_fit_a, mac_cready_fit_b = MacCreadyConstant
optimal_y_line = np.arange(0, 3.1, 0.1)
optimal_x_line = f_linear(optimal_y_line, mac_cready_fit_a, mac_cready_fit_b)

# ============================================================
# MODIFICATION 4 : filtrage des points journaliers par âge
# ============================================================
basic_mean_horizontal_speed = np.array(basic_mean_horizontal_speed)
basic_thermal_climbrate      = np.array(basic_thermal_climbrate)
basic_thermal_climbrate_err  = np.array(basic_thermal_climbrate_err)
basic_mean_horizontal_speed_err = np.array(basic_mean_horizontal_speed_err)
basic_age_group = np.array(basic_age_group)

if len(basic_thermal_climbrate) == 0 or len(basic_mean_horizontal_speed) == 0:
    import sys; sys.exit()

# Suppression des NaN sur les erreurs
valid_mask = (~np.isnan(basic_mean_horizontal_speed_err)) & (~np.isnan(basic_thermal_climbrate_err))
basic_mean_horizontal_speed     = basic_mean_horizontal_speed[valid_mask]
basic_thermal_climbrate          = basic_thermal_climbrate[valid_mask]
basic_mean_horizontal_speed_err  = basic_mean_horizontal_speed_err[valid_mask]
basic_thermal_climbrate_err      = basic_thermal_climbrate_err[valid_mask]
basic_age_group                  = basic_age_group[valid_mask]

# Filtre sigma (outliers) sur la vitesse horizontale globale
std_dev  = np.std(basic_mean_horizontal_speed)
mean_val = np.mean(basic_mean_horizontal_speed)
sigma    = (basic_mean_horizontal_speed - mean_val) / std_dev
sigma_mask = (sigma >= -2) & (sigma <= 2)

basic_mean_horizontal_speed    = basic_mean_horizontal_speed[sigma_mask]
basic_thermal_climbrate        = basic_thermal_climbrate[sigma_mask]
basic_mean_horizontal_speed_err = basic_mean_horizontal_speed_err[sigma_mask]
basic_thermal_climbrate_err    = basic_thermal_climbrate_err[sigma_mask]
basic_age_group                = basic_age_group[sigma_mask]
# ============================================================

# ============================================================
# MODIFICATION 5 : figure finale avec droite lm par age_group
# ============================================================
age_groupes_unique = sorted(set(basic_age_group))
color_map = {
    age: (AGE_PALETTE[0] if is_adult(age) else AGE_PALETTE[1])
    for age in age_groupes_unique
}

fig_last, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 9))

# --- Polaire de vol (commune à tous les âges) ---
axs.errorbar(x, y, yerr=y_err, fmt="o", c="blue", ms=5, zorder=5, label="Polar (all individuals)")
axs.plot(x_line, y_line_mean, c="blue", lw=2, label="Quadratic fit")

# --- Tangentes MacCready (communes) ---
axs.plot(optimal_x_line, optimal_y_line, c="red", lw=1.5, alpha=0.75, ls="--", label="MacCready optimal")
for tx, ty, ty_tang in [
    (tangent_x1_0, 0, tangent_y1_0),
    (tangent_x1_1, 1, tangent_y1_1),
    (tangent_x1_2, 2, tangent_y1_2),
    (tangent_x1_3, 3, tangent_y1_3),
]:
    axs.plot([0, tx], [ty, ty],      ":", c="grey", alpha=0.5, lw=0.9)
    axs.plot([tx, tx], [ty_tang, ty], ":", c="grey", alpha=0.5, lw=0.9)
    axs.plot([0, tx], [ty, ty_tang],  "--", c="grey", alpha=0.4, lw=0.9)

# --- Scatter + droite linéaire PAR groupe d'âge ---
y_linear_fit_range = np.arange(0, basic_thermal_climbrate.max() + 0.5, 0.1)

for age in age_groupes_unique:
    mask = basic_age_group == age
    col  = color_map[age]
    display_label = get_age_label(age)

    spd_age      = basic_mean_horizontal_speed[mask]
    climbrate_age = basic_thermal_climbrate[mask]
    spd_err_age  = basic_mean_horizontal_speed_err[mask]
    cr_err_age   = basic_thermal_climbrate_err[mask]

    # Points avec barres d'erreur
    axs.errorbar(
        spd_age, climbrate_age,
        xerr=spd_err_age, yerr=cr_err_age,
        fmt="o", color=col, alpha=0.75, ms=5, label=f"{display_label} (n={mask.sum()})"
    )

    if mask.sum() >= 3:
        try:
            lin_fit, _ = sc.optimize.curve_fit(f_linear, climbrate_age, spd_age)
            a_lin, b_lin = lin_fit
            x_lin_fit = f_linear(y_linear_fit_range, a_lin, b_lin)
            axs.plot(x_lin_fit, y_linear_fit_range, color=col, lw=2,
                     label=f"lm {display_label} (slope={a_lin:.2f})")

            # --- Écart-type autour de la droite ---
            residuals = spd_age - f_linear(climbrate_age, a_lin, b_lin)
            std_residuals = np.std(residuals, ddof=2)
            axs.fill_betweenx(
                y_linear_fit_range,
                x_lin_fit - std_residuals,
                x_lin_fit + std_residuals,
                color=col, alpha=0.15
            )
        except RuntimeError:
            print(f"[WARN] Fit linéaire impossible pour le groupe d'âge '{age}'")

axs.axhline(0, color="black", lw=0.7, alpha=0.4)
axs.grid(True, alpha=0.3)
axs.set_xlim([0, 20.5])
axs.set_ylim([-3, basic_thermal_climbrate.max() + 0.5])
axs.set_title(f'Cross-Country Optimisation Strategy {df_main["Species"].unique()[0]}', fontsize=11)
axs.set_ylabel('Vertical Speed (m/s)', fontsize=14)
axs.set_xlabel('Horizontal Speed (m/s)', fontsize=14)
axs.tick_params(axis='both', labelsize=12)  # taille des chiffres sur les axes
axs.legend(fontsize=10, loc="lower left")
fig_last.tight_layout()
plt.show(block=True)
# ============================================================

# ============================================================
# PERMUTATION TEST: slope and intercept
# adult (>=5yrs) vs juvenile (<=3yrs)
# ============================================================
np.random.seed(42)
n_perm = 9999

age_bool_adult = np.array([is_adult(a) for a in basic_age_group])

x_all = basic_thermal_climbrate      # explanatory variable (climb rate)
y_all = basic_mean_horizontal_speed  # response variable (air speed)
group_all = age_bool_adult.copy()    # True = adult, False = juvenile

def fit_slope_intercept(x, y):
    a, b = sc.optimize.curve_fit(f_linear, x, y)[0]
    return a, b

# --- Observed values ---
a_adult_obs, b_adult_obs = fit_slope_intercept(x_all[group_all], y_all[group_all])
a_juv_obs, b_juv_obs     = fit_slope_intercept(x_all[~group_all], y_all[~group_all])

obs_diff_slope     = a_juv_obs - a_adult_obs
obs_diff_intercept = b_juv_obs - b_adult_obs

# --- Null distribution via label permutation ---
perm_diff_slope = np.full(n_perm, np.nan)
perm_diff_intercept = np.full(n_perm, np.nan)

for i in range(n_perm):
    shuffled = np.random.permutation(group_all)
    try:
        a_a, b_a = fit_slope_intercept(x_all[shuffled], y_all[shuffled])
        a_j, b_j = fit_slope_intercept(x_all[~shuffled], y_all[~shuffled])
        perm_diff_slope[i]     = a_j - a_a
        perm_diff_intercept[i] = b_j - b_a
    except RuntimeError:
        continue

perm_diff_slope     = perm_diff_slope[~np.isnan(perm_diff_slope)]
perm_diff_intercept = perm_diff_intercept[~np.isnan(perm_diff_intercept)]

p_slope     = np.mean(np.abs(perm_diff_slope)     >= np.abs(obs_diff_slope))
p_intercept = np.mean(np.abs(perm_diff_intercept) >= np.abs(obs_diff_intercept))

print(f"Observed slope difference     : {obs_diff_slope:.3f}  (p = {p_slope:.4f})")
print(f"Observed intercept difference : {obs_diff_intercept:.3f}  (p = {p_intercept:.4f})")

# ============================================================
# FIGURE: null distributions + observed value
# ============================================================
fig_perm, axs_perm = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, perm_dist, obs_val, p_val, title in [
    (axs_perm[0], perm_diff_slope, obs_diff_slope, p_slope, "Slope difference"),
    (axs_perm[1], perm_diff_intercept, obs_diff_intercept, p_intercept, "Intercept difference"),
]:
    ax.hist(perm_dist, bins=40, color="#B0B0B0", edgecolor="white", alpha=0.85,
             label=f"Null distribution\n(n={n_perm} permutations)")
    ax.axvline(obs_val, color="#C1272D", lw=2.2, ls="--",
                label=f"Observed value\n= {obs_val:.2f}")
    ax.set_title(f"{title}\np = {p_val:.4f}", fontsize=11)
    ax.set_xlabel("Difference (≤ 3 yrs − ≥ 5 yrs)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)

fig_perm.suptitle("Permutation test: ≥ 5 yrs vs ≤ 3 yrs", fontsize=12)
fig_perm.tight_layout()
plt.savefig(
    r"C:\Users\alrodrigues\Documents\M2 AMU\STAGE M2\Figures\permutation_test_age_polar.png",
    dpi=300
)
plt.show(block=True)

# =================================================================================

# --- Performances de vol moyennes ---
x_best_glide_mean    = np.sqrt(c_mean / a_mean)
y_best_glide_mean    = (2 * c_mean) + b_mean * np.sqrt(c_mean / a_mean)
best_glide_ratio_mean = np.absolute(x_best_glide_mean / y_best_glide_mean)
x_min_sink_mean      = -b_mean / (2 * a_mean)
y_min_sink_mean      = f(x_min_sink_mean, *constantsQuadErr)



# --- Garde-fou : le vertex de la parabole doit tomber dans la plage de données observées ---
x_data_min, x_data_max = x.min(), x.max()

min_sink_in_range = (x_data_min <= x_min_sink_mean <= x_data_max)

if not min_sink_in_range:
    print(f"[ATTENTION] x_min_sink_mean = {x_min_sink_mean:.2f} m/s est HORS de la plage "
          f"de données observées [{x_data_min:.1f}, {x_data_max:.1f}] m/s.")
    print("Ce point est une extrapolation mathématique du fit quadratique, "
          "sans signification physique. À interpréter avec prudence ou à exclure.")

# Idem pour le point de meilleure finesse, par cohérence
best_glide_in_range = (x_data_min <= x_best_glide_mean <= x_data_max)

if not best_glide_in_range:
    print(f"[ATTENTION] x_best_glide_mean = {x_best_glide_mean:.2f} m/s est HORS de la plage "
          f"de données observées [{x_data_min:.1f}, {x_data_max:.1f}] m/s.")

print(f"\n--- Vérification du garde-fou ---")
print(f"x_min_sink_mean = {x_min_sink_mean:.3f}")
print(f"Plage x observée (bins retenus) = [{x.min():.1f}, {x.max():.1f}]")
print(f"Dans la plage ? {x_data_min <= x_min_sink_mean <= x_data_max}")

#### On conserve la valeur même si elle est hors de la plage observée (extrapolation),
#### le flag min_sink_in_range permet de le signaler dans le rapport sans exclure la valeur
x_min_sink_mean_reported = x_min_sink_mean
y_min_sink_mean_reported = y_min_sink_mean

# ----------------------------------------------------------------------------------------------

print(f"Vitesse meilleure finesse: {x_best_glide_mean:.1f} m/s")
print(f"Taux de chute minimum: {y_min_sink_mean:.2f} m/s")
print(f"Meilleure finesse: {best_glide_ratio_mean:.1f}")

record = False
output_path = r"C:\Users\alrodrigues\Documents\M2 AMU\STAGE M2\RODRIGUES_CIRCAETE\PYTHON\adaptive_soaring_strategy\output_data\Result_soaring_strategy_unique_polar.csv"
headers = ["Species", "a_mean", "b_mean", "c_mean",
           "x_best_glide_mean", "y_best_glide_mean", "best_glide_ratio_mean",
           "x_min_sink_mean", "y_min_sink_mean", "a_linear", "b_linear",
           "min_sink_in_range", "best_glide_in_range"]
results = [
    df_main["Species"].unique()[0],
    a_mean, b_mean, c_mean,
    x_best_glide_mean, y_best_glide_mean, best_glide_ratio_mean,
    x_min_sink_mean_reported, y_min_sink_mean_reported,   # <-- NaN si hors plage
    mac_cready_fit_a, mac_cready_fit_b,
    min_sink_in_range, best_glide_in_range
]
if record:
    import os
    file_exists = os.path.isfile(output_path) and os.path.getsize(output_path) > 0
    with open(output_path, "a+", newline="") as result_updater:
        csv_writer = csv.writer(result_updater)
        if not file_exists:
            csv_writer.writerow(headers)
        csv_writer.writerow(results)

# --- Statistiques thermiques ---
raw_radius_samples = np.hstack(raw_radius_samples) if len(raw_radius_samples) > 0 else np.array([])
raw_radius_samples = [ele for ele in raw_radius_samples if ele > 0]
raw_K_samples      = np.hstack(raw_K_samples)   if len(raw_K_samples)   > 0 else np.array([])
raw_Vxy_samples    = np.hstack(raw_Vxy_samples) if len(raw_Vxy_samples) > 0 else np.array([])
raw_Vz_samples     = np.hstack(raw_Vz_samples)  if len(raw_Vz_samples)  > 0 else np.array([])

mean_radius  = np.mean(raw_radius_samples)
median_radius = np.median(raw_radius_samples)
mean_Vxy     = np.mean(raw_Vxy_samples)
median_Vxy   = np.median(raw_Vxy_samples)
mean_Vz      = np.mean(raw_Vz_samples)
median_Vz    = np.median(raw_Vz_samples)