import glob
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
# from .utils import resize
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.interpolate import interp1d

DEVICE_HZ = 100  # Hz
WINDOW_SEC = 10  # seconds
WINDOW_OVERLAP_SEC = 0  # i'm doing no overlap as I probably don't really care about activity boundaries
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%
TARGET_HZ = 30  # Hz
TARGET_WINDOW_LEN = int(TARGET_HZ * WINDOW_SEC)


# DATAFILES = "/Users/catong/repos/video-imu/data/"
# DATAFILES = DATAFILES + "wisdm/wisdm-dataset/raw/watch/accel/*.txt"
datafolder = os.path.join("/mnt/parscratch/users/acp25lmc/joined_data_parquet")
sites = ["MS10"] # ["MS21", "MS10", "MS24", "MS25"]
OUTDIR = os.path.join("/mnt/parscratch/users/acp25lmc/ssl-data/mobd_sampled")
num_workers = 4  # update this based on number of cores requested

def resize(X, length, axis=1):
    """Resize the temporal length using linear interpolation.
    X must be of shape (N,M,C) (channels last) or (N,C,M) (channels first),
    where N is the batch size, M is the temporal length, and C is the number
    of channels.
    If X is channels-last, use axis=1 (default).
    If X is channels-first, use axis=2.
    """
    length_orig = X.shape[axis]
    t_orig = np.linspace(0, 1, length_orig, endpoint=True)
    t_new = np.linspace(0, 1, length, endpoint=True)
    X = interp1d(t_orig, X, kind="linear", axis=axis, assume_sorted=True)(
        t_new
    )
    return X


def is_good_quality(w):
    """Window quality check"""

    if w.isna().any().any():
        print("missing values - window rejected")
        return False

    if len(w) != WINDOW_LEN:
        print(f"window length {len(w)}, expected length {WINDOW_LEN} - window rejected")
        return False

    return True


def process_windows_sampled(file_list, window_step_len, window_len, target_window_len, outdir, hours=24, max_windows_per_person=1500):
    """
    Function to selectively sample from 7 days of IMU data following ssl-wearables methodology
    (choose one 24 hour period and randomly sample 1500 windows from that period)
    """
    os.makedirs(outdir, exist_ok=True)

    X_all = []
    Y_all = []
    T_all = []
    P_all = []

    columns = ["time_acc", "acc_x", "acc_y", "acc_z", "timestamp", "p_id", "overall_nep_status"]

    for datafile in file_list:
        try:
            one_person_data = pd.read_parquet(
                datafile,
                columns=columns
            )
        except Exception as e:
            print(f"\n[ERROR] skipping {datafile}: {e}")
            continue

        if one_person_data['time_acc'].isna().any():
            print(f"\n[WARNING] skipping {os.path.basename(datafile)}: File contains NaN values in 'time_acc'.")
            continue

        # one_person_data.index = range(1, len(one_person_data) + 1)
        pid = one_person_data["p_id"].max()

        # ensure chronological sort
        # one_person_data['time_acc'] = pd.to_datetime(one_person_data['time_acc'])
        one_person_data = one_person_data.sort_values('time_acc')

        # find first and last times
        min_time = one_person_data['time_acc'].min()
        max_time = one_person_data['time_acc'].max()

        seconds_in_a_sample = hours * 60 * 60
        total_duration_seconds = max_time - min_time

        # # lastest possible start must be 24 hours before end of file
        # latest_possible_start = max_time - pd.Timedelta(days=days)
        #
        if total_duration_seconds <= seconds_in_a_sample:
            # If the person has less than 24 hours of total data, take whatever they have
            day_data = one_person_data
            print(f"PID {pid}: Total data duration ({total_duration_seconds / 3600:.2f} hours) is less than {hours} hours. Using all data.")
        else:
            # Calculate the latest possible float value where a 24-hr window could start
            latest_possible_start = max_time - seconds_in_a_sample

            # Pick a random starting float timestamp within the valid range
            start_time = np.random.uniform(min_time, latest_possible_start)
            end_time = start_time + seconds_in_a_sample

            # Slice the 24-hour window using the floats
            day_data = one_person_data[
                (one_person_data['time_acc'] >= start_time) & (one_person_data['time_acc'] < end_time)]
            print(
                f"PID {pid}: Total duration {total_duration_seconds / 3600:.2f} hours. Picked a {hours}hr window starting at float {start_time}")

        X_person = []
        Y_person = []
        T_person = []
        P_person = []

        # 2. Extract valid windows from this specific 24 hours
        for i in range(0, len(day_data), window_step_len):
            w = day_data.iloc[i: i + window_len]

            if not is_good_quality(w):
                continue

            x = w[["acc_x", "acc_y", "acc_z"]].values
            t = w["timestamp"].max()
            y = w["overall_nep_status"].max()

            X_person.append(x)
            Y_person.append(y)
            T_person.append(t)
            P_person.append(pid)

        num_extracted = len(X_person)

        # 3. Restrictive sampling: Downsample to exactly 1500 windows
        if num_extracted > 0:
            if num_extracted > max_windows_per_person:
                sampled_indices = np.random.choice(num_extracted, max_windows_per_person, replace=False)
                sampled_indices.sort()  # Keep chronological order

                X_person = [X_person[idx] for idx in sampled_indices]
                Y_person = [Y_person[idx] for idx in sampled_indices]
                T_person = [T_person[idx] for idx in sampled_indices]
                P_person = [P_person[idx] for idx in sampled_indices]
                print(
                    f"PID {pid}: Picked random day. Sampled {max_windows_per_person} from {num_extracted} valid windows.")
            else:
                print(
                    f"PID {pid}: Picked random day. Kept all {num_extracted} windows (under {max_windows_per_person}).")

            X_all.extend(X_person)
            Y_all.extend(Y_person)
            T_all.extend(T_person)
            P_all.extend(P_person)

    # 4. Final array building & saving
    if len(X_all) > 0:
        X = np.array(X_all)
        Y = np.array(Y_all)
        T = np.array(T_all)
        P = np.array(P_all)

        X = X / 9.81
        X = resize(X, target_window_len)

        np.save(os.path.join(outdir, "X.npy"), X)
        np.save(os.path.join(outdir, "Y.npy"), Y)
        np.save(os.path.join(outdir, "time.npy"), T)
        np.save(os.path.join(outdir, "pid.npy"), P)

        print("\nDataset successfully created with random-day restrictive sampling!")
        print("Final X shape:", X.shape)
    else:
        print("\n[ERROR] No valid windows extracted.")



def locate_sensor_data(root_folder, suffix: str=".csv.gz", tag_search: bool=False, tags: list=None):
    """
    Function to recursively search a root directory and return a list of filepaths of sensor data
    :param root_folder: path to root directory
    :param suffix: suffix of filetypes to
    :param tag_search: Flags whether there are additional search terms to include
    :param tags: List of additional search terms to include
    :return: list of filepaths of sensor data
    """
    # Create a Path object
    root = Path(root_folder)
    file_list = []
    print("scanning for files")
    for root, dirs, files in os.walk(root_folder):
        # print(f"scanning: {root}, {dirs}")
        for filename in files:
            if tag_search:
                if filename.endswith(suffix) and any(tag in filename for tag in tags):
                    file_list.append(os.path.join(root, filename))
            else:
                if filename.endswith(suffix):
                    file_list.append(os.path.join(root, filename))
    print(f"{len(file_list)} files found")
    return file_list


if __name__ == "__main__":
    file_list = []
    for site in sites:
        site_folder = os.path.join(datafolder, site)
        tmp_file_list = locate_sensor_data(site_folder, suffix=".parquet", tag_search=False)
        file_list.extend(tmp_file_list)
        print(file_list)

    process_windows_sampled(file_list,
                    window_step_len=WINDOW_STEP_LEN,
                    window_len=WINDOW_LEN,
                    target_window_len=TARGET_WINDOW_LEN,
                    outdir=OUTDIR)