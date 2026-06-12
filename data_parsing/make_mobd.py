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
sites = ["MS10"]# ["MS21", "MS10", "MS24", "MS25"]
OUTDIR = os.path.join("/mnt/parscratch/users/acp25lmc/ssl-data/mobd_clean")
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


def process_windows(file_list, window_step_len, window_len, target_window_len, outdir):

    os.makedirs(outdir, exist_ok=True)

    x_path = os.path.join(outdir, "X.npy")
    y_path = os.path.join(outdir, "Y.npy")
    t_path = os.path.join(outdir, "time.npy")
    p_path = os.path.join(outdir, "pid.npy")

    for p in [x_path, y_path, t_path, p_path]:
        if os.path.exists(p): os.remove(p)

    # initiate count
    total_files_processed = 0

    columns = ["time_acc", "acc_x", "acc_y", "acc_z", "timestamp", "p_id", "overall_nep_status"]

    for datafile in file_list:
        try:
            one_person_data_t = pd.read_parquet(
                datafile,
                columns=columns
            )
        except Exception as e:
            print(f"\n[ERROR] skipping {datafile}: {e}")
            continue

        one_person_data_t.index = range(1, len(one_person_data_t) + 1)
        pid = one_person_data_t["p_id"].max()

        X_list = []
        Y_list = []
        T_list = []
        P_list = []

        # return one_person_data_t, pid
        for i in range(0, len(one_person_data_t), window_step_len):
            w = one_person_data_t.iloc[i : i + window_len]

            if not is_good_quality(w):
                continue

            x = w[["acc_x", "acc_y", "acc_z"]].values
            t = w["timestamp"].max()
            y = w["overall_nep_status"].max()


            X_list.append(x)
            Y_list.append(y)
            T_list.append(t)
            P_list.append(pid)

        if len(X_list) > 0:
            X_block = np.array(X_list)
            X_block = X_block / 9.81 # convert to g
            X_block = resize(X_block, target_window_len)

            Y_block = np.array(Y_list)
            T_block = np.array(T_list)
            P_block = np.array(P_list)

            print(X_block.shape, Y_block.shape, T_block.shape, P_block.shape)

            # if first file create .npy, otherwise append
            with open(x_path, 'ab' if total_files_processed > 0 else 'wb') as f:
                np.save(f, X_block)
            with open(y_path, 'ab' if total_files_processed > 0 else 'wb') as f:
                np.save(f, Y_block)
            with open(t_path, 'ab' if total_files_processed > 0 else 'wb') as f:
                np.save(f, T_block)
            with open(p_path, 'ab' if total_files_processed > 0 else 'wb') as f:
                np.save(f, P_block)

            total_files_processed += 1
            print(f"Processed & saved {len(X_list)} windows from {os.path.basename(datafile)}.")

    # # Convert to numpy arrays
    # X = np.array(X_list)
    # Y = np.array(Y_list)
    # T = np.array(T_list)
    # P = np.array(P_list)

    # print(X.shape, Y.shape, T.shape, P.shape)

    # # fixing unit to g
    # X = X / 9.81
    # # downsample to 30 Hz
    # X = resize(X, target_window_len)

    print("dataset made!")

    # os.system(f"mkdir -p {outdir}")
    # np.save(os.path.join(outdir, "X"), X)
    # np.save(os.path.join(outdir, "Y"), Y)
    # np.save(os.path.join(outdir, "time"), T)
    # np.save(os.path.join(outdir, "pid"), P)

    # print(f"Saved in {outdir}")
    # print("X shape:", X.shape)
    # print("Y distribution:", len(set(Y)))
    # print(pd.Series(Y).value_counts())
    # print("User distribution:", len(set(P)))
    # print(pd.Series(P).value_counts())

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

def run_parallel_process(func, item_list, max_workers, **kwargs):
    """
    A function to run embarrassingly parallel processes on multiple items
    :param func: The function to run in parallel
    :param item_list: List of items to iterate over
    :param max_workers: Number of cores to parallelise over
    :param kwargs: Arguments required for func
    :return: void
    """
    # partial "pre-fills" the function with the kwargs we have passed
    executable_func = partial(func, **kwargs)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # create the list of jobs
        futures = {executor.submit(executable_func, item): item for item in item_list}

        # create the progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="processing..."):
            try:
                future.result()
            except Exception as e:
                print(f"Item {futures[future]} generated an exception: {e}")

def process_all_files(datafolder, sites, window_step_len, window_len, target_window_len, outdir):
    for site in sites:
        site_folder = os.path.join(datafolder, site)
        file_list = locate_sensor_data(site_folder, suffix=".parquet", tag_search=False)
        print(f"\n--- Site: {site} | Total files: {len(file_list)} ---")

        run_parallel_process(process_windows,
                             file_list,
                             num_workers,
                             window_step_len=window_step_len,
                             window_len=window_len,
                             target_window_len=target_window_len,
                             outdir=outdir
                             )

if __name__ == "__main__":
    file_list = []
    for site in sites:
        site_folder = os.path.join(datafolder, site)
        tmp_file_list = locate_sensor_data(site_folder, suffix=".parquet", tag_search=False)
        file_list.extend(tmp_file_list)
        print(file_list)

    process_windows(file_list,
                    window_step_len=WINDOW_STEP_LEN,
                    window_len=WINDOW_LEN,
                    target_window_len=TARGET_WINDOW_LEN,
                    outdir=OUTDIR)

    # process_all_files(datafolder,
    #                   sites,
    #                   window_step_len=WINDOW_STEP_LEN,
    #                   window_len=WINDOW_LEN,
    #                   target_window_len=TARGET_WINDOW_LEN,
    #                   outdir=OUTDIR)