import pandas as pd
import requests
import os
import shutil
from tqdm import tqdm


classes = ["new", "old"]


def download_img(url, file_path):
    r = requests.get(url)
    with open(file_path, "wb") as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)


def download(root):
    for c in classes:
        folder_path = os.path.join(root, c)
        try:
            os.makedirs(folder_path)
        except FileExistsError:
            print(f"Folder {folder_path} has already existed.")
        folder_path = os.path.join(root, "img", c)
        try:
            os.makedirs(folder_path)
        except FileExistsError:
            print(f"Folder {folder_path} has already existed.")

    df = pd.read_csv(os.path.join(root, "image_pairs.csv"))
    print(df.head())

    to_be_deleted = []
    with open(os.path.join(root, "to_be_deleted.txt"), "r") as f:
        for line in f:
            to_be_deleted.append(int(line))

    columns = ["new_id", "new_num", "new_url", "old_id", "old_num", "old_url"]
    new_df = pd.DataFrame([], columns=columns)

    # TODO: dowload with multithread
    for i, row in tqdm(enumerate(df.itertuples(index=False)), total=len(df.index)):
        if i in to_be_deleted:
            print(f"Ignored image: {i}, {row}")
            continue

        new_row = {}
        for j, c in enumerate(classes):
            url = row[j]
            filename = url.split("/")[-1]
            filepath = os.path.join(root, "img", c, filename)

            if not os.path.exists(os.path.join(root, "img", c, filename)):
                download_img(url, filepath)

            # copy downloaded file to sequential filename
            image_id, ext = filename.split(".")
            new_filename = "{:04d}.{}".format(i, ext)
            new_filepath = os.path.join(root, c, new_filename)
            shutil.copy2(filepath, new_filepath)

            # create row for new csv file
            new_row[f"{c}_url"] = url
            new_row[f"{c}_id"] = image_id
            new_row[f"{c}_num"] = i
        new_df = new_df.append(new_row, ignore_index=True)

    # save new pairs to new csv file
    new_df.to_csv(os.path.join(root, "new_image_pairs.csv"), index=False)


ROOT = "../data_raw"
download(ROOT)
