import csv
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
import psycopg2
import numpy as np
from PIL import Image
import requests
from urllib3.exceptions import ConnectionError, ReadTimeoutError
from io import UnsupportedOperation


# langs = ['en', 'fr', 'es', 'it', 'de', 'pl', 'nl', 'ru']


def download_image(url, t=None, save_path=None, target_size=256, download_max_mb=5, image_max_mp=10, verbose=False, timeout=1.):
    headers = {
        'User-Agent': 'Dataset crawler for https://www.kaggle.com/c/wikipedia-image-caption',
        'From': 'jakub.r.zadrozny@gmail.com'
    }

    try:
        r = requests.get(url, stream=True, headers=headers, timeout=timeout)
    except requests.exceptions.RequestException as e:
        if verbose:
            print("FAILED request to {} failed".format(url), e)
        return None

    if not r.ok or (download_max_mb > 0 and 'Content-length' not in r.headers):
        if verbose:
            print(r.status_code, "request to {} failed".format(url))
        return None
    if download_max_mb > 0:
        size = int(r.headers['Content-length'])
        if size > download_max_mb * 1000 * 1000:
            if verbose:
                print("file at {}: too large ({} MB, limit is {} MB)".format(url, round(size/1000000, 2), download_max_mb))
            return None

    try:
        ext = Path(url).suffix.lower()
        if ext == '.svg':
            return None
        else:
            img = Image.open(r.raw)

        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        if w * h > image_max_mp * 1000 * 1000:
            if verbose:
                print("file at {}: too many pixels ({} MP, limit is {} MP)".format(url, round(w*h/1000000, 2), image_max_mp))
            return None

        if save_path is not None:
            ratio = max(target_size / w, target_size / h)
            img = img.resize((int(w * ratio), int(h * ratio)))
            img.save(save_path)

        if t is not None:
            img = t(img)
        
        return img
    except (OSError, ReadTimeoutError, ValueError, UnsupportedOperation, ConnectionError) as e:
        if verbose:
            print("file at {}: unknown error".format(url), e)
        return None


def init_worker(f):
    f.conn = psycopg2.connect(user="kubaz", dbname="wikicaptions", port=5433, host="/var/run/postgresql/")


def download_db_based(idx, save_dir, target_size=256, verbose=False):
    path = save_dir / (str(idx) + ".jpg")
    if path.exists():
        return

    cur = download_db_based.conn.cursor()
    q = "SELECT url, lang FROM images WHERE id=%s;"
    cur.execute(q, (idx,))
    if cur.rowcount > 0:
        url, lang = cur.fetchone()
        # if lang in langs:
        img = download_image(url, save_path=path, target_size=target_size, verbose=verbose)
        if img is None:
            rand_img = (np.random.rand(target_size, target_size, 3) * 255).astype(np.uint8)
            img = Image.fromarray(rand_img, mode="RGB")
            img.save(path)


def download_list_based(img_data, save_dir, target_size=256, verbose=False):
    path = save_dir / (str(img_data['id']) + ".jpg")
    if path.exists():
        return

    img = download_image(
        img_data['image_url'], 
        save_path=path, 
        target_size=target_size, 
        verbose=verbose,
        download_max_mb=0,
        image_max_mp=1000,
        timeout=5.,
    )
    if img is None:
        rand_img = (np.random.rand(target_size, target_size, 3) * 255).astype(np.uint8)
        img = Image.fromarray(rand_img, mode="RGB")
        img.save(path)


if __name__ == "__main__":
    test = True
    n_workers = 8

    if test:
        save_dir = Path("data/test_images")
        list_filepath = "data/test.tsv"
        to_download = []
        with open(list_filepath, newline='') as f:
            delimiter = '\t'
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                to_download.append({
                    "id": row["id"],
                    "image_url": row["image_url"],
                })
        _download_image = partial(download_list_based, save_dir=save_dir, verbose=True)
        pool = Pool(n_workers)
    else:
        save_dir = Path("data/train_images")
        conn = psycopg2.connect(user="kubaz", dbname="wikicaptions", port=5433, host="/var/run/postgresql/")
        cur = conn.cursor()
        q = "SELECT MAX(id) FROM images;"
        cur.execute(q)
        maxid = cur.fetchone()[0]
        del conn
        to_download = [int(x) for x in np.random.permutation(maxid)]
        _download_image = partial(download_db_based, save_dir=save_dir)
        pool = Pool(n_workers, initializer=init_worker, initargs=(download_db_based,))
    
    for _ in tqdm(pool.imap(_download_image, to_download), total=len(to_download)):
        pass

    pool.close()
    pool.join()
