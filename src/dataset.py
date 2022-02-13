from collections import defaultdict
from pathlib import Path
import csv
from pathlib import Path
from urllib.parse import unquote

from PIL import Image, ImageFile
import psycopg2
from tqdm.notebook import tqdm
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from download import download_image


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def extract_train_data(in_dir):
    conn = psycopg2.connect("dbname=wikicaptions user=kubaz")
    cur = conn.cursor()
    q = "INSERT INTO images (id, url, caption, lang) VALUES (%s, %s, %s, %s);"
    idx = 0
    in_dir = Path(in_dir)
    for fname in tqdm(in_dir.glob('*.tsv')):
        with open(fname, newline='') as in_f:
            reader = csv.DictReader(in_f, delimiter="\t")
            for row in tqdm(reader):
                url = row["image_url"]
                caption = ' '.join(
                    row["caption_title_and_reference_description"].split())[:2500]
                lang = row["language"]
                cur.execute(q, (idx, url, caption, lang))
                idx += 1
        conn.commit()
    cur.close()
    conn.close()


class DBWikiImageDataset(Dataset):
    def __init__(self, dbname, user, image_transform=None, text_transform=None, save_dir=None, port=5432, host="/tmp/"):
        self.dbname = dbname
        self.user = user
        self.port = port
        self.host = host
        self.conn = psycopg2.connect(dbname=self.dbname, user=self.user, host=self.host, port=self.port)
        cur = self.conn.cursor()
        q = "SELECT COUNT(id) FROM images;"
        cur.execute(q)
        self.N = cur.fetchone()[0]
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.save_dir = save_dir

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['conn']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.conn = psycopg2.connect(dbname=self.dbname, user=self.user, host=self.host, port=self.port)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        idx = int(idx)
        q = "SELECT url, caption, lang FROM images WHERE id=%s;"
        cur = self.conn.cursor()
        cur.execute(q, (idx,))
        url, caption, lang = cur.fetchone()

        fname = str(idx) + '.jpg'
        fpath = self.save_dir / fname
        if fpath.exists():
            img = Image.open(str(fpath))
            w, h = img.size
            if w < 230 or h < 230:
                ratio = max(256 / w, 256 / h)
                img = img.resize((int(w * ratio), int(h * ratio)))
                img.save(fpath)
        else:
            img = download_image(url, save_path=fpath)

        if img is None:
            return None
        if self.image_transform is not None:
            img = self.image_transform(img)
        
        return img, caption, lang


class MemWikiImageDataset(Dataset):
    def __init__(self, csv_file, tokenizer, image_transform=None, max_tokens=128,
                 test=False, save_dir=None, download=False, num_langs=0, langs=None):
        super().__init__()
        if not download and save_dir is None:
            raise ValueError("Specify save_dir or set download=True")

        self.extension = '.jpg'
        if not download:
            ids_to_keep = set(int(f.stem) for f in save_dir.glob('*' + self.extension))
        else:
            ids_to_keep = set()
        
        self.index = []
        lang_freqs = defaultdict(int)
        with open(csv_file, newline='') as f:
            delimiter = ',' if not test else '\t'
            reader = csv.DictReader(f, delimiter=delimiter)
            for i, row in enumerate(reader):
                # if i > 1000000:
                #     break
                if int(row['id']) in ids_to_keep or download:
                    self.index.append(row)
                    if not test:
                        lang_freqs[row['lang']] += 1

        if langs is not None:
            langs_to_keep = set(langs)
            self.index = list(filter(lambda row: row['lang'] in langs_to_keep, self.index))
        elif num_langs > 0:
            langs_to_keep = set(sorted(lang_freqs.keys(), key=lambda x: lang_freqs[x], reverse=True)[:num_langs])
            self.index = list(filter(lambda row: row['lang'] in langs_to_keep, self.index))

        self.test = test
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.save_dir = save_dir
        self.download = download

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        row = self.index[idx]
        url_key = "url" if not self.test else "image_url"
        fname = str(row["id"]) + self.extension
        fpath = self.save_dir / fname
        if fpath.exists():
            try:
                img = Image.open(str(fpath))
            except ValueError:
                print(fpath)
                url_key = "url" if not self.test else "image_url"
                img = download_image(row[url_key], save_path=fpath)
                if img is None:
                    return None

            w, h = img.size
            if w < 230 or h < 230:
                ratio = max(256 / w, 256 / h)
                img = img.resize((int(w * ratio), int(h * ratio)))
                img.save(fpath)
        elif self.download:
            img = download_image(row[url_key], save_path=fpath)
            if img is None:
                return None
        else:
            raise RuntimeError("file {} missing, but download=False".format(fpath))

        if self.image_transform is not None:
            img_t = self.image_transform(img)
            img.close()
            img = img_t

        url_filename = Path(unquote(row[url_key])).stem
        url_tokens = self.tokenizer(url_filename, truncation=True, max_length=self.max_tokens).input_ids

        if self.test:
            return int(row['id']), img, url_tokens

        tokens = self.tokenizer(row['caption'], truncation=True, max_length=self.max_tokens).input_ids
        return img, tokens, url_tokens


def load_corpus(csv_file, langs=["en"]):
    texts = []
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f, delimiter=',')
        for i, row in enumerate(reader):
            if row['lang'] in langs:
                texts.append(row['caption'])
    return texts


def collate_ignore_nones(data):
    return default_collate(list(filter(lambda x: x is not None, data)))


class CollatePadMLM:
    def __init__(self, pad_token):
        self.pad_token = pad_token

    def __call__(self, batch):
        max_len = len(batch[0][0]) if isinstance(batch[0], tuple) else len(batch[0])
        padded_batch = []
        for seqs in batch:
            print(seqs.shape)
            seqs_tensor = torch.LongTensor(seqs)
            if seqs_tensor.ndim == 1:
                seqs_tensor = seqs_tensor.unsqueeze(0)
            padded_seq = F.pad(seqs_tensor, (0, max_len - seqs_tensor.shape[1]), value=self.pad_token)
            padded_batch.append(padded_seq)
        padded_batch = torch.stack(padded_batch, dim=1)
        padded_seqs = [t.squeeze(0) for t in torch.split(padded_batch, 1, dim=0)]
        if len(padded_seqs) == 1:
            return padded_seqs[0]
        else:
            return padded_seqs


class CollatePadClip:
    def __init__(self, pad_token):
        self.pad_token = pad_token

    def __call__(self, batch):
        imgs = []
        max_caption_len = 0
        max_url_len = 0
        for img, caption, url in batch:
            imgs.append(img)
            max_caption_len = max(max_caption_len, len(caption))
            max_url_len = max(max_url_len, len(url))
        
        padded_captions = [] 
        padded_urls = []
        for _, caption, url in batch:
            padded_captions.append(F.pad(
                torch.LongTensor(caption), 
                (0, max_caption_len - len(caption)), 
                value=self.pad_token,
            ))
            padded_urls.append(F.pad(
                torch.LongTensor(url), 
                (0, max_url_len - len(url)), 
                value=self.pad_token,
            ))
        
        imgs = torch.stack(imgs, dim=0)
        captions = torch.stack(padded_captions, dim=0)
        urls = torch.stack(padded_urls, dim=0)
        return imgs, captions, urls


class CollatePadClipTest:
    def __init__(self, pad_token):
        self.pad_token = pad_token

    def __call__(self, batch):
        ids = []
        imgs = []
        max_url_len = 0
        for id, img, url in batch:
            ids.append(id)
            imgs.append(img)
            max_url_len = max(max_url_len, len(url))
        
        padded_urls = []
        for _, _, url in batch:
            padded_urls.append(F.pad(
                torch.LongTensor(url), 
                (0, max_url_len - len(url)), 
                value=self.pad_token,
            ))
        
        ids = torch.LongTensor(ids)
        imgs = torch.stack(imgs, dim=0)
        urls = torch.stack(padded_urls, dim=0)
        return ids, imgs, urls
