import os
import csv
from pathlib import Path
from tqdm import trange, tqdm

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from dataset import MemWikiImageDataset, CollatePadClipTest as CollatePad
from models import ImageCaptionEncoder


data_dir = Path("data")
img_dir = Path("test_images")
captions_file = "test_caption_list.csv"
db_file = "test.tsv"


def read_captions():
    captions = []
    with open(data_dir / captions_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            captions.append(row['caption_title_and_reference_description'])
    return captions


def compute_caption_features(model, tokenizer, device='cpu', bs=128):
    captions = read_captions()
    features = []
    with torch.no_grad():
        for i in trange(0, len(captions), bs):
            captions_batch = captions[i:i+bs]
            tokens = tokenizer(
                captions_batch, 
                padding=True, 
                return_tensors='pt', 
                truncation=True, 
                max_length=128
            ).input_ids.to(device)
            features.append(model.encode_text(tokens, model.captions_proj).cpu())

    all_features = torch.cat(features, dim=0)
    torch.save(all_features, data_dir / "test_caption_features")
    return all_features


def compute_image_features(model, tokenizer, device='cpu', bs=128):
    t = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    ds = MemWikiImageDataset(
        data_dir / db_file,
        tokenizer,
        image_transform=t, 
        save_dir=data_dir / img_dir,
        test=True,
    )
    print("Image dataset length:", len(ds))

    pad_token = tokenizer.pad_token_id
    loader = DataLoader(ds, batch_size=bs, num_workers=8, shuffle=False, collate_fn=CollatePad(pad_token))

    img_features = []
    url_features = []
    ids = []
    with torch.no_grad():
        for batch_ids, imgs, urls in tqdm(loader):
            imgs = imgs.to(device)
            urls = urls.to(device)
            img_features.append(model.encode_images(imgs).cpu())
            url_features.append(model.encode_text(urls, model.urls_proj).cpu())
            ids.append(torch.LongTensor(batch_ids))

    all_features = {
        "img_features": torch.cat(img_features, dim=0),
        "url_features": torch.cat(url_features, dim=0),
        "ids": torch.cat(ids), 
    }
    torch.save(all_features, data_dir / "test_image_features")
    return all_features


def match_captions(model, bs=8, device='cpu'):
    images_data = torch.load(data_dir / "test_image_features")
    img_features = images_data["img_features"]
    url_features = images_data["url_features"]
    image_ids = images_data["ids"].numpy()
    
    caption_features = torch.load(data_dir / "test_caption_features").to(device)

    matched_image_ids = []
    matched_caption_ids = []
    for i in trange(0, image_ids.shape[0], bs):
        img_features_chunk = img_features[i:i+bs].to(device)
        url_features_chunk = url_features[i:i+bs].to(device)

        logits_images = model.get_logits(caption_features, img_features_chunk, model.t_captions)
        logits_urls = model.get_logits(caption_features, url_features_chunk, model.t_urls)
        logits = logits_images + torch.exp(model.log_url_weight) * logits_urls

        probs = F.softmax(logits, dim=0)
        best_probs, best_ind = torch.topk(probs, 5, dim=0)
        for j in range(img_features_chunk.shape[0]):
            for k in range(5):
                matched_image_ids.append(image_ids[i+j])
                matched_caption_ids.append(best_ind[k, j])

    captions = read_captions()

    with open(data_dir / "submission.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'caption_title_and_reference_description'])
        writer.writeheader()
        for image_id, caption_id in zip(matched_image_ids, matched_caption_ids):
            writer.writerow({
                'id': image_id,
                'caption_title_and_reference_description': captions[caption_id],
            })


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/distiluse-base-multilingual-cased-v1')
    
    ckpt_path = "models/lightning_logs/version_23/checkpoints/epoch=7-step=268999.ckpt"
    model = ImageCaptionEncoder.load_from_checkpoint(ckpt_path)
    model = model.eval().to(device)

    compute_image_features(model, tokenizer, device=device, bs=256)
    compute_caption_features(model, tokenizer, device=device, bs=256)
    match_captions(model, device=device)