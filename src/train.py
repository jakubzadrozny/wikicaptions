import os
from argparse import ArgumentParser
from pathlib import Path
import numpy as np

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from transformers import AutoTokenizer
# from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from dataset import MemWikiImageDataset, CollatePadClip as CollatePad, load_corpus
from models import ImageCaptionEncoder

data_dir = Path("data/")
models_dir = Path("models/")
imgs_dir = "train_images"
db_file = "captions_db.csv"

# langs = ['en', 'fr', 'es', 'it', 'de', 'pl', 'nl', 'ru']
vocab_size = 30000

train_ratio = 0.9
max_ds_size = 1000000000


# def retrain_tokenizer():
#     corpus = load_corpus(data_dir / db_file, langs=langs)
#     tokenizer = Tokenizer(models.BPE())
#     tokenizer.normalizer = normalizers.NFKC()
#     tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
#     tokenizer.decoder = decoders.ByteLevel()
#     trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=['<pad>', '<cls>'])
#     tokenizer.train_from_iterator(corpus, trainer=trainer)
#     tokenizer.save(str(models_dir / "tokenizer.json"))


def main(args):
    # retrain_tokenizer()
    # tokenizer = Tokenizer.from_file(str(models_dir / "tokenizer.json"))
    # pad_token = tokenizer.token_to_id("<pad>")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/distiluse-base-multilingual-cased-v1')
    pad_token = tokenizer.pad_token_id

    t = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    ds = MemWikiImageDataset(
        data_dir / db_file,
        tokenizer,
        image_transform=t, 
        save_dir=data_dir / imgs_dir,
    )

    N = len(ds)
    N_train = int(train_ratio * min(N, max_ds_size))
    pi = np.random.permutation(np.arange(N))
    train_ind = pi[:N_train]
    test_ind = pi[N_train:max_ds_size]
    ds_train = Subset(ds, train_ind)
    ds_test = Subset(ds, test_ind)
    print("Dataset length:", len(ds_train))

    train_loader = DataLoader(ds_train, batch_size=88, num_workers=8, shuffle=True, collate_fn=CollatePad(pad_token))
    test_loader = DataLoader(ds_test, batch_size=88, num_workers=8, shuffle=False, collate_fn=CollatePad(pad_token))

    model = ImageCaptionEncoder(lr=3e-5, multilingual=True, entropy_weight=0.1)
    # pretrained_path = "models/clip/lightning_logs/version_26/checkpoints/latest.ckpt"
    # pretrained = ImageCaptionEncoder.load_from_checkpoint(pretrained_path, multilingual=False)
    # model.vision_model = pretrained.vision_model

    train_checkpoint = ModelCheckpoint(
        every_n_train_steps=500,
        save_top_k=1,
    )
    best_checkpoint = ModelCheckpoint(
        monitor='val_clf',
        mode='min',
        save_top_k=-1,
    )
    trainer = Trainer.from_argparse_args(
        args, 
        default_root_dir=str(models_dir), 
        gradient_clip_val=0.5,
        callbacks=[train_checkpoint, best_checkpoint],
    )
    trainer.fit(model, train_loader, test_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)