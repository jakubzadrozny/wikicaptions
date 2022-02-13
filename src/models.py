import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
import torchvision
from transformers import DistilBertConfig, DistilBertModel, AutoModel
import pytorch_lightning as pl


def dict_to_device(d, device):
    for key in d:
        if isinstance(d[key], torch.Tensor):
            d[key] = d[key].to(device)
    return d


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


class ImageCaptionEncoder(pl.LightningModule):
    def __init__(self, vocab_size=51000, hidden_dim=768, entropy_weight=1.0, lr=1e-4,
                 normalize_features=False, use_cosine=True, multilingual=True):
        super().__init__()

        # config = DistilBertConfig(
        #     vocab_size=vocab_size,
        #     hidden_dim=hidden_dim,
        #     n_layers=8,
        #     n_heads=15,
        # )
        # self.text_model = DistilBertModel(config)
        if multilingual:
            self.text_model = AutoModel.from_pretrained('sentence-transformers/distiluse-base-multilingual-cased-v1')
        else:
            self.text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.captions_proj = nn.Linear(hidden_dim, hidden_dim)
        self.urls_proj = nn.Linear(hidden_dim, hidden_dim)

        self.vision_model = torchvision.models.resnet50(pretrained=True)
        self.vision_model.fc = nn.Linear(2048, hidden_dim)
        # self.vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        # self.vision_proj = nn.Linear(self.vision_model.config.hidden_size, hidden_size)
        # freeze(self.vision_model)

        self.criterion = nn.NLLLoss()

        self.entropy_weight = entropy_weight
        self.use_cosine = use_cosine
        self.normalize_features = normalize_features
        self.log_url_weight = nn.parameter.Parameter(torch.tensor(0.))
        self.t_captions = nn.parameter.Parameter(torch.tensor(0.))
        self.t_urls = nn.parameter.Parameter(torch.tensor(0.))
        self.lr = lr


    def normalize(self, features):
        features_len = torch.sqrt(torch.sum(features ** 2, dim=-1, keepdim=True))
        features_normalized = features / features_len
        return features_normalized

        
    def encode_images(self, images):
        # vision_outputs = self.vision_model(pixel_values=images)[1]    
        # features = self.vision_proj(vision_outputs)
        features = self.vision_model(images)
        if self.normalize_features:
            features = self.normalize(features)
        return features
        
        
    def encode_text(self, tokens, proj):
        text_outputs = self.text_model(tokens)
        features = text_outputs.last_hidden_state[:, 0, :]
        features = proj(features)
        if self.normalize_features:
            features = self.normalize(features)
        return features


    def get_logits(self, features1, features2, t):
        if self.use_cosine:
            return torch.mm(features1, features2.T) * torch.exp(t)
        else:
            return torch.cdist(features1, features2) * torch.exp(t)


    def get_losses(self, logits):
        bs = logits.shape[0]
        log_probs = F.log_softmax(logits, dim=1)
        labels = torch.arange(bs).to(logits.device)
        clf_loss = self.criterion(log_probs, labels)
        probs = torch.exp(log_probs)
        entropy = torch.mean(-torch.sum(probs * log_probs, dim=-1))
        preds = torch.argmax(log_probs, dim=-1)
        accuracy = torch.sum(preds == labels) / bs
        return clf_loss, entropy, accuracy

    
    def process_batch(self, batch):
        images, captions, urls = batch
        
        image_features = self.encode_images(images)
        caption_features = self.encode_text(captions, self.captions_proj)
        url_features = self.encode_text(urls, self.urls_proj)

        logits_images = self.get_logits(caption_features, image_features, self.t_captions)
        logits_urls = self.get_logits(caption_features, url_features, self.t_urls)
        logits = logits_images + torch.exp(self.log_url_weight) * logits_urls

        clf_loss_text, entropy_text, accuracy_text = self.get_losses(logits)
        clf_loss_images, entropy_images, accuracy_images = self.get_losses(logits.T)

        clf_loss = (clf_loss_text + clf_loss_images) / 2
        entropy = (entropy_text + entropy_images) / 2
        accuracy = (accuracy_text + accuracy_images) / 2
        loss = clf_loss + self.entropy_weight * entropy
        return loss, clf_loss, entropy, accuracy


    def training_step(self, batch, batch_idx):
        batch_size = batch[0].shape[0]
        loss, clf_loss, entropy, accuracy = self.process_batch(batch)
        
        self.log('train_clf', clf_loss, on_step=True, on_epoch=True, logger=True, batch_size=batch_size, prog_bar=True)
        self.log('train_entropy', entropy, on_step=True, on_epoch=True, logger=True, batch_size=batch_size, prog_bar=True)
        self.log('train_acc', accuracy, on_step=True, on_epoch=True, logger=True, batch_size=batch_size, prog_bar=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, batch_size=batch_size)
        return loss


    def validation_step(self, batch, batch_idx):
        batch_size = batch[0].shape[0]
        _, clf_loss, entropy, accuracy = self.process_batch(batch)
        
        self.log('val_clf', clf_loss, on_epoch=True, logger=True, batch_size=batch_size)
        self.log('val_entropy', entropy, on_epoch=True, logger=True, batch_size=batch_size)
        self.log('val_acc', accuracy, on_epoch=True, logger=True, batch_size=batch_size)
        return clf_loss


    def test_step(self, batch, batch_idx):
        batch_size = batch[0].shape[0]
        _, clf_loss, entropy, accuracy = self.process_batch(batch)
        
        self.log('test_clf', clf_loss, on_epoch=True, logger=True, batch_size=batch_size)
        self.log('test_entropy', entropy, on_epoch=True, logger=True, batch_size=batch_size)
        self.log('test_acc', accuracy, on_epoch=True, logger=True, batch_size=batch_size)
        return clf_loss


    def configure_optimizers(self):
        params = sum([list(m.parameters()) for m in [
            self.vision_model,
            self.text_model,
            self.captions_proj,
            self.urls_proj,
        ]], []) + [self.log_url_weight, self.t_captions, self.t_urls]
        return Adam(params, lr=self.lr)
