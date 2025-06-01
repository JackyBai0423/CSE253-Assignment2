import os
import torchaudio
from audiocraft.models import MusicGen
from transformers import get_scheduler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import random
import wandb
import json
from torch.utils.data import Dataset
from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout



class AudioDataset(Dataset):
    def __init__(self, jsonl_path, base_path=None):
        self.entries = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if base_path is not None:
                    entry["path"] = os.path.join(base_path, entry["path"])
                self.entries.append(entry)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        return entry["path"], entry["description"]


def preprocess_audio(audio_path, model: MusicGen, duration: int = 30):
    wav, sr = torchaudio.load(audio_path)
    wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
    wav = wav.mean(dim=0, keepdim=True)
    if wav.shape[1] < model.sample_rate * duration:
        return None
    end_sample = int(model.sample_rate * duration)
    start_sample = random.randrange(0, max(wav.shape[1] - end_sample, 1))
    wav = wav[:, start_sample : start_sample + end_sample]

    wav = wav.cuda().unsqueeze(1)
    with torch.no_grad():
        gen_audio = model.compression_model.encode(wav)
    codes, scale = gen_audio
    return codes


def one_hot_encode(tensor, num_classes=2048):
    shape = tensor.shape
    one_hot = torch.zeros((shape[0], shape[1], num_classes), device=tensor.device)
    for i in range(shape[0]):
        for j in range(shape[1]):
            index = tensor[i, j].item()
            one_hot[i, j, index] = 1
    return one_hot


def train(
    dataset_path: str,
    model_id: str,
    lr: float,
    epochs: int,
    use_wandb: bool,
    no_label: bool = False,
    tune_text: bool = False,
    save_step: int = None,
    grad_acc: int = 8,
    use_scaler: bool = False,
    weight_decay: float = 1e-5,
    warmup_steps: int = 10,
    batch_size: int = 10,
    use_cfg: bool = False
):
    if use_wandb:
        run = wandb.init(project="audiocraft")

    model = MusicGen.get_pretrained(model_id)
    model.lm = model.lm.to(torch.float32)

    dataset = AudioDataset(dataset_path)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(
        model.lm.condition_provider.parameters() if tune_text else model.lm.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )
    scheduler = get_scheduler(
        "cosine",
        optimizer,
        warmup_steps,
        int(epochs * len(train_dataloader) / grad_acc),
    )

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler()

    save_models = save_step is not None
    os.makedirs("models/", exist_ok=True)
    current_step = 0

    for epoch in range(epochs):
        for batch_idx, (audio, text) in enumerate(train_dataloader):
            optimizer.zero_grad()
            all_codes = []
            texts = []

            for audio_path, label in zip(audio, text):
                codes = preprocess_audio(audio_path, model)
                if codes is None:
                    continue
                codes = torch.cat([codes, codes], dim=0) if use_cfg else codes
                all_codes.append(codes)
                texts.append(label)

            if len(all_codes) == 0:
                continue

            attributes, _ = model._prepare_tokens_and_attributes(texts, None)
            conditions = attributes
            if use_cfg:
                null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
                conditions = conditions + null_conditions

            tokenized = model.lm.condition_provider.tokenize(conditions)
            condition_tensors = model.lm.condition_provider(tokenized)

            codes = torch.cat(all_codes, dim=0)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                lm_output = model.lm.compute_predictions(
                    codes=codes, conditions=[], condition_tensors=condition_tensors
                )

                logits = lm_output.logits[0]
                mask = lm_output.mask[0]
                targets = one_hot_encode(codes[0], num_classes=2048)

                logits = logits.cuda()
                mask = mask.cuda().view(-1)
                masked_logits = logits.view(-1, 2048)[mask]
                masked_targets = targets.view(-1, 2048)[mask]

                loss = criterion(masked_logits, masked_targets)

            current_step += 1 / grad_acc
            (scaler.scale(loss) if use_scaler else loss).backward()

            if use_wandb:
                wandb.log({"loss": loss.item()})

            print(f"Epoch {epoch} Batch {batch_idx}: Loss = {loss.item():.4f}")

            if batch_idx % grad_acc != grad_acc - 1:
                continue

            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.lm.parameters(), 0.5)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()

            if save_models and int(current_step) % save_step == 0:
                torch.save(model.lm.state_dict(), f"models/lm_{int(current_step)}.pt")

    torch.save(model.lm.state_dict(), "models/lm_final.pt")


if __name__ == "__main__":
    train(
        dataset_path="train_gpt.jsonl",
        model_id="facebook/musicgen-medium",
        lr=1e-4,
        epochs=5,
        use_wandb=False,
        no_label=False,
        tune_text=False,
        save_step=100,
        grad_acc=4,
        batch_size=1,
        use_scaler=True,
        use_cfg=True,
    )
