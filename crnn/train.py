import os
import csv
import torch
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from dataset import CaptchaDataset, CHARS, collate_fn, IDX2CHAR, labels_to_text
from model import CRNN
from tqdm import tqdm

# === 파서 추가 ===
parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true', help='중단 지점부터 이어서 학습')
args = parser.parse_args()

# === 설정 ===
BATCH_SIZE = 32
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "best_crnn_model.pth"
CHECKPOINT_PATH = "checkpoint.pth"
criterion = nn.CTCLoss(blank=0, zero_infinity=True)

def decode_predictions(preds):
    pred_labels = preds.argmax(2).permute(1, 0)
    decoded_texts = []
    for pred in pred_labels:
        pred_str = ""
        prev_idx = -1
        for idx in pred:
            idx = idx.item()
            if idx != 0 and idx != prev_idx:
                pred_str += IDX2CHAR.get(idx, "")
            prev_idx = idx
        decoded_texts.append(pred_str)
    return decoded_texts

def train():
    train_dataset = CaptchaDataset("dataset")
    val_dataset = CaptchaDataset("dataset")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = CRNN(50, 1, len(CHARS) + 1, 256).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    start_epoch = 0
    best_val_loss = float("inf")
    train_losses, val_losses, val_accuracies = [], [], []

    if args.resume and os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"✅ 체크포인트에서 {start_epoch} 에포크부터 이어서 학습합니다")

    with open("train_log.csv", "a" if args.resume else "w", newline="") as f:
        log_writer = csv.writer(f)
        if not args.resume:
            log_writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val Accuracy"])

        for epoch in range(start_epoch, EPOCHS):
            model.train()
            train_loss = 0.0
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", ncols=100)
            for imgs, labels, label_lengths in loop:
                imgs, labels, label_lengths = imgs.to(DEVICE), labels.to(DEVICE), label_lengths.to(DEVICE)
                preds = model(imgs)
                preds_log_softmax = preds.log_softmax(2)
                preds_size = torch.full(size=(preds.size(1),), fill_value=preds.size(0), dtype=torch.int32).to(DEVICE)
                loss = criterion(preds_log_softmax, labels, preds_size, label_lengths)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                loop.set_postfix(loss=f"{loss.item():.4f}")
            avg_train_loss = train_loss / len(train_loader)

            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
                for imgs, labels, label_lengths in loop:
                    imgs, labels, label_lengths = imgs.to(DEVICE), labels.to(DEVICE), label_lengths.to(DEVICE)
                    preds = model(imgs)
                    preds_log_softmax = preds.log_softmax(2)
                    preds_size = torch.full(size=(preds.size(1),), fill_value=preds.size(0), dtype=torch.int32).to(DEVICE)
                    loss = criterion(preds_log_softmax, labels, preds_size, label_lengths)
                    val_loss += loss.item()
                    loop.set_postfix(val_loss=loss.item())

                    decoded_preds = decode_predictions(preds)
                    target_texts = []
                    start = 0
                    for length in label_lengths:
                        target_texts.append(labels_to_text(labels[start:start+length].cpu().numpy()))
                        start += length

                    for pred, target in zip(decoded_preds, target_texts):
                        if pred == target:
                            correct += 1
                        total += 1

            avg_val_loss = val_loss / len(val_loader)
            accuracy = correct / total
            val_accuracies.append(accuracy)
            log_writer.writerow([epoch+1, avg_train_loss, avg_val_loss, accuracy])
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # 체크포인트 저장
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, CHECKPOINT_PATH)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"✅ Best model saved with val loss {best_val_loss:.4f}")

    print(f"✅ 학습 완료! 최종 정확도: {val_accuracies[-1] * 100:.2f}%")
    os.system("python predict.py")

if __name__ == "__main__":
    train()
import os
import csv
import torch
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from dataset import CaptchaDataset, CHARS, collate_fn, IDX2CHAR, labels_to_text
from model import CRNN
from tqdm import tqdm

# === 파서 추가 ===
parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true', help='중단 지점부터 이어서 학습')
args = parser.parse_args()

# === 설정 ===
BATCH_SIZE = 32
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "best_crnn_model.pth"
CHECKPOINT_PATH = "checkpoint.pth"
criterion = nn.CTCLoss(blank=0, zero_infinity=True)

def decode_predictions(preds):
    pred_labels = preds.argmax(2).permute(1, 0)
    decoded_texts = []
    for pred in pred_labels:
        pred_str = ""
        prev_idx = -1
        for idx in pred:
            idx = idx.item()
            if idx != 0 and idx != prev_idx:
                pred_str += IDX2CHAR.get(idx, "")
            prev_idx = idx
        decoded_texts.append(pred_str)
    return decoded_texts

def train():
    train_dataset = CaptchaDataset("dataset")
    val_dataset = CaptchaDataset("dataset")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = CRNN(50, 1, len(CHARS) + 1, 256).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    start_epoch = 0
    best_val_loss = float("inf")
    train_losses, val_losses, val_accuracies = [], [], []

    if args.resume and os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"✅ 체크포인트에서 {start_epoch} 에포크부터 이어서 학습합니다")

    with open("train_log.csv", "a" if args.resume else "w", newline="") as f:
        log_writer = csv.writer(f)
        if not args.resume:
            log_writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val Accuracy"])

        for epoch in range(start_epoch, EPOCHS):
            model.train()
            train_loss = 0.0
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", ncols=100)
            for imgs, labels, label_lengths in loop:
                imgs, labels, label_lengths = imgs.to(DEVICE), labels.to(DEVICE), label_lengths.to(DEVICE)
                preds = model(imgs)
                preds_log_softmax = preds.log_softmax(2)
                preds_size = torch.full(size=(preds.size(1),), fill_value=preds.size(0), dtype=torch.int32).to(DEVICE)
                loss = criterion(preds_log_softmax, labels, preds_size, label_lengths)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                loop.set_postfix(loss=f"{loss.item():.4f}")
            avg_train_loss = train_loss / len(train_loader)

            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
                for imgs, labels, label_lengths in loop:
                    imgs, labels, label_lengths = imgs.to(DEVICE), labels.to(DEVICE), label_lengths.to(DEVICE)
                    preds = model(imgs)
                    preds_log_softmax = preds.log_softmax(2)
                    preds_size = torch.full(size=(preds.size(1),), fill_value=preds.size(0), dtype=torch.int32).to(DEVICE)
                    loss = criterion(preds_log_softmax, labels, preds_size, label_lengths)
                    val_loss += loss.item()
                    loop.set_postfix(val_loss=loss.item())

                    decoded_preds = decode_predictions(preds)
                    target_texts = []
                    start = 0
                    for length in label_lengths:
                        target_texts.append(labels_to_text(labels[start:start+length].cpu().numpy()))
                        start += length

                    for pred, target in zip(decoded_preds, target_texts):
                        if pred == target:
                            correct += 1
                        total += 1

            avg_val_loss = val_loss / len(val_loader)
            accuracy = correct / total
            val_accuracies.append(accuracy)
            log_writer.writerow([epoch+1, avg_train_loss, avg_val_loss, accuracy])
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # 체크포인트 저장
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, CHECKPOINT_PATH)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"✅ Best model saved with val loss {best_val_loss:.4f}")

    print(f"✅ 학습 완료! 최종 정확도: {val_accuracies[-1] * 100:.2f}%")
    os.system("python predict.py")

if __name__ == "__main__":
    train()
