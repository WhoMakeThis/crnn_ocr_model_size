import os
import csv
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from dataset import CaptchaDataset, CHARS, collate_fn, IDX2CHAR, labels_to_text
from model import CRNN
from tqdm import tqdm

print("✅ labels_to_text import 완료")

BATCH_SIZE = 16
EPOCHS = 30
DEVICE = torch.device("cpu")  # 명시적으로 CPU 사용
SAVE_PATH = "best_crnn_model.pth"
criterion = nn.CTCLoss(blank=0, zero_infinity=True)

SAMPLE_EPOCHS = [5, 10, 15, 20, 25, 30]
SAMPLE_DIR = "sample_predictions"
os.makedirs(SAMPLE_DIR, exist_ok=True)

def decode_predictions(preds, chars=CHARS):
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # 학습률 낮춤
    best_val_loss = float("inf")
    train_losses, val_losses, val_accuracies = [], [], []

    with open("train_log.csv", "w", newline="") as f:
        log_writer = csv.writer(f)
        log_writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val Accuracy"])

        for epoch in range(EPOCHS):
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
            sample_log = []
            with torch.no_grad():
                loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
                for batch_idx, (imgs, labels, label_lengths) in enumerate(loop):
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

                    if epoch + 1 in SAMPLE_EPOCHS and batch_idx == 0:
                        for i in range(min(5, len(decoded_preds))):
                            sample_log.append(f"[Sample {i+1}] 정답: {target_texts[i]} | 예측: {decoded_preds[i]}")

            avg_val_loss = val_loss / len(val_loader)
            accuracy = correct / total
            val_accuracies.append(accuracy)
            print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.4f}")
            log_writer.writerow([epoch+1, avg_train_loss, avg_val_loss, accuracy])
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"✅ Best model saved with val loss {best_val_loss:.4f}")

            if sample_log:
                with open(os.path.join(SAMPLE_DIR, f"epoch_{epoch+1:02d}_samples.txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(sample_log))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, EPOCHS + 1), val_losses, label='Val Loss', marker='x')
    plt.title("CRNN Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("CTC Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_plot.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), val_accuracies, label='Val Accuracy', marker='o', color='green')
    plt.title("Validation Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_plot.png")
    plt.close()

    print(f"✅ 학습 완료! 최종 정확도: {val_accuracies[-1] * 100:.2f}%")

if __name__ == "__main__":
    train()
    os.system("python predict.py")
