import os
import csv
import torch
import matplotlib.pyplot as plt
from PIL import Image
from model import CRNN
from dataset import CHARS
import torchvision.transforms as transforms
from difflib import SequenceMatcher
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import multiprocessing

DEVICE = torch.device("cpu")
IDX2CHAR = {idx + 1: char for idx, char in enumerate(CHARS)}
IDX2CHAR[0] = ""

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((50, 150)),
    transforms.ToTensor()
])

class CaptchaTestDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")
        image = transform(image)
        filename = os.path.basename(img_path)
        label = os.path.splitext(filename)[0][:5]
        return image, label, filename

def decode_prediction(preds):
    preds = preds.permute(1, 0, 2)
    pred_labels = torch.argmax(preds, dim=2)
    decoded_texts = []
    for pred in pred_labels:
        pred_str = ""
        prev = -1
        for p in pred:
            p = p.item()
            if p != prev and p != 0:
                pred_str += IDX2CHAR.get(p, "")
            prev = p
        decoded_texts.append(pred_str)
    return decoded_texts

def char_accuracy(pred, truth):
    correct = sum(p == t for p, t in zip(pred, truth))
    return correct / max(len(truth), 1)

def calc_similarity(pred, truth):
    return SequenceMatcher(None, pred, truth).ratio()

if __name__ == '__main__':
    multiprocessing.freeze_support()

    model = CRNN(50, 1, len(CHARS) + 1, 256).to(DEVICE)
    model.load_state_dict(torch.load("best_crnn_model.pth", map_location=DEVICE))
    model.eval()

    test_dir = "dataset"
    image_paths = sorted([
        os.path.join(test_dir, fname) for fname in os.listdir(test_dir)
        if fname.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    dataset = CaptchaTestDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

    results = []
    with torch.no_grad():
        for images, labels, filenames in tqdm(dataloader, desc="Predicting"):
            images = images.to(DEVICE)
            preds = model(images)
            decoded_texts = decode_prediction(preds)
            for pred, truth, fname in zip(decoded_texts, labels, filenames):
                is_correct = (pred == truth)
                acc = round(char_accuracy(pred, truth), 3)
                similarity = round(calc_similarity(pred, truth), 3)
                results.append({
                    "filename": fname,
                    "ground_truth": truth,
                    "prediction": pred,
                    "is_correct": is_correct,
                    "char_accuracy": acc,
                    "similarity": similarity
                })

    total = len(results)
    correct = sum(r["is_correct"] for r in results)
    avg_char_acc = sum(r["char_accuracy"] for r in results) / total
    avg_similarity = sum(r["similarity"] for r in results) / total

    print("\n=== 예측 요약 ===")
    print(f"✅ 전체 이미지 수: {total}")
    print(f"✅ 정확히 맞춘 개수: {correct} ({correct / total * 100:.2f}%)")
    print(f"✅ 문자 단위 정확도 평균: {avg_char_acc * 100:.2f}%")
    print(f"✅ 유사도 평균: {avg_similarity * 100:.2f}%")

    plt.figure(figsize=(8, 6))
    labels = ['문자 정답률', '문자 유사도', '완전 정답률']
    values = [avg_char_acc * 100, avg_similarity * 100, (correct / total) * 100]
    bars = plt.bar(labels, values)
    plt.ylim(0, 100)
    plt.ylabel("정확도 (%)")
    plt.title("CRNN 예측 정확도 요약")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.2f}%', ha='center')
    plt.tight_layout()
    plt.savefig("accuracy_plot.png")
    plt.close()

    csv_path = "crnn_prediction_results_with_accuracy.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "ground_truth", "prediction", "is_correct", "char_accuracy", "similarity"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"✅ 예측 결과가 {csv_path}에 저장되었습니다.")
