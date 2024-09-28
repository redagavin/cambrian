import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPVisionModel, AutoImageProcessor, AutoModel
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm

class ImageNetSubset(Dataset):
    def __init__(self, dataset, preprocess_fn):
        self.dataset = dataset
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        label = item['label']
        pixel_values = self.preprocess_fn(image)
        return {'pixel_values': pixel_values, 'label': label}

def extract_features(model, dataloader, device):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            inputs = batch['pixel_values'].to(device)
            outputs = model(inputs)
            features.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
            labels.extend(batch['label'].numpy())
    return np.vstack(features), np.array(labels)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print("Loading ImageNet dataset...")
    dataset = load_dataset("imagenet-1k", split="validation")
    dataset = dataset.shuffle(seed=42).select(range(10000))  # Limit size for faster processing

    # CLIP preprocessing
    clip_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    def clip_preprocess(image):
        return clip_processor(images=image, return_tensors="pt")['pixel_values'][0]

    # DINOv2 preprocessing
    dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    def dinov2_preprocess(image):
        return dinov2_processor(images=image, return_tensors="pt")['pixel_values'][0]

    # Create datasets and dataloaders
    clip_dataset = ImageNetSubset(dataset, clip_preprocess)
    dinov2_dataset = ImageNetSubset(dataset, dinov2_preprocess)

    train_size = int(0.8 * len(dataset))
    clip_train_dataset = torch.utils.data.Subset(clip_dataset, range(train_size))
    clip_test_dataset = torch.utils.data.Subset(clip_dataset, range(train_size, len(dataset)))
    dinov2_train_dataset = torch.utils.data.Subset(dinov2_dataset, range(train_size))
    dinov2_test_dataset = torch.utils.data.Subset(dinov2_dataset, range(train_size, len(dataset)))

    clip_train_dataloader = DataLoader(clip_train_dataset, batch_size=32, shuffle=True, num_workers=4)
    clip_test_dataloader = DataLoader(clip_test_dataset, batch_size=32, num_workers=4)
    dinov2_train_dataloader = DataLoader(dinov2_train_dataset, batch_size=32, shuffle=True, num_workers=4)
    dinov2_test_dataloader = DataLoader(dinov2_test_dataset, batch_size=32, num_workers=4)

    # Evaluate CLIP ViT
    print("Evaluating CLIP ViT...")
    clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_train_features, clip_train_labels = extract_features(clip_model, clip_train_dataloader, device)
    clip_test_features, clip_test_labels = extract_features(clip_model, clip_test_dataloader, device)

    print("Training logistic regression for CLIP ViT...")
    clip_classifier = LogisticRegression(max_iter=1000, n_jobs=-1)
    clip_classifier.fit(clip_train_features, clip_train_labels)
    clip_predictions = clip_classifier.predict(clip_test_features)
    clip_accuracy = accuracy_score(clip_test_labels, clip_predictions)
    print(f"CLIP ViT Accuracy: {clip_accuracy:.4f}")

    # Evaluate DINOv2 ViT
    print("Evaluating DINOv2 ViT...")
    dinov2_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    dinov2_train_features, dinov2_train_labels = extract_features(dinov2_model, dinov2_train_dataloader, device)
    dinov2_test_features, dinov2_test_labels = extract_features(dinov2_model, dinov2_test_dataloader, device)

    print("Training logistic regression for DINOv2 ViT...")
    dinov2_classifier = LogisticRegression(max_iter=1000, n_jobs=-1)
    dinov2_classifier.fit(dinov2_train_features, dinov2_train_labels)
    dinov2_predictions = dinov2_classifier.predict(dinov2_test_features)
    dinov2_accuracy = accuracy_score(dinov2_test_labels, dinov2_predictions)
    print(f"DINOv2 ViT Accuracy: {dinov2_accuracy:.4f}")

if __name__ == "__main__":
    main()