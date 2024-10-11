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
import argparse
import multiprocessing
import os

class CustomDataset(Dataset):
    def __init__(self, dataset, preprocess_fn, image_column='image', label_column='label'):
        self.dataset = dataset
        self.preprocess_fn = preprocess_fn
        self.image_column = image_column
        self.label_column = label_column

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.image_column].convert('RGB')
        label = item[self.label_column]
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
            if isinstance(outputs.last_hidden_state, torch.Tensor):
                features.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
            else:
                features.append(outputs[0][:, 0, :].cpu().numpy())
            labels.extend(batch['label'].numpy())
    return np.vstack(features), np.array(labels)

def load_and_prepare_dataset(dataset_name, preprocess_fn, split='train', max_samples=None, seed=42):
    if dataset_name == 'imagenet-1k':
        dataset = load_dataset("imagenet-1k", split=split)
        image_column, label_column = 'image', 'label'
    elif dataset_name == 'cifar100':
        dataset = load_dataset("cifar100", split=split)
        image_column, label_column = 'img', 'fine_label'
    elif dataset_name == 'sun397':
        dataset = load_dataset("tanganke/sun397", split=split)
        image_column, label_column = 'image', 'label'
    elif dataset_name == 'stanford_cars':
        if split == 'train':
            dataset = load_dataset("Multimodal-Fatima/StanfordCars_train", split='train')
        else:
            dataset = load_dataset("Multimodal-Fatima/StanfordCars_test", split='test')
        image_column, label_column = 'image', 'label'
    elif dataset_name == 'oxford_flowers':
        dataset = load_dataset("nelorth/oxford-flowers", split=split)
        image_column, label_column = 'image', 'label'
    elif dataset_name == 'food101':
        dataset = load_dataset("ethz/food101", split=split)
        image_column, label_column = 'image', 'label'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if max_samples and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))

    return CustomDataset(dataset, preprocess_fn, image_column=image_column, label_column=label_column)

def process_dataset(args, dataset_info, gpu_id):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    dataset_name, granularity = dataset_info
    
    if args.model_type == 'clip':
        model = CLIPVisionModel.from_pretrained(args.pretrained_model).to(device)
        processor = AutoImageProcessor.from_pretrained(args.pretrained_model)
    elif args.model_type == 'dinov2':
        model = AutoModel.from_pretrained(args.pretrained_model).to(device)
        processor = AutoImageProcessor.from_pretrained(args.pretrained_model)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    def preprocess_fn(image):
        return processor(images=image, return_tensors="pt")['pixel_values'][0]

    print(f"\nProcessing {args.model_type.upper()} ({args.pretrained_model}) on {dataset_name} ({granularity}) using GPU {gpu_id}...")
    train_dataset = load_and_prepare_dataset(dataset_name, preprocess_fn, split='train', max_samples=args.max_samples, seed=args.seed)
    test_dataset = load_and_prepare_dataset(dataset_name, preprocess_fn, split='validation' if dataset_name in ['imagenet-1k', 'food101'] else 'test', max_samples=args.max_samples, seed=args.seed)

    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0)

    train_features, train_labels = extract_features(model, train_dataloader, device)
    test_features, test_labels = extract_features(model, test_dataloader, device)

    return dataset_name, train_features, train_labels, test_features, test_labels

def evaluate_model(train_features, train_labels, test_features, test_labels):
    classifier = LogisticRegression(max_iter=1000, n_jobs=1)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    score = accuracy_score(test_labels, predictions)
    return score

def main(args):
    datasets = [
        ('imagenet-1k', 'Coarse-grained'),
        ('cifar100', 'Coarse-grained'),
        ('sun397', 'Coarse-grained'),
        ('stanford_cars', 'Fine-grained'),
        ('oxford_flowers', 'Fine-grained'),
        ('food101', 'Fine-grained')
    ]

    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    if num_gpus == 0:
        print("No GPUs available. Running on CPU.")
        results = [process_dataset(args, dataset, -1) for dataset in datasets]
    else:
        with multiprocessing.Pool(num_gpus) as pool:
            results = pool.starmap(process_dataset, [(args, dataset, i % num_gpus) for i, dataset in enumerate(datasets)])

    print("\nEvaluating models...")
    for dataset_name, train_features, train_labels, test_features, test_labels in results:
        score = evaluate_model(train_features, train_labels, test_features, test_labels)
        print(f"{args.model_type.upper()} ({args.pretrained_model}) Accuracy on {dataset_name}: {score:.4f}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Evaluate CLIP or DINOv2 on multiple tasks")
    parser.add_argument('--model_type', type=str, choices=['clip', 'dinov2'], required=True, help='Type of model to evaluate')
    parser.add_argument('--pretrained_model', type=str, required=True, help='Pretrained model to load (e.g., "openai/clip-vit-base-patch32" or "facebook/dinov2-base")')
    parser.add_argument('--max_samples', type=int, default=10000, help='Maximum number of samples to use per dataset (default: 10000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for dataset shuffling (default: 42)')
    args = parser.parse_args()
    main(args)