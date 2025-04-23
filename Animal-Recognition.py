!pip install datasets transformers torch torchvision

from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from PIL import Image

# ✅ Définir le chemin du dataset sur Kaggle
DATASET_PATH = "/kaggle/input/animal-image-dataset-90-different-animals/animals/animals"

# ✅ Charger le modèle et le feature extractor
model_id = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_id)

# ✅ Définir la transformation d'image (resize et normalisation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

# ✅ Charger le dataset avec ImageFolder (PyTorch détecte les classes automatiquement)
dataset = ImageFolder(root=DATASET_PATH, transform=transform)

# ✅ Obtenir la liste des classes (animaux)
labels = dataset.classes
num_labels = len(labels)

# ✅ Diviser les données en train/test (80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# ✅ Créer un DataLoader pour l'entraînement et le test
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# ✅ Charger ViT en mode multi-label classification
model = ViTForImageClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    problem_type="multi_label_classification"")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ Définir les arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./vit-animals",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=4,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    load_best_model_at_end=True,
)

# ✅ Fonction pour calculer les métriques
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (logits > 0).astype(int)
    f1 = f1_score(labels, predictions, average="macro")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1}

# ✅ Entraîner avec Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

train_results = trainer.train()

# ✅ Sauvegarder le modèle
trainer.save_model()

# ✅ Charger le modèle sauvegardé
model_path = "./vit-animals"
model = ViTForImageClassification.from_pretrained(model_path)
model.to(device)

# ✅ Fonction d'évaluation
def evaluate_model(model, dataloader):
    model.eval()  # Mode évaluation
    predictions = []
    true_labels = []
    
    with torch.no_grad():  # Désactive le calcul du gradient
        for batch in dataloader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            logits = outputs.logits  # Obtenir les logits
            preds = (logits > 0).cpu().numpy()  # Convertir en classes (0 ou 1)

            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    # Calculer la précision et le F1-score
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="macro")
    
    print(f"📊 Précision : {acc:.4f}")
    print(f"📊 F1-score : {f1:.4f}")

# ✅ Lancer l'évaluation sur les données de test
evaluate_model(model, test_loader)

# ✅ Prédiction sur une image de test
image_path = "/chemin/vers/une/image.jpg"
image = Image.open(image_path)
input_tensor = transform(image).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    outputs = model(input_tensor)
    logits = outputs.logits
    preds = (logits > 0).cpu().numpy().flatten()

for i, label in enumerate(labels):
    if preds[i] == 1:
        print(f"✅ L'image contient un : {label}")

# ✅ Sauvegarder le modèle en TorchScript
scripted_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224, device=device))
torch.jit.save(scripted_model, "vit_model.pt")
print("✅ Modèle sauvegardé en TorchScript !")

