"""
Évaluation du modèle et explicabilité via Grad-CAM.

Fournit :
- Métriques complètes (accuracy, kappa, AUC, sensibilité/spécificité par classe)
- Matrice de confusion normalisée
- Courbes ROC multi-classes
- Grad-CAM pour la visualisation des zones d'attention
- Rapport d'évaluation complet (sauvegardé en JSON)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


# ------------------------------------------------------------------ #
#  Métriques d'évaluation                                              #
# ------------------------------------------------------------------ #

def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> Dict:
    """
    Calcule un ensemble complet de métriques d'évaluation.
    
    Args:
        y_true: Labels réels (N,) entiers.
        y_pred_proba: Probabilités prédites (N, num_classes).
        
    Returns:
        Dictionnaire structuré avec toutes les métriques.
    """
    y_pred = np.argmax(y_pred_proba, axis=1)
    num_classes = y_pred_proba.shape[1]

    # Métriques globales
    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    # AUC multi-classe (one-vs-rest)
    y_true_onehot = np.eye(num_classes)[y_true]
    try:
        auc_macro = roc_auc_score(y_true_onehot, y_pred_proba, multi_class="ovr", average="macro")
        auc_per_class = roc_auc_score(y_true_onehot, y_pred_proba, multi_class="ovr", average=None)
    except ValueError:
        auc_macro = 0.0
        auc_per_class = [0.0] * num_classes

    # Rapport par classe
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )

    # Sensibilité clinique : détection des cas référables (grade >= 2)
    referable = (y_true >= 2).astype(int)
    pred_referable = (y_pred >= 2).astype(int)
    sensitivity_referable = np.sum((pred_referable == 1) & (referable == 1)) / max(np.sum(referable), 1)
    specificity_referable = np.sum((pred_referable == 0) & (referable == 0)) / max(np.sum(1 - referable), 1)

    # Matrice de confusion
    conf_matrix = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "global": {
            "accuracy": float(accuracy),
            "quadratic_weighted_kappa": float(kappa),
            "auc_macro": float(auc_macro),
            "sensitivity_referable": float(sensitivity_referable),
            "specificity_referable": float(specificity_referable),
            "total_samples": int(len(y_true)),
        },
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(report[CLASS_NAMES[i]]["precision"]),
                "recall": float(report[CLASS_NAMES[i]]["recall"]),
                "f1_score": float(report[CLASS_NAMES[i]]["f1-score"]),
                "auc": float(auc_per_class[i]) if isinstance(auc_per_class, (list, np.ndarray)) else 0.0,
                "support": int(report[CLASS_NAMES[i]]["support"]),
            }
            for i in range(num_classes)
        },
        "confusion_matrix": conf_matrix,
    }

    return metrics


def generate_evaluation_report(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    output_dir: str,
    tta: bool = False,
) -> Dict:
    """
    Génère un rapport d'évaluation complet et sauvegarde les artefacts.
    
    Args:
        model: Modèle entraîné.
        test_dataset: Dataset de test (tf.data).
        output_dir: Répertoire de sortie pour le rapport.
        tta: Activer le Test-Time Augmentation.
        
    Returns:
        Dictionnaire des métriques.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collecte des prédictions
    all_labels = []
    all_predictions = []

    if tta:
        from src.models.optimize import TestTimeAugmentation
        tta_predictor = TestTimeAugmentation(model, n_augmentations=8)

    for images, labels in test_dataset:
        if tta:
            preds = tta_predictor.predict(images.numpy())
        else:
            preds = model.predict(images, verbose=0)

        all_labels.append(np.argmax(labels.numpy(), axis=1))
        all_predictions.append(preds)

    y_true = np.concatenate(all_labels)
    y_pred_proba = np.concatenate(all_predictions)

    # Calcul des métriques
    metrics = compute_comprehensive_metrics(y_true, y_pred_proba)

    # Sauvegarde du rapport JSON
    report_path = output_path / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Rapport sauvegardé : %s", report_path)
    logger.info(
        "Résultats — Accuracy: %.3f, QWK: %.3f, AUC: %.3f, "
        "Sensibilité (réf.): %.3f, Spécificité (réf.): %.3f",
        metrics["global"]["accuracy"],
        metrics["global"]["quadratic_weighted_kappa"],
        metrics["global"]["auc_macro"],
        metrics["global"]["sensitivity_referable"],
        metrics["global"]["specificity_referable"],
    )

    return metrics


# ------------------------------------------------------------------ #
#  Grad-CAM pour l'explicabilité                                       #
# ------------------------------------------------------------------ #

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Génère des cartes de chaleur montrant les régions de l'image qui
    contribuent le plus à la décision du modèle. Essentiel pour :
    - La confiance clinique (le modèle regarde-t-il les bonnes zones ?)
    - Le debugging (détection de biais ou de raccourcis)
    - La communication avec les radiologues
    
    Référence :
        Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
        via Gradient-based Localization" (2017).
    
    Exemple :
        >>> gradcam = GradCAM(model, layer_name="top_conv")
        >>> heatmap = gradcam.compute_heatmap(image)
        >>> overlay = gradcam.overlay_heatmap(heatmap, original_image)
    """

    def __init__(
        self,
        model: tf.keras.Model,
        layer_name: Optional[str] = None,
    ):
        self.model = model

        # Identification de la couche cible
        if layer_name:
            self.target_layer = self._find_layer(layer_name)
        else:
            self.target_layer = self._find_last_conv_layer()

        # Sous-modèle pour l'extraction
        self.grad_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[self.target_layer.output, self.model.output],
        )

        logger.info("Grad-CAM initialisé — couche cible : %s", self.target_layer.name)

    def _find_layer(self, name: str):
        """Recherche récursive d'une couche par nom."""
        for layer in self.model.layers:
            if layer.name == name:
                return layer
            if hasattr(layer, "layers"):
                for sub_layer in layer.layers:
                    if sub_layer.name == name:
                        return sub_layer
        raise ValueError(f"Couche '{name}' non trouvée")

    def _find_last_conv_layer(self):
        """Trouve la dernière couche convolutive du modèle."""
        last_conv = None
        for layer in self.model.layers:
            if hasattr(layer, "layers"):
                for sub_layer in layer.layers:
                    if "conv" in sub_layer.name.lower():
                        last_conv = sub_layer
            elif "conv" in layer.name.lower():
                last_conv = layer

        if last_conv is None:
            raise ValueError("Aucune couche convolutive trouvée")
        return last_conv

    @tf.function
    def _compute_gradients(
        self, image: tf.Tensor, class_idx: Optional[int] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Calcule les gradients de la classe cible par rapport aux feature maps."""
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image, training=False)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        return conv_outputs, grads

    def compute_heatmap(
        self,
        image: np.ndarray,
        class_idx: Optional[int] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Calcule la carte de chaleur Grad-CAM pour une image.
        
        Args:
            image: Image d'entrée (H, W, 3) ou (1, H, W, 3).
            class_idx: Classe cible (None = classe prédite).
            normalize: Normaliser la heatmap dans [0, 1].
            
        Returns:
            Carte de chaleur (H, W) dans [0, 1].
        """
        if image.ndim == 3:
            image = np.expand_dims(image, 0)

        image_tensor = tf.cast(image, tf.float32)
        conv_outputs, grads = self._compute_gradients(image_tensor, class_idx)

        # Global Average Pooling des gradients
        weights = tf.reduce_mean(grads, axis=(1, 2))

        # Combinaison linéaire pondérée des feature maps
        cam = tf.reduce_sum(
            conv_outputs * tf.reshape(weights, [1, 1, 1, -1]),
            axis=-1,
        )

        # ReLU : on ne garde que les contributions positives
        cam = tf.nn.relu(cam)
        cam = cam.numpy()[0]

        # Resize à la taille de l'image originale
        cam = cv2.resize(cam, (image.shape[2], image.shape[1]))

        if normalize and cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam

    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        original_image: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Superpose la heatmap Grad-CAM sur l'image originale.
        
        Args:
            heatmap: Carte de chaleur (H, W) dans [0, 1].
            original_image: Image originale BGR (H, W, 3).
            alpha: Transparence de la heatmap.
            colormap: Colormap OpenCV.
            
        Returns:
            Image avec superposition (H, W, 3) en uint8.
        """
        # Conversion en colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap), colormap,
        )

        # Gestion de la normalisation de l'image originale
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)

        if original_image.shape[:2] != heatmap_colored.shape[:2]:
            heatmap_colored = cv2.resize(
                heatmap_colored, (original_image.shape[1], original_image.shape[0]),
            )

        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay

    def generate_explanations(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        output_dir: str,
        n_samples: int = 20,
    ) -> None:
        """
        Génère des visualisations Grad-CAM pour un échantillon d'images.
        
        Sauvegarde les overlays pour inspection visuelle par les cliniciens.
        """
        output_path = Path(output_dir) / "gradcam"
        output_path.mkdir(parents=True, exist_ok=True)

        indices = np.random.choice(len(images), min(n_samples, len(images)), replace=False)

        for idx in indices:
            image = images[idx]
            label = labels[idx]

            heatmap = self.compute_heatmap(image)
            overlay = self.overlay_heatmap(heatmap, image)

            # Prédiction
            pred = self.model.predict(np.expand_dims(image, 0), verbose=0)
            pred_class = np.argmax(pred)
            confidence = float(pred[0, pred_class])

            # Sauvegarde
            filename = f"sample_{idx}_true_{CLASS_NAMES[label]}_pred_{CLASS_NAMES[pred_class]}_{confidence:.2f}.png"
            cv2.imwrite(str(output_path / filename), overlay)

        logger.info(
            "%d visualisations Grad-CAM sauvegardées dans %s",
            len(indices), output_path,
        )
