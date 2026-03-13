"""
Pipeline d'entraînement avec suivi MLflow.

Implémente une stratégie d'entraînement en deux phases :
  Phase 1 — Warmup : backbone gelé, head uniquement (5-10 epochs)
  Phase 2 — Fine-tuning : dégel progressif du backbone (20-50 epochs)

Toutes les métriques, paramètres et artefacts sont trackés via MLflow.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
import yaml

from src.data.dataset import RetinalDataset
from src.models.architecture import RetinalClassifier
from src.models.losses import WeightedFocalLoss, QuadraticWeightedKappaLoss

logger = logging.getLogger(__name__)


class MLflowCallback(tf.keras.callbacks.Callback):
    """
    Callback Keras pour le logging MLflow en temps réel.
    
    Log automatiquement les métriques d'entraînement et de validation
    à chaque epoch, ainsi que le learning rate courant.
    """

    def on_epoch_end(self, epoch: int, logs: dict = None):
        if logs is None:
            return

        for key, value in logs.items():
            mlflow.log_metric(key, float(value), step=epoch)

        # Log du learning rate courant
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        mlflow.log_metric("learning_rate", lr, step=epoch)


class QuadraticKappaMetric(tf.keras.metrics.Metric):
    """
    Métrique Keras pour le Quadratic Weighted Kappa.
    
    Calcule le QWK sur les prédictions accumulées au cours d'une epoch,
    permettant un suivi précis de la métrique de référence.
    """

    def __init__(self, num_classes: int = 5, name: str = "qw_kappa", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion = self.add_weight(
            "confusion",
            shape=(num_classes, num_classes),
            initializer="zeros",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_idx = tf.argmax(y_true, axis=-1)
        y_pred_idx = tf.argmax(y_pred, axis=-1)

        new_conf = tf.math.confusion_matrix(
            y_true_idx, y_pred_idx, num_classes=self.num_classes, dtype=tf.float32,
        )
        self.confusion.assign_add(new_conf)

    def result(self):
        total = tf.reduce_sum(self.confusion)
        if total == 0:
            return 0.0

        norm_conf = self.confusion / total

        hist_true = tf.reduce_sum(norm_conf, axis=1)
        hist_pred = tf.reduce_sum(norm_conf, axis=0)
        expected = tf.tensordot(hist_true, hist_pred, axes=0)

        n = self.num_classes
        weights = tf.constant(
            [[float((i - j) ** 2) / float((n - 1) ** 2) for j in range(n)] for i in range(n)],
            dtype=tf.float32,
        )

        numerator = tf.reduce_sum(weights * norm_conf)
        denominator = tf.reduce_sum(weights * expected) + tf.keras.backend.epsilon()

        return 1.0 - numerator / denominator

    def reset_state(self):
        self.confusion.assign(tf.zeros_like(self.confusion))


class TrainingPipeline:
    """
    Pipeline d'entraînement complet pour le modèle de rétinopathie.
    
    Orchestre le chargement des données, la construction du modèle,
    l'entraînement en deux phases et le suivi MLflow.
    
    Exemple :
        >>> config = load_config("configs/train_config.yaml")
        >>> pipeline = TrainingPipeline(config)
        >>> pipeline.run(experiment_name="retinai-v2", run_name="focal-loss")
    """

    def __init__(self, config: dict):
        self.config = config
        self._setup_gpu()

    def _setup_gpu(self) -> None:
        """Configure la mémoire GPU (growth mode pour éviter l'OOM)."""
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU détecté : %s", [g.name for g in gpus])
        else:
            logger.warning("Aucun GPU détecté — entraînement sur CPU")

    def _build_callbacks(self, output_dir: str) -> list:
        """Construit la liste des callbacks Keras."""
        callbacks = [
            # Suivi MLflow
            MLflowCallback(),
            # Early stopping sur le kappa
            tf.keras.callbacks.EarlyStopping(
                monitor="val_qw_kappa",
                patience=self.config.get("early_stopping_patience", 10),
                mode="max",
                restore_best_weights=True,
                verbose=1,
            ),
            # Réduction du learning rate sur plateau
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1,
            ),
            # Sauvegarde du meilleur modèle
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(output_dir, "best_model.keras"),
                monitor="val_qw_kappa",
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
        ]
        return callbacks

    def _compile_model(
        self,
        model: tf.keras.Model,
        learning_rate: float,
        class_weights: Dict[int, float],
    ) -> tf.keras.Model:
        """Compile le modèle avec l'optimiseur et les métriques."""
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=self.config.get("weight_decay", 1e-4),
        )

        loss_fn = WeightedFocalLoss(
            gamma=self.config.get("focal_gamma", 2.0),
            class_weights=class_weights,
            label_smoothing=self.config.get("label_smoothing", 0.05),
        )

        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(
                name="auc", multi_label=True, num_labels=5,
            ),
            QuadraticKappaMetric(num_classes=5, name="qw_kappa"),
        ]

        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
        return model

    def run(
        self,
        experiment_name: str = "retinai",
        run_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Exécute le pipeline d'entraînement complet.
        
        Args:
            experiment_name: Nom de l'expérience MLflow.
            run_name: Nom du run (optionnel).
            
        Returns:
            Dictionnaire des métriques finales sur le set de validation.
        """
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):
            # ---- Log des paramètres ----
            mlflow.log_params({
                "backbone": self.config.get("backbone", "efficientnetv2-s"),
                "input_size": self.config.get("input_size", 512),
                "batch_size": self.config.get("batch_size", 16),
                "warmup_epochs": self.config.get("warmup_epochs", 5),
                "finetune_epochs": self.config.get("finetune_epochs", 30),
                "warmup_lr": self.config.get("warmup_lr", 1e-3),
                "finetune_lr": self.config.get("finetune_lr", 1e-4),
                "focal_gamma": self.config.get("focal_gamma", 2.0),
                "dropout_rate": self.config.get("dropout_rate", 0.4),
                "mixup_alpha": self.config.get("mixup_alpha", 0.2),
                "label_smoothing": self.config.get("label_smoothing", 0.05),
            })

            # ---- Chargement des données ----
            logger.info("Chargement du dataset...")
            dataset = RetinalDataset(
                data_dir=self.config["data_dir"],
                labels_csv=self.config["labels_csv"],
                target_size=self.config.get("input_size", 512),
                batch_size=self.config.get("batch_size", 16),
                mixup_alpha=self.config.get("mixup_alpha", 0.2),
            )

            splits = dataset.get_splits()
            train_ds = dataset.build_dataset(splits["train"], shuffle=True, repeat=True)
            val_ds = dataset.build_dataset(splits["val"], shuffle=False, repeat=False)

            steps_per_epoch = len(splits["train"]) // self.config.get("batch_size", 16)

            # Log de la distribution des classes
            for cls, count in dataset.class_distribution.items():
                mlflow.log_metric(f"class_{cls}_count", int(count))
            mlflow.log_params({
                f"class_weight_{k}": round(v, 3)
                for k, v in dataset.class_weights.items()
            })

            # ---- Construction du modèle ----
            logger.info("Construction du modèle...")
            classifier = RetinalClassifier(
                input_size=self.config.get("input_size", 512),
                num_classes=5,
                backbone=self.config.get("backbone", "efficientnetv2-s"),
                dropout_rate=self.config.get("dropout_rate", 0.4),
            )

            output_dir = self.config.get("output_dir", "models/training")
            os.makedirs(output_dir, exist_ok=True)
            callbacks = self._build_callbacks(output_dir)

            # ===== PHASE 1 : Warmup (backbone gelé) =====
            logger.info("=" * 60)
            logger.info("PHASE 1 — Warmup (backbone gelé)")
            logger.info("=" * 60)

            model = classifier.build_model(freeze_backbone=True)
            model = self._compile_model(
                model,
                learning_rate=self.config.get("warmup_lr", 1e-3),
                class_weights=dataset.class_weights,
            )

            warmup_epochs = self.config.get("warmup_epochs", 5)
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=warmup_epochs,
                steps_per_epoch=steps_per_epoch,
                callbacks=callbacks,
                verbose=1,
            )
            mlflow.log_metric("warmup_completed", 1)

            # ===== PHASE 2 : Fine-tuning (backbone partiellement dégelé) =====
            logger.info("=" * 60)
            logger.info("PHASE 2 — Fine-tuning (backbone partiellement dégelé)")
            logger.info("=" * 60)

            model = classifier.unfreeze_backbone(model)
            model = self._compile_model(
                model,
                learning_rate=self.config.get("finetune_lr", 1e-4),
                class_weights=dataset.class_weights,
            )

            finetune_epochs = self.config.get("finetune_epochs", 30)
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=warmup_epochs + finetune_epochs,
                initial_epoch=warmup_epochs,
                steps_per_epoch=steps_per_epoch,
                callbacks=callbacks,
                verbose=1,
            )

            # ---- Sauvegarde et log des artefacts ----
            final_model_path = os.path.join(output_dir, "final_model")
            model.save(final_model_path)
            mlflow.log_artifacts(output_dir, artifact_path="model")

            # Métriques finales
            final_metrics = {
                k: float(v[-1]) if isinstance(v, list) else float(v)
                for k, v in history.history.items()
                if k.startswith("val_")
            }
            logger.info("Métriques finales : %s", final_metrics)

            return final_metrics


def load_config(config_path: str) -> dict:
    """Charge la configuration depuis un fichier YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entraînement RetinAI")
    parser.add_argument("--config", required=True, help="Fichier de configuration YAML")
    parser.add_argument("--experiment-name", default="retinai")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config = load_config(args.config)
    pipeline = TrainingPipeline(config)
    metrics = pipeline.run(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )
    print(f"\nRésultats finaux : {metrics}")
