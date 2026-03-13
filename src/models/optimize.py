"""
Optimisation du modèle : pruning, quantification et Test-Time Augmentation.

Fournit des outils pour réduire la taille et le temps d'inférence du modèle
tout en maintenant (voire améliorant) les performances de classification.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Test-Time Augmentation (TTA)                                       #
# ------------------------------------------------------------------ #

class TestTimeAugmentation:
    """
    Test-Time Augmentation pour l'inférence robuste.
    
    Applique N transformations à chaque image à l'inférence et agrège
    les prédictions, réduisant la variance et améliorant la robustesse.
    
    Les transformations exploitent l'invariance rotationnelle des images
    de fond d'œil (pas d'orientation anatomique fixe).
    
    Transformations appliquées (8 au total) :
        - Image originale
        - Rotation 90°, 180°, 270°
        - Flip horizontal
        - Flip vertical
        - Flip horizontal + rotation 90°
        - Flip vertical + rotation 90°
    
    Exemple :
        >>> tta = TestTimeAugmentation(model)
        >>> predictions = tta.predict(image_batch)
        >>> # predictions est la moyenne des 8 prédictions
    """

    def __init__(
        self,
        model: tf.keras.Model,
        n_augmentations: int = 8,
        aggregation: str = "mean",
    ):
        self.model = model
        self.n_augmentations = min(n_augmentations, 8)
        self.aggregation = aggregation

    @staticmethod
    def _get_transforms() -> List[callable]:
        """Retourne la liste des transformations TTA."""
        transforms = [
            lambda x: x,                                          # Original
            lambda x: tf.image.rot90(x, k=1),                    # 90°
            lambda x: tf.image.rot90(x, k=2),                    # 180°
            lambda x: tf.image.rot90(x, k=3),                    # 270°
            lambda x: tf.image.flip_left_right(x),               # H-flip
            lambda x: tf.image.flip_up_down(x),                  # V-flip
            lambda x: tf.image.rot90(tf.image.flip_left_right(x), k=1),
            lambda x: tf.image.rot90(tf.image.flip_up_down(x), k=1),
        ]
        return transforms

    def predict(self, images: np.ndarray) -> np.ndarray:
        """
        Prédiction avec TTA.
        
        Args:
            images: Batch d'images (B, H, W, 3).
            
        Returns:
            Prédictions agrégées (B, num_classes).
        """
        transforms = self._get_transforms()[:self.n_augmentations]
        all_predictions = []

        for transform in transforms:
            augmented = transform(images)
            preds = self.model.predict(augmented, verbose=0)
            all_predictions.append(preds)

        stacked = np.stack(all_predictions, axis=0)

        if self.aggregation == "mean":
            return np.mean(stacked, axis=0)
        elif self.aggregation == "geometric_mean":
            log_preds = np.log(np.clip(stacked, 1e-7, 1.0))
            geo_mean = np.exp(np.mean(log_preds, axis=0))
            # Re-normalisation
            return geo_mean / np.sum(geo_mean, axis=-1, keepdims=True)
        else:
            raise ValueError(f"Agrégation inconnue : {self.aggregation}")

    def predict_with_uncertainty(
        self, images: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prédiction avec estimation de l'incertitude.
        
        L'écart-type entre les prédictions TTA fournit une mesure de
        l'incertitude prédictive, utile pour le triage clinique.
        
        Returns:
            Tuple (prédictions moyennes, incertitudes par classe).
        """
        transforms = self._get_transforms()[:self.n_augmentations]
        all_predictions = []

        for transform in transforms:
            augmented = transform(images)
            preds = self.model.predict(augmented, verbose=0)
            all_predictions.append(preds)

        stacked = np.stack(all_predictions, axis=0)
        mean_preds = np.mean(stacked, axis=0)
        uncertainty = np.std(stacked, axis=0)

        return mean_preds, uncertainty


# ------------------------------------------------------------------ #
#  Pruning structurel                                                  #
# ------------------------------------------------------------------ #

class ModelPruner:
    """
    Pruning structurel du modèle pour réduction de taille et latence.
    
    Utilise l'API tensorflow_model_optimization pour élaguer les poids
    de faible magnitude, réduisant la complexité du modèle sans
    dégradation significative des performances.
    
    Stratégie : pruning progressif (polynomial decay) du taux initial
    au taux final sur N steps.
    """

    def __init__(
        self,
        initial_sparsity: float = 0.0,
        final_sparsity: float = 0.5,
        begin_step: int = 0,
        end_step: int = 10000,
    ):
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.begin_step = begin_step
        self.end_step = end_step

    def apply_pruning(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        Applique le pruning sur les couches Dense et Conv2D du modèle.
        
        Note : Nécessite un fine-tuning post-pruning de quelques epochs
        pour récupérer les performances.
        """
        try:
            import tensorflow_model_optimization as tfmot
        except ImportError:
            logger.error(
                "tensorflow-model-optimization non installé. "
                "Installez-le via : pip install tensorflow-model-optimization"
            )
            return model

        pruning_params = {
            "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=self.initial_sparsity,
                final_sparsity=self.final_sparsity,
                begin_step=self.begin_step,
                end_step=self.end_step,
            ),
        }

        def apply_pruning_to_layer(layer):
            """Applique le pruning aux couches éligibles."""
            if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
                return tfmot.sparsity.keras.prune_low_magnitude(
                    layer, **pruning_params
                )
            return layer

        pruned_model = tf.keras.models.clone_model(
            model, clone_function=apply_pruning_to_layer,
        )

        logger.info(
            "Pruning appliqué — sparsité cible : %.1f%%",
            self.final_sparsity * 100,
        )

        return pruned_model

    @staticmethod
    def strip_pruning(model: tf.keras.Model) -> tf.keras.Model:
        """Supprime les wrappers de pruning pour l'export."""
        try:
            import tensorflow_model_optimization as tfmot
            return tfmot.sparsity.keras.strip_pruning(model)
        except ImportError:
            return model


# ------------------------------------------------------------------ #
#  Quantification                                                      #
# ------------------------------------------------------------------ #

class ModelQuantizer:
    """
    Quantification du modèle pour optimisation de l'inférence.
    
    Supporte trois niveaux de quantification :
        - dynamic : quantification dynamique des poids (INT8)
        - float16 : réduction de précision des poids
        - full_int8 : quantification complète INT8 (nécessite un dataset représentatif)
    """

    def __init__(self, quantization_mode: str = "dynamic"):
        self.mode = quantization_mode

    def quantize(
        self,
        model: tf.keras.Model,
        representative_dataset: Optional[callable] = None,
        output_path: Optional[str] = None,
    ) -> bytes:
        """
        Quantifie le modèle via TFLite.
        
        Args:
            model: Modèle Keras entraîné.
            representative_dataset: Générateur d'échantillons pour calibration.
            output_path: Chemin de sauvegarde du modèle TFLite.
            
        Returns:
            Modèle quantifié en bytes (format TFLite).
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        if self.mode == "dynamic":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            logger.info("Quantification dynamique INT8")

        elif self.mode == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            logger.info("Quantification Float16")

        elif self.mode == "full_int8":
            if representative_dataset is None:
                raise ValueError(
                    "Un dataset représentatif est requis pour la quantification INT8 complète"
                )
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            ]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            logger.info("Quantification complète INT8")
        else:
            raise ValueError(f"Mode inconnu : {self.mode}")

        tflite_model = converter.convert()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(tflite_model)

            size_mb = len(tflite_model) / (1024 * 1024)
            logger.info("Modèle quantifié sauvegardé : %s (%.1f MB)", output_path, size_mb)

        return tflite_model

    @staticmethod
    def benchmark_tflite(
        tflite_model: bytes,
        test_input: np.ndarray,
        n_runs: int = 100,
    ) -> Dict[str, float]:
        """
        Benchmark du modèle TFLite (latence, throughput).
        
        Args:
            tflite_model: Modèle TFLite en bytes.
            test_input: Image de test (1, H, W, 3).
            n_runs: Nombre d'itérations pour le benchmark.
            
        Returns:
            Dict avec latence moyenne, P95, P99 et throughput.
        """
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Adaptation du type d'entrée si nécessaire
        input_dtype = input_details[0]["dtype"]
        if input_dtype == np.uint8:
            test_input = (test_input * 255).astype(np.uint8)
        elif input_dtype == np.float32:
            test_input = test_input.astype(np.float32)

        # Warmup
        for _ in range(10):
            interpreter.set_tensor(input_details[0]["index"], test_input)
            interpreter.invoke()

        # Benchmark
        latencies = []
        for _ in range(n_runs):
            start = time.perf_counter()
            interpreter.set_tensor(input_details[0]["index"], test_input)
            interpreter.invoke()
            latencies.append((time.perf_counter() - start) * 1000)

        latencies = np.array(latencies)

        results = {
            "mean_latency_ms": float(np.mean(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "throughput_fps": float(1000 / np.mean(latencies)),
        }

        logger.info(
            "Benchmark TFLite — mean=%.1fms, P95=%.1fms, throughput=%.0f FPS",
            results["mean_latency_ms"],
            results["p95_latency_ms"],
            results["throughput_fps"],
        )

        return results


# ------------------------------------------------------------------ #
#  Export ONNX                                                         #
# ------------------------------------------------------------------ #

def export_to_onnx(
    model: tf.keras.Model,
    output_path: str,
    input_size: int = 512,
    opset_version: int = 13,
) -> None:
    """
    Exporte le modèle au format ONNX pour interopérabilité.
    
    L'export ONNX permet le déploiement sur des runtimes optimisés
    (ONNX Runtime, TensorRT) pour une latence d'inférence minimale.
    """
    try:
        import tf2onnx
    except ImportError:
        logger.error("tf2onnx non installé : pip install tf2onnx")
        return

    spec = (tf.TensorSpec((1, input_size, input_size, 3), tf.float32, name="input"),)

    model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=opset_version,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(model_proto.SerializeToString())

    logger.info("Modèle ONNX exporté : %s", output_path)


# ------------------------------------------------------------------ #
#  Pipeline d'optimisation complet                                     #
# ------------------------------------------------------------------ #

def optimize_pipeline(
    model_path: str,
    output_dir: str,
    pruning_rate: float = 0.3,
    quantize: str = "dynamic",
    export_onnx: bool = True,
) -> Dict[str, str]:
    """
    Pipeline d'optimisation complet : pruning → quantification → export.
    
    Args:
        model_path: Chemin du modèle Keras entraîné.
        output_dir: Répertoire de sortie.
        pruning_rate: Taux de sparsité cible.
        quantize: Mode de quantification.
        export_onnx: Si True, exporte aussi en ONNX.
        
    Returns:
        Dict avec les chemins des modèles optimisés.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    logger.info("Chargement du modèle : %s", model_path)
    model = tf.keras.models.load_model(model_path)

    artifacts = {}

    # 1. Pruning
    if pruning_rate > 0:
        logger.info("Application du pruning (%.0f%%)...", pruning_rate * 100)
        pruner = ModelPruner(final_sparsity=pruning_rate)
        model = pruner.apply_pruning(model)
        model = ModelPruner.strip_pruning(model)
        pruned_path = str(output / "pruned_model")
        model.save(pruned_path)
        artifacts["pruned"] = pruned_path

    # 2. Quantification TFLite
    logger.info("Quantification (%s)...", quantize)
    quantizer = ModelQuantizer(quantization_mode=quantize)
    tflite_path = str(output / f"model_{quantize}.tflite")
    quantizer.quantize(model, output_path=tflite_path)
    artifacts["tflite"] = tflite_path

    # 3. Export ONNX
    if export_onnx:
        logger.info("Export ONNX...")
        onnx_path = str(output / "model.onnx")
        export_to_onnx(model, onnx_path)
        artifacts["onnx"] = onnx_path

    logger.info("Optimisation terminée — artefacts : %s", artifacts)
    return artifacts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimisation du modèle RetinAI")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--pruning-rate", type=float, default=0.3)
    parser.add_argument("--quantize", choices=["dynamic", "float16", "full_int8"], default="dynamic")
    parser.add_argument("--no-onnx", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    optimize_pipeline(
        model_path=args.model_path,
        output_dir=args.output_path,
        pruning_rate=args.pruning_rate,
        quantize=args.quantize,
        export_onnx=not args.no_onnx,
    )
