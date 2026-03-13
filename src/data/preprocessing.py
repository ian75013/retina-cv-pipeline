"""
Pipeline de prétraitement d'images rétiniennes.

Implémente les techniques standard de preprocessing pour l'imagerie de fond d'œil :
- Extraction et crop circulaire de la région d'intérêt (ROI)
- Normalisation Ben-Graham (soustraction du flou gaussien local)
- CLAHE adaptatif (Contrast Limited Adaptive Histogram Equalization)
- Redimensionnement et padding intelligent
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration du pipeline de prétraitement."""
    target_size: int = 512
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    apply_ben_graham: bool = True
    ben_graham_sigma: int = 10
    crop_circle: bool = True
    circle_margin: float = 0.05
    normalize: bool = True


class RetinalImagePreprocessor:
    """
    Préprocesseur spécialisé pour les images de fond d'œil.
    
    Pipeline de traitement :
        1. Détection et extraction de la ROI circulaire
        2. Normalisation Ben-Graham pour l'illumination
        3. CLAHE pour le rehaussement de contraste
        4. Resize et normalisation finale
    
    Exemple d'utilisation :
        >>> config = PreprocessingConfig(target_size=512, apply_clahe=True)
        >>> preprocessor = RetinalImagePreprocessor(config)
        >>> processed = preprocessor.process_image("fundus_001.png")
        >>> print(processed.shape)  # (512, 512, 3)
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        if self.config.apply_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_grid_size,
            )
        logger.info(
            "Preprocessor initialisé — target_size=%d, CLAHE=%s, Ben-Graham=%s",
            self.config.target_size,
            self.config.apply_clahe,
            self.config.apply_ben_graham,
        )

    # ------------------------------------------------------------------ #
    #  Étape 1 : Extraction de la ROI circulaire                         #
    # ------------------------------------------------------------------ #

    def extract_circular_roi(self, image: np.ndarray) -> np.ndarray:
        """
        Détecte le disque rétinien et crop l'image autour de celui-ci.
        
        Les images de fond d'œil contiennent typiquement un disque circulaire
        sur fond noir. Cette méthode détecte le cercle via seuillage adaptatif
        puis crop avec une marge configurable.
        
        Args:
            image: Image BGR (H, W, 3).
            
        Returns:
            Image croppée autour du disque rétinien.
        """
        if not self.config.crop_circle:
            return image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Seuillage pour séparer le fond noir du disque rétinien
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        # Recherche du contour principal (le disque)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            logger.warning("Aucun contour détecté — retour de l'image originale")
            return image

        # Plus grand contour = disque rétinien
        largest = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(largest)
        cx, cy, radius = int(cx), int(cy), int(radius)

        # Ajout d'une marge
        margin = int(radius * self.config.circle_margin)
        r = radius + margin

        # Crop avec gestion des bords
        h, w = image.shape[:2]
        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(w, cx + r)
        y2 = min(h, cy + r)

        cropped = image[y1:y2, x1:x2]

        # Application du masque circulaire
        mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
        center = (cropped.shape[1] // 2, cropped.shape[0] // 2)
        cv2.circle(mask, center, min(center), 255, -1)
        result = cv2.bitwise_and(cropped, cropped, mask=mask)

        return result

    # ------------------------------------------------------------------ #
    #  Étape 2 : Normalisation Ben-Graham                                #
    # ------------------------------------------------------------------ #

    def apply_ben_graham_normalization(self, image: np.ndarray) -> np.ndarray:
        """
        Normalisation Ben-Graham pour uniformiser l'illumination.
        
        Technique introduite par Ben Graham (Kaggle Diabetic Retinopathy 2015) :
            result = image - GaussianBlur(image, sigma) + 128
            
        Soustrait la composante basse fréquence (illumination non-uniforme)
        en ne conservant que les détails haute fréquence (lésions, vaisseaux).
        
        Args:
            image: Image BGR (H, W, 3).
            
        Returns:
            Image normalisée.
        """
        if not self.config.apply_ben_graham:
            return image

        sigma = self.config.ben_graham_sigma
        # Le kernel doit être impair et couvrir ~3 sigma
        ksize = sigma * 6 + 1
        if ksize % 2 == 0:
            ksize += 1

        blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)

        # Soustraction avec recentrage à 128
        result = cv2.addWeighted(image, 4, blurred, -4, 128)

        return result

    # ------------------------------------------------------------------ #
    #  Étape 3 : CLAHE (Contrast Limited Adaptive Histogram Eq.)         #
    # ------------------------------------------------------------------ #

    def apply_clahe_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Rehaussement de contraste via CLAHE sur l'espace LAB.
        
        Le CLAHE est appliqué uniquement sur le canal L (luminance) de
        l'espace colorimétrique LAB, préservant ainsi les informations
        chromatiques tout en améliorant le contraste local.
        
        Args:
            image: Image BGR (H, W, 3).
            
        Returns:
            Image avec contraste rehaussé.
        """
        if not self.config.apply_clahe:
            return image

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        channels = list(cv2.split(lab))
        channels[0] = self._clahe.apply(channels[0])
        lab_enhanced = cv2.merge(channels)
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return result

    # ------------------------------------------------------------------ #
    #  Étape 4 : Resize et normalisation                                 #
    # ------------------------------------------------------------------ #

    def resize_and_normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Redimensionnement avec préservation du ratio et normalisation [0, 1].
        
        Utilise INTER_AREA pour la réduction et INTER_CUBIC pour l'agrandissement,
        avec padding noir pour obtenir une image carrée.
        
        Args:
            image: Image BGR (H, W, 3).
            
        Returns:
            Image redimensionnée et normalisée, shape (target_size, target_size, 3).
        """
        target = self.config.target_size
        h, w = image.shape[:2]

        # Calcul du facteur d'échelle (préserve le ratio)
        scale = target / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

        # Padding centré pour obtenir une image carrée
        canvas = np.zeros((target, target, 3), dtype=resized.dtype)
        y_offset = (target - new_h) // 2
        x_offset = (target - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        if self.config.normalize:
            canvas = canvas.astype(np.float32) / 255.0

        return canvas

    # ------------------------------------------------------------------ #
    #  Pipeline complet                                                   #
    # ------------------------------------------------------------------ #

    def process_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Exécute le pipeline complet de prétraitement sur une image.
        
        Args:
            image_path: Chemin vers l'image source.
            
        Returns:
            Image prétraitée (target_size, target_size, 3) ou None si erreur.
        """
        image = cv2.imread(image_path)
        if image is None:
            logger.error("Impossible de lire l'image : %s", image_path)
            return None

        # Pipeline séquentiel
        image = self.extract_circular_roi(image)
        image = self.apply_ben_graham_normalization(image)
        image = self.apply_clahe_enhancement(image)
        image = self.resize_and_normalize(image)

        return image

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        labels_csv: Optional[str] = None,
        n_workers: int = 4,
    ) -> pd.DataFrame:
        """
        Traitement batch d'un répertoire d'images avec parallélisation.
        
        Args:
            input_dir: Répertoire contenant les images sources.
            output_dir: Répertoire de destination.
            labels_csv: Fichier CSV optionnel avec colonnes [image, label].
            n_workers: Nombre de workers pour le traitement parallèle.
            
        Returns:
            DataFrame avec le statut de traitement pour chaque image.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
        image_files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in extensions
        ]

        logger.info(
            "Traitement de %d images (%d workers)", len(image_files), n_workers
        )

        results = []

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for img_file in image_files:
                out_file = output_path / f"{img_file.stem}.npy"
                future = executor.submit(self._process_and_save, str(img_file), str(out_file))
                futures[future] = img_file.name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    success = future.result()
                    results.append({"image": name, "status": "ok" if success else "error"})
                except Exception as e:
                    logger.error("Erreur sur %s : %s", name, e)
                    results.append({"image": name, "status": "error"})

        df = pd.DataFrame(results)
        n_ok = (df["status"] == "ok").sum()
        logger.info("Terminé — %d/%d images traitées avec succès", n_ok, len(df))

        return df

    def _process_and_save(self, input_path: str, output_path: str) -> bool:
        """Traite une image et sauvegarde le résultat en .npy."""
        processed = self.process_image(input_path)
        if processed is None:
            return False
        np.save(output_path, processed)
        return True


# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prétraitement d'images rétiniennes")
    parser.add_argument("--input-dir", required=True, help="Répertoire source")
    parser.add_argument("--output-dir", required=True, help="Répertoire destination")
    parser.add_argument("--target-size", type=int, default=512)
    parser.add_argument("--apply-clahe", action="store_true")
    parser.add_argument("--apply-ben-graham", action="store_true")
    parser.add_argument("--n-workers", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config = PreprocessingConfig(
        target_size=args.target_size,
        apply_clahe=args.apply_clahe,
        apply_ben_graham=args.apply_ben_graham,
    )
    preprocessor = RetinalImagePreprocessor(config)
    report = preprocessor.process_directory(
        args.input_dir, args.output_dir, n_workers=args.n_workers,
    )
    print(report.to_string(index=False))
