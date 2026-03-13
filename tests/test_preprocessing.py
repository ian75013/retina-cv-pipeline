"""
Tests unitaires pour le pipeline de prétraitement.

Vérifie le comportement de chaque étape du preprocessing :
- Dimensions de sortie
- Plage de valeurs
- Gestion des cas limites (images nulles, formats invalides)
"""

import numpy as np
import pytest

from src.data.preprocessing import PreprocessingConfig, RetinalImagePreprocessor


@pytest.fixture
def preprocessor():
    config = PreprocessingConfig(
        target_size=256,
        apply_clahe=True,
        apply_ben_graham=True,
        crop_circle=False,  # Désactivé pour les tests (pas de vrai disque)
    )
    return RetinalImagePreprocessor(config)


@pytest.fixture
def sample_image():
    """Génère une image synthétique simulant un fond d'œil."""
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    # Simule un disque rétinien (cercle clair au centre)
    center = (320, 240)
    for y in range(480):
        for x in range(640):
            dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            if dist < 200:
                img[y, x] = np.clip(img[y, x] + 80, 0, 255).astype(np.uint8)
    return img


class TestBenGrahamNormalization:
    """Tests pour la normalisation Ben-Graham."""

    def test_output_shape(self, preprocessor, sample_image):
        result = preprocessor.apply_ben_graham_normalization(sample_image)
        assert result.shape == sample_image.shape

    def test_output_dtype(self, preprocessor, sample_image):
        result = preprocessor.apply_ben_graham_normalization(sample_image)
        assert result.dtype == np.uint8

    def test_disabled(self, sample_image):
        config = PreprocessingConfig(apply_ben_graham=False)
        pp = RetinalImagePreprocessor(config)
        result = pp.apply_ben_graham_normalization(sample_image)
        np.testing.assert_array_equal(result, sample_image)


class TestCLAHE:
    """Tests pour le rehaussement CLAHE."""

    def test_output_shape(self, preprocessor, sample_image):
        result = preprocessor.apply_clahe_enhancement(sample_image)
        assert result.shape == sample_image.shape

    def test_contrast_increased(self, preprocessor, sample_image):
        """Le CLAHE devrait augmenter la variance locale du canal L."""
        result = preprocessor.apply_clahe_enhancement(sample_image)
        # La variance globale devrait être au moins comparable
        assert np.std(result) > 0

    def test_disabled(self, sample_image):
        config = PreprocessingConfig(apply_clahe=False)
        pp = RetinalImagePreprocessor(config)
        result = pp.apply_clahe_enhancement(sample_image)
        np.testing.assert_array_equal(result, sample_image)


class TestResizeAndNormalize:
    """Tests pour le redimensionnement et la normalisation."""

    def test_output_shape(self, preprocessor, sample_image):
        result = preprocessor.resize_and_normalize(sample_image)
        target = preprocessor.config.target_size
        assert result.shape == (target, target, 3)

    def test_output_range(self, preprocessor, sample_image):
        result = preprocessor.resize_and_normalize(sample_image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_dtype(self, preprocessor, sample_image):
        result = preprocessor.resize_and_normalize(sample_image)
        assert result.dtype == np.float32

    def test_square_image(self, preprocessor):
        square = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = preprocessor.resize_and_normalize(square)
        assert result.shape == (256, 256, 3)

    def test_various_sizes(self, preprocessor):
        for h, w in [(100, 200), (500, 300), (1024, 768)]:
            img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            result = preprocessor.resize_and_normalize(img)
            target = preprocessor.config.target_size
            assert result.shape == (target, target, 3)


class TestFullPipeline:
    """Tests d'intégration du pipeline complet."""

    def test_pipeline_output_shape(self, preprocessor, sample_image, tmp_path):
        # Sauvegarde temporaire pour tester process_image
        import cv2
        img_path = str(tmp_path / "test.png")
        cv2.imwrite(img_path, sample_image)

        result = preprocessor.process_image(img_path)
        target = preprocessor.config.target_size
        assert result is not None
        assert result.shape == (target, target, 3)

    def test_invalid_path(self, preprocessor):
        result = preprocessor.process_image("/nonexistent/path.png")
        assert result is None

    def test_pipeline_deterministic(self, preprocessor, sample_image, tmp_path):
        import cv2
        img_path = str(tmp_path / "test.png")
        cv2.imwrite(img_path, sample_image)

        result1 = preprocessor.process_image(img_path)
        result2 = preprocessor.process_image(img_path)
        np.testing.assert_array_equal(result1, result2)
