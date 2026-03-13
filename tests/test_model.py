"""
Tests unitaires pour l'architecture du modèle et les fonctions de loss.
"""

import numpy as np
import pytest
import tensorflow as tf

from src.models.architecture import RetinalClassifier, SpatialAttentionModule
from src.models.losses import WeightedFocalLoss, QuadraticWeightedKappaLoss


class TestSpatialAttention:
    """Tests du module d'attention spatiale."""

    def test_output_shape(self):
        layer = SpatialAttentionModule(reduction_ratio=16)
        x = tf.random.normal([2, 16, 16, 256])
        output = layer(x)
        assert output.shape == x.shape

    def test_attention_range(self):
        layer = SpatialAttentionModule(reduction_ratio=8)
        x = tf.random.normal([1, 8, 8, 128])
        output = layer(x)
        # La sortie devrait être modulée (pas identique à l'entrée)
        assert not np.allclose(output.numpy(), x.numpy(), atol=1e-3)


class TestRetinalClassifier:
    """Tests de l'architecture du classifieur."""

    @pytest.fixture
    def classifier(self):
        return RetinalClassifier(
            input_size=128,  # Petit pour les tests
            num_classes=5,
            backbone="efficientnetv2-s",
            dropout_rate=0.3,
        )

    def test_model_builds(self, classifier):
        model = classifier.build_model(freeze_backbone=True)
        assert model is not None
        assert model.output_shape == (None, 5)

    def test_model_output_shape(self, classifier):
        model = classifier.build_model(freeze_backbone=True)
        dummy_input = np.random.rand(1, 128, 128, 3).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)
        assert output.shape == (1, 5)

    def test_output_is_probability(self, classifier):
        model = classifier.build_model(freeze_backbone=True)
        dummy_input = np.random.rand(2, 128, 128, 3).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)
        # Somme des probabilités ≈ 1
        np.testing.assert_allclose(output.sum(axis=1), [1.0, 1.0], atol=1e-5)
        # Toutes les probabilités >= 0
        assert (output >= 0).all()

    def test_frozen_backbone(self, classifier):
        model = classifier.build_model(freeze_backbone=True)
        backbone = model.layers[2]
        trainable_count = sum(1 for l in backbone.layers if l.trainable)
        # Backbone gelé → aucune couche entraînable
        assert trainable_count == 0

    def test_unfreeze(self, classifier):
        model = classifier.build_model(freeze_backbone=True)
        model = classifier.unfreeze_backbone(model)
        backbone = model.layers[2]
        trainable_count = sum(1 for l in backbone.layers if l.trainable)
        assert trainable_count > 0


class TestWeightedFocalLoss:
    """Tests de la Focal Loss pondérée."""

    def test_basic_computation(self):
        loss_fn = WeightedFocalLoss(gamma=2.0)
        y_true = tf.constant([[1, 0, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0.9, 0.05, 0.02, 0.02, 0.01],
                               [0.1, 0.1, 0.6, 0.1, 0.1]], dtype=tf.float32)
        loss = loss_fn(y_true, y_pred)
        assert loss.numpy() > 0
        assert np.isfinite(loss.numpy())

    def test_perfect_prediction_low_loss(self):
        loss_fn = WeightedFocalLoss(gamma=2.0)
        y_true = tf.constant([[1, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred_good = tf.constant([[0.95, 0.01, 0.02, 0.01, 0.01]], dtype=tf.float32)
        y_pred_bad = tf.constant([[0.1, 0.3, 0.3, 0.2, 0.1]], dtype=tf.float32)

        loss_good = loss_fn(y_true, y_pred_good)
        loss_bad = loss_fn(y_true, y_pred_bad)
        assert loss_good < loss_bad

    def test_gamma_zero_equals_ce(self):
        """Avec gamma=0, la Focal Loss devrait être proche de la CE."""
        loss_focal = WeightedFocalLoss(gamma=0.0, label_smoothing=0.0)
        loss_ce = tf.keras.losses.CategoricalCrossentropy()

        y_true = tf.constant([[1, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0.7, 0.1, 0.1, 0.05, 0.05]], dtype=tf.float32)

        fl = loss_focal(y_true, y_pred).numpy()
        ce = loss_ce(y_true, y_pred).numpy()
        np.testing.assert_allclose(fl, ce, rtol=0.1)

    def test_with_class_weights(self):
        weights = {0: 0.5, 1: 2.0, 2: 1.5, 3: 3.0, 4: 4.0}
        loss_fn = WeightedFocalLoss(gamma=2.0, class_weights=weights)
        y_true = tf.constant([[0, 0, 0, 0, 1]], dtype=tf.float32)  # Classe rare
        y_pred = tf.constant([[0.3, 0.2, 0.2, 0.2, 0.1]], dtype=tf.float32)
        loss = loss_fn(y_true, y_pred)
        assert loss.numpy() > 0


class TestQWKLoss:
    """Tests de la loss Quadratic Weighted Kappa."""

    def test_perfect_agreement(self):
        loss_fn = QuadraticWeightedKappaLoss(num_classes=5)
        y_true = tf.eye(5)
        y_pred = tf.eye(5)
        loss = loss_fn(y_true, y_pred)
        # Kappa parfait → loss ≈ 0
        assert loss.numpy() < 0.1

    def test_output_range(self):
        loss_fn = QuadraticWeightedKappaLoss(num_classes=5)
        y_true = tf.constant([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0.5, 0.3, 0.1, 0.05, 0.05],
                               [0.1, 0.5, 0.2, 0.1, 0.1]], dtype=tf.float32)
        loss = loss_fn(y_true, y_pred)
        assert 0 <= loss.numpy() <= 2.0  # 1 - kappa, kappa ∈ [-1, 1]
