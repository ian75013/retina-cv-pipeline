"""
Architecture du modèle : EfficientNetV2 + module d'attention spatiale.

Le modèle combine un backbone pré-entraîné (EfficientNetV2-S) avec un
module d'attention spatiale custom permettant de focaliser l'analyse sur
les régions diagnostiquement pertinentes (micro-anévrismes, exsudats,
hémorragies, néo-vascularisation).
"""

import logging
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers, Model

logger = logging.getLogger(__name__)


class SpatialAttentionModule(layers.Layer):
    """
    Module d'attention spatiale pour la focalisation sur les lésions.
    
    Génère une carte d'attention 2D à partir des feature maps, permettant
    au réseau de pondérer spatialement les régions d'intérêt.
    
    Architecture :
        features → Conv1x1(reduce) → ReLU → Conv1x1(1) → Sigmoid → attention_map
        output = features * attention_map
    
    Référence :
        Inspiré de CBAM (Woo et al., 2018) adapté au contexte médical.
    """

    def __init__(self, reduction_ratio: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        reduced = max(channels // self.reduction_ratio, 8)

        self.channel_avg = layers.GlobalAveragePooling2D(keepdims=True)
        self.channel_max = layers.GlobalMaxPooling2D(keepdims=True)

        # Channel attention (SE-like)
        self.channel_fc1 = layers.Dense(reduced, activation="relu")
        self.channel_fc2 = layers.Dense(channels, activation=None)

        # Spatial attention
        self.spatial_conv = layers.Conv2D(
            1, kernel_size=7, padding="same", activation="sigmoid"
        )
        self.concat = layers.Concatenate(axis=-1)

        super().build(input_shape)

    def call(self, inputs, training=None):
        # ---- Channel Attention ----
        avg_pool = self.channel_avg(inputs)
        max_pool = self.channel_max(inputs)

        avg_feat = self.channel_fc2(self.channel_fc1(avg_pool))
        max_feat = self.channel_fc2(self.channel_fc1(max_pool))

        channel_attention = tf.nn.sigmoid(avg_feat + max_feat)
        refined = inputs * channel_attention

        # ---- Spatial Attention ----
        avg_spatial = tf.reduce_mean(refined, axis=-1, keepdims=True)
        max_spatial = tf.reduce_max(refined, axis=-1, keepdims=True)
        spatial_concat = self.concat([avg_spatial, max_spatial])
        spatial_attention = self.spatial_conv(spatial_concat)

        output = refined * spatial_attention

        return output

    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config


class RetinalClassifier:
    """
    Classifieur de rétinopathie diabétique basé sur EfficientNetV2.
    
    Architecture complète :
        Input → EfficientNetV2-S (backbone gelé/fine-tuné) → 
        SpatialAttention → GlobalAvgPool → Dropout → Dense(256) → 
        Dropout → Dense(5, softmax)
    
    Stratégie de fine-tuning :
        1. Phase 1 : backbone gelé, entraînement du head uniquement (warmup)
        2. Phase 2 : dégel progressif des derniers blocs du backbone
    
    Exemple :
        >>> classifier = RetinalClassifier(input_size=512, num_classes=5)
        >>> model = classifier.build_model()
        >>> model.summary()
    """

    BACKBONE_CONFIGS = {
        "efficientnetv2-s": {
            "class": tf.keras.applications.EfficientNetV2S,
            "preprocess": tf.keras.applications.efficientnet_v2.preprocess_input,
            "unfreeze_from": "block6a_expand_conv",
        },
        "efficientnetv2-m": {
            "class": tf.keras.applications.EfficientNetV2M,
            "preprocess": tf.keras.applications.efficientnet_v2.preprocess_input,
            "unfreeze_from": "block6a_expand_conv",
        },
    }

    def __init__(
        self,
        input_size: int = 512,
        num_classes: int = 5,
        backbone: str = "efficientnetv2-s",
        dropout_rate: float = 0.4,
        attention_reduction: int = 16,
    ):
        self.input_size = input_size
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.dropout_rate = dropout_rate
        self.attention_reduction = attention_reduction

        if backbone not in self.BACKBONE_CONFIGS:
            raise ValueError(
                f"Backbone inconnu : {backbone}. "
                f"Choix : {list(self.BACKBONE_CONFIGS.keys())}"
            )

    def build_model(self, freeze_backbone: bool = True) -> Model:
        """
        Construit le modèle complet.
        
        Args:
            freeze_backbone: Si True, gèle le backbone (phase 1 warmup).
            
        Returns:
            Modèle Keras compilable.
        """
        config = self.BACKBONE_CONFIGS[self.backbone_name]
        input_shape = (self.input_size, self.input_size, 3)

        # --- Input et preprocessing ---
        inputs = layers.Input(shape=input_shape, name="input_image")
        preprocessed = layers.Lambda(
            config["preprocess"], name="backbone_preprocess"
        )(inputs)

        # --- Backbone ---
        backbone = config["class"](
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
        backbone.trainable = not freeze_backbone

        features = backbone(preprocessed, training=not freeze_backbone)

        # --- Attention spatiale ---
        attended = SpatialAttentionModule(
            reduction_ratio=self.attention_reduction,
            name="spatial_attention",
        )(features)

        # --- Classification head ---
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(attended)
        x = layers.BatchNormalization(name="head_bn")(x)
        x = layers.Dropout(self.dropout_rate, name="dropout_1")(x)
        x = layers.Dense(256, activation="relu", name="fc_256")(x)
        x = layers.Dropout(self.dropout_rate * 0.5, name="dropout_2")(x)

        outputs = layers.Dense(
            self.num_classes, activation="softmax", name="classification"
        )(x)

        model = Model(inputs=inputs, outputs=outputs, name="retinal_classifier")

        logger.info(
            "Modèle construit — backbone=%s, params=%s, trainable=%s",
            self.backbone_name,
            f"{model.count_params():,}",
            f"{sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}",
        )

        return model

    def unfreeze_backbone(
        self,
        model: Model,
        from_layer: Optional[str] = None,
    ) -> Model:
        """
        Dégèle le backbone à partir d'une couche donnée (fine-tuning phase 2).
        
        Args:
            model: Modèle avec backbone gelé.
            from_layer: Nom de la couche à partir de laquelle dégeler.
                       Si None, utilise la valeur par défaut du backbone.
            
        Returns:
            Modèle avec les couches supérieures du backbone dégelées.
        """
        config = self.BACKBONE_CONFIGS[self.backbone_name]
        unfreeze_from = from_layer or config["unfreeze_from"]

        backbone = model.layers[2]  # Index du backbone dans le modèle
        backbone.trainable = True

        found = False
        frozen_count = 0
        for layer in backbone.layers:
            if layer.name == unfreeze_from:
                found = True
            if not found:
                layer.trainable = False
                frozen_count += 1
            else:
                layer.trainable = True

        trainable_count = sum(
            tf.keras.backend.count_params(w) for w in model.trainable_weights
        )
        logger.info(
            "Fine-tuning activé — %d couches gelées, %s params entraînables",
            frozen_count,
            f"{trainable_count:,}",
        )

        return model

    @staticmethod
    def get_gradcam_model(model: Model, target_layer: str = "top_conv") -> Model:
        """
        Crée un sous-modèle pour l'extraction de Grad-CAM.
        
        Permet de visualiser les zones d'attention du modèle, essentiel
        pour la confiance clinique et l'explicabilité.
        
        Args:
            model: Modèle entraîné.
            target_layer: Couche cible pour l'extraction des gradients.
            
        Returns:
            Modèle produisant (feature_maps, predictions).
        """
        backbone = model.layers[2]
        try:
            target = backbone.get_layer(target_layer)
        except ValueError:
            # Fallback : dernière couche convolutive
            conv_layers = [l for l in backbone.layers if "conv" in l.name]
            target = conv_layers[-1]
            logger.info("Couche Grad-CAM fallback : %s", target.name)

        gradcam_model = Model(
            inputs=model.inputs,
            outputs=[target.output, model.output],
        )
        return gradcam_model
