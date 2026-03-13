"""
Fonctions de perte spécialisées pour la classification de rétinopathie.

Implémente la Focal Loss pondérée, adaptée aux datasets fortement
déséquilibrés typiques de l'imagerie médicale (prévalence grade 0 > 70%).
"""

import tensorflow as tf
from tensorflow.keras import backend as K


class WeightedFocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss pondérée par classe.
    
    Combine deux mécanismes de gestion du déséquilibre :
    
    1. **Focal modulation** (Lin et al., 2017) :
       Réduit la contribution des exemples bien classés (easy negatives)
       via un facteur (1 - p_t)^gamma, forçant le modèle à se concentrer
       sur les cas difficiles.
       
    2. **Pondération par classe** :
       Applique des poids inversement proportionnels à la fréquence
       de chaque classe, compensant le sur-échantillonnage naturel
       du grade 0 (No DR).
    
    Formulation :
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma: Paramètre de focalisation (défaut=2.0).
               gamma=0 → Cross-Entropy standard.
               gamma>0 → Réduit la contribution des exemples faciles.
        class_weights: Dict {classe: poids} ou None.
        label_smoothing: Lissage des labels (régularisation).
    
    Exemple :
        >>> weights = {0: 0.5, 1: 2.0, 2: 1.5, 3: 3.0, 4: 4.0}
        >>> loss_fn = WeightedFocalLoss(gamma=2.0, class_weights=weights)
        >>> model.compile(optimizer="adam", loss=loss_fn)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: dict = None,
        label_smoothing: float = 0.05,
        name: str = "weighted_focal_loss",
    ):
        super().__init__(name=name)
        self.gamma = gamma
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calcul de la Focal Loss.
        
        Args:
            y_true: Labels one-hot (batch_size, num_classes).
            y_pred: Probabilités prédites (batch_size, num_classes).
            
        Returns:
            Perte scalaire moyennée sur le batch.
        """
        # Label smoothing
        num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true_smooth = y_true * (1.0 - self.label_smoothing) + (
            self.label_smoothing / num_classes
        )

        # Clipping pour stabilité numérique
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())

        # Cross-entropy par classe
        ce = -y_true_smooth * tf.math.log(y_pred)

        # Facteur focal : (1 - p_t)^gamma
        p_t = tf.reduce_sum(y_true_smooth * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)

        # Application du facteur focal
        focal_ce = focal_weight * ce

        # Pondération par classe
        if self.class_weights is not None:
            weights = tf.constant(
                [self.class_weights.get(i, 1.0) for i in range(int(num_classes))],
                dtype=tf.float32,
            )
            # Broadcast sur le batch
            weights = tf.reshape(weights, [1, -1])
            focal_ce = focal_ce * weights

        # Moyenne sur les classes puis sur le batch
        loss = tf.reduce_sum(focal_ce, axis=-1)
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "class_weights": self.class_weights,
            "label_smoothing": self.label_smoothing,
        })
        return config


class QuadraticWeightedKappaLoss(tf.keras.losses.Loss):
    """
    Loss basée sur le Quadratic Weighted Kappa (QWK).
    
    Le QWK est la métrique standard d'évaluation pour la classification
    ordinale de la rétinopathie diabétique. Cette loss optimise
    directement cette métrique via une approximation différentiable.
    
    Particulièrement adapté car les erreurs de classification entre
    grades proches (0→1) sont moins pénalisées que les erreurs
    entre grades éloignés (0→4).
    """

    def __init__(self, num_classes: int = 5, name: str = "qwk_loss"):
        super().__init__(name=name)
        self.num_classes = num_classes

        # Matrice de pénalité quadratique
        weights = tf.zeros([num_classes, num_classes])
        indices = []
        values = []
        for i in range(num_classes):
            for j in range(num_classes):
                indices.append([i, j])
                values.append(float((i - j) ** 2) / float((num_classes - 1) ** 2))

        self.weight_matrix = tf.constant(
            [[float((i - j) ** 2) / float((num_classes - 1) ** 2)
              for j in range(num_classes)]
             for i in range(num_classes)],
            dtype=tf.float32,
        )

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calcul de la loss QWK différentiable."""
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())

        # Matrice de confusion soft (différentiable)
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        confusion = tf.matmul(tf.transpose(y_true), y_pred) / batch_size

        # Distributions marginales
        hist_true = tf.reduce_sum(y_true, axis=0) / batch_size
        hist_pred = tf.reduce_sum(y_pred, axis=0) / batch_size

        # Matrice attendue sous indépendance
        expected = tf.tensordot(hist_true, hist_pred, axes=0)

        # QWK = 1 - (sum(W * O) / sum(W * E))
        numerator = tf.reduce_sum(self.weight_matrix * confusion)
        denominator = tf.reduce_sum(self.weight_matrix * expected) + K.epsilon()

        kappa = 1.0 - numerator / denominator

        # On retourne 1 - kappa comme loss (à minimiser)
        return 1.0 - kappa

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config
