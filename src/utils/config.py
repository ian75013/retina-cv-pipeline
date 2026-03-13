"""
Gestion centralisée de la configuration.

Charge et valide les fichiers YAML de configuration, avec support
des variables d'environnement et des valeurs par défaut.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# Valeurs par défaut globales
DEFAULTS = {
    "backbone": "efficientnetv2-s",
    "input_size": 512,
    "batch_size": 16,
    "warmup_epochs": 5,
    "finetune_epochs": 30,
    "warmup_lr": 1e-3,
    "finetune_lr": 1e-4,
    "weight_decay": 1e-4,
    "focal_gamma": 2.0,
    "label_smoothing": 0.05,
    "dropout_rate": 0.4,
    "mixup_alpha": 0.2,
    "cutmix_alpha": 1.0,
    "early_stopping_patience": 10,
    "output_dir": "models/training",
}

# Clés obligatoires (doivent être définies dans le YAML)
REQUIRED_KEYS = {"data_dir", "labels_csv"}


def load_config(config_path: str, override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Charge la configuration depuis un fichier YAML.

    Ordre de priorité (du plus bas au plus haut) :
        1. Valeurs par défaut (DEFAULTS)
        2. Fichier YAML
        3. Variables d'environnement (préfixe RETINAI_)
        4. Overrides passés en argument

    Args:
        config_path: Chemin vers le fichier YAML.
        override: Dict de valeurs à forcer (priorité maximale).

    Returns:
        Configuration fusionnée et validée.

    Raises:
        FileNotFoundError: Si le fichier YAML n'existe pas.
        ValueError: Si des clés obligatoires sont manquantes.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {config_path}")

    # 1. Valeurs par défaut
    config = dict(DEFAULTS)

    # 2. Fichier YAML
    with open(path, "r") as f:
        yaml_config = yaml.safe_load(f) or {}
    config.update(yaml_config)

    # 3. Variables d'environnement
    env_overrides = _load_env_overrides()
    config.update(env_overrides)

    # 4. Overrides explicites
    if override:
        config.update(override)

    # Validation
    _validate_config(config)

    logger.info("Configuration chargée depuis %s (%d paramètres)", config_path, len(config))
    return config


def _load_env_overrides() -> Dict[str, Any]:
    """
    Charge les overrides depuis les variables d'environnement.

    Convention : RETINAI_<CLE_MAJUSCULE>
    Exemples :
        RETINAI_BATCH_SIZE=32
        RETINAI_BACKBONE=efficientnetv2-m
    """
    prefix = "RETINAI_"
    overrides = {}

    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):].lower()
            overrides[config_key] = _parse_env_value(value)

    if overrides:
        logger.info("Overrides d'environnement détectés : %s", list(overrides.keys()))

    return overrides


def _parse_env_value(value: str) -> Any:
    """Parse une valeur d'environnement vers le type Python approprié."""
    # Booléens
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False

    # Nombres
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        pass

    return value


def _validate_config(config: Dict[str, Any]) -> None:
    """Valide que la configuration contient toutes les clés obligatoires."""
    missing = REQUIRED_KEYS - set(config.keys())
    if missing:
        raise ValueError(
            f"Clés de configuration manquantes : {missing}. "
            f"Ajoutez-les dans le fichier YAML ou via RETINAI_<CLE>."
        )

    # Validations de plage
    if config.get("batch_size", 1) < 1:
        raise ValueError("batch_size doit être >= 1")
    if config.get("input_size", 1) < 32:
        raise ValueError("input_size doit être >= 32")
    if not 0 <= config.get("dropout_rate", 0) < 1:
        raise ValueError("dropout_rate doit être dans [0, 1)")
    if config.get("focal_gamma", 0) < 0:
        raise ValueError("focal_gamma doit être >= 0")
