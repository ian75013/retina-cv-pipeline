"""
Logging structuré pour le pipeline RetinAI.

Configure un système de logging centralisé avec :
- Sortie console colorée (niveaux INFO+)
- Sortie fichier JSON structuré (niveaux DEBUG+)
- Contexte automatique (module, fonction, numéro de ligne)
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Formateur JSON pour les logs structurés (ELK, Datadog, etc.)."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Ajout de champs extra personnalisés
        for key in ("experiment", "run_id", "epoch", "metric"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        return json.dumps(log_entry, ensure_ascii=False)


class ColorFormatter(logging.Formatter):
    """Formateur console avec couleurs ANSI."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Vert
        "WARNING": "\033[33m",   # Jaune
        "ERROR": "\033[31m",     # Rouge
        "CRITICAL": "\033[1;31m", # Rouge gras
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname:<8}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    json_logs: bool = True,
) -> None:
    """
    Configure le système de logging global.

    Args:
        level: Niveau minimum de log (DEBUG, INFO, WARNING, ERROR).
        log_dir: Répertoire pour les fichiers de log (None = pas de fichier).
        json_logs: Si True, les fichiers de log sont au format JSON.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Suppression des handlers existants
    root_logger.handlers.clear()

    # Handler console (coloré)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = "%(asctime)s │ %(levelname)s │ %(name)-25s │ %(message)s"
    console_handler.setFormatter(ColorFormatter(console_format, datefmt="%H:%M:%S"))
    root_logger.addHandler(console_handler)

    # Handler fichier (JSON ou texte)
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension = "jsonl" if json_logs else "log"
        file_path = log_path / f"retinai_{timestamp}.{extension}"

        file_handler = logging.FileHandler(str(file_path), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)

        if json_logs:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
            file_handler.setFormatter(logging.Formatter(file_format))

        root_logger.addHandler(file_handler)
        logging.info("Logs fichier activés : %s", file_path)

    # Réduction du bruit des librairies tierces
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("absl").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.INFO)
