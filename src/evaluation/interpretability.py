"""
Module d'interprétabilité et de calibration de confiance.

Complète l'évaluation avec :
- Calibration de confiance (reliability diagram, ECE)
- Analyse des erreurs par sous-groupes
- Seuils de décision optimaux pour le triage clinique
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

logger = logging.getLogger(__name__)

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, float]:
    """
    Calcule l'Expected Calibration Error (ECE).

    L'ECE mesure l'écart entre la confiance du modèle et sa précision réelle.
    Un modèle bien calibré devrait avoir ECE ≈ 0 (quand il dit 80% de confiance,
    il a raison 80% du temps).

    Essentiel en contexte médical pour que les cliniciens puissent se fier
    aux scores de confiance pour le triage.

    Args:
        y_true: Labels réels (N,) entiers.
        y_pred_proba: Probabilités prédites (N, num_classes).
        n_bins: Nombre de bins pour le calcul.

    Returns:
        Dict avec ECE, MCE (Maximum Calibration Error) et données du diagramme.
    """
    y_pred = np.argmax(y_pred_proba, axis=1)
    confidences = np.max(y_pred_proba, axis=1)
    accuracies = (y_pred == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_data = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            gap = abs(avg_accuracy - avg_confidence)

            ece += gap * prop_in_bin
            mce = max(mce, gap)

            bin_data.append({
                "bin_center": (bin_boundaries[i] + bin_boundaries[i + 1]) / 2,
                "avg_confidence": float(avg_confidence),
                "avg_accuracy": float(avg_accuracy),
                "count": int(in_bin.sum()),
                "gap": float(gap),
            })

    return {
        "ece": float(ece),
        "mce": float(mce),
        "n_bins": n_bins,
        "bins": bin_data,
    }


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    min_sensitivity: float = 0.90,
) -> Dict[str, Dict[str, float]]:
    """
    Détermine les seuils de décision optimaux pour le triage clinique.

    Pour chaque niveau de sévérité, trouve le seuil qui maximise la spécificité
    tout en maintenant une sensibilité minimale (défaut : 90%).

    Ceci permet de définir des règles de triage automatique :
      - Si P(grade >= 2) > seuil → référer à l'ophtalmologue
      - Sinon → contrôle de routine

    Args:
        y_true: Labels réels (N,).
        y_pred_proba: Probabilités prédites (N, num_classes).
        min_sensitivity: Sensibilité minimale requise.

    Returns:
        Dict avec les seuils optimaux par niveau de binarisation.
    """
    results = {}

    # Seuil "referable" : grade >= 2 vs < 2
    binary_true = (y_true >= 2).astype(int)
    referable_prob = y_pred_proba[:, 2:].sum(axis=1)

    precision, recall, thresholds = precision_recall_curve(binary_true, referable_prob)

    # Trouver le seuil qui maintient sensibilité >= min_sensitivity
    valid = recall[:-1] >= min_sensitivity
    if valid.any():
        # Parmi les seuils valides, maximiser la précision
        best_idx = np.argmax(precision[:-1][valid])
        actual_idx = np.where(valid)[0][best_idx]
        optimal_threshold = float(thresholds[actual_idx])
        optimal_precision = float(precision[actual_idx])
        optimal_recall = float(recall[actual_idx])
    else:
        # Fallback : seuil avec la meilleure sensibilité
        optimal_threshold = float(thresholds[0])
        optimal_precision = float(precision[0])
        optimal_recall = float(recall[0])

    results["referable_vs_non_referable"] = {
        "threshold": optimal_threshold,
        "sensitivity": optimal_recall,
        "precision": optimal_precision,
        "description": "Grade >= 2 (modéré+) détecté comme référable",
    }

    # Seuil "sight-threatening" : grade >= 3 vs < 3
    binary_severe = (y_true >= 3).astype(int)
    severe_prob = y_pred_proba[:, 3:].sum(axis=1)

    if binary_severe.sum() > 0:
        prec_s, rec_s, thresh_s = precision_recall_curve(binary_severe, severe_prob)
        valid_s = rec_s[:-1] >= min_sensitivity

        if valid_s.any():
            best_s = np.argmax(prec_s[:-1][valid_s])
            idx_s = np.where(valid_s)[0][best_s]
            results["sight_threatening"] = {
                "threshold": float(thresh_s[idx_s]),
                "sensitivity": float(rec_s[idx_s]),
                "precision": float(prec_s[idx_s]),
                "description": "Grade >= 3 (sévère+) détecté comme menaçant la vue",
            }

    return results


def analyze_error_patterns(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    confidences: np.ndarray = None,
) -> Dict[str, any]:
    """
    Analyse les patterns d'erreurs du modèle.

    Identifie :
    - Les confusions les plus fréquentes entre classes
    - Les cas à haute confiance erronés (les plus dangereux cliniquement)
    - La distribution des erreurs par grade de sévérité

    Args:
        y_true: Labels réels (N,).
        y_pred_proba: Probabilités prédites (N, num_classes).
        confidences: Confiances associées (si None, calculées depuis y_pred_proba).

    Returns:
        Dict structuré avec l'analyse des erreurs.
    """
    y_pred = np.argmax(y_pred_proba, axis=1)
    if confidences is None:
        confidences = np.max(y_pred_proba, axis=1)

    errors = y_pred != y_true
    n_errors = errors.sum()
    error_rate = n_errors / len(y_true)

    # Confusions les plus fréquentes
    confusion_pairs = {}
    for true, pred in zip(y_true[errors], y_pred[errors]):
        pair = f"{CLASS_NAMES[true]} → {CLASS_NAMES[pred]}"
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

    # Tri par fréquence décroissante
    sorted_confusions = sorted(
        confusion_pairs.items(), key=lambda x: x[1], reverse=True
    )

    # Erreurs à haute confiance (> 0.8) — les plus dangereuses
    high_conf_errors = errors & (confidences > 0.8)
    n_high_conf_errors = high_conf_errors.sum()

    high_conf_details = []
    for i in np.where(high_conf_errors)[0]:
        high_conf_details.append({
            "index": int(i),
            "true_label": CLASS_NAMES[y_true[i]],
            "predicted": CLASS_NAMES[y_pred[i]],
            "confidence": float(confidences[i]),
            "severity_gap": abs(int(y_true[i]) - int(y_pred[i])),
        })

    # Erreurs cliniquement critiques (sous-estimation de >= 2 grades)
    severity_gap = np.abs(y_true.astype(int) - y_pred.astype(int))
    critical_underestimations = errors & (y_pred < y_true) & (severity_gap >= 2)

    return {
        "total_errors": int(n_errors),
        "error_rate": float(error_rate),
        "top_confusions": sorted_confusions[:10],
        "high_confidence_errors": {
            "count": int(n_high_conf_errors),
            "rate": float(n_high_conf_errors / max(n_errors, 1)),
            "details": sorted(high_conf_details, key=lambda x: -x["confidence"])[:20],
        },
        "critical_underestimations": {
            "count": int(critical_underestimations.sum()),
            "description": "Cas où le modèle sous-estime la sévérité de >= 2 grades",
        },
        "mean_severity_gap_on_errors": float(severity_gap[errors].mean()) if n_errors > 0 else 0,
    }


def generate_clinical_summary(
    metrics: Dict,
    calibration: Dict,
    thresholds: Dict,
    error_analysis: Dict,
) -> str:
    """
    Génère un résumé clinique lisible des performances du modèle.

    Ce résumé est destiné aux cliniciens et décideurs, pas aux data scientists.
    Il traduit les métriques techniques en implications cliniques concrètes.
    """
    lines = [
        "=" * 70,
        "RÉSUMÉ CLINIQUE — Performances du modèle RetinAI",
        "=" * 70,
        "",
        "DÉTECTION DES CAS RÉFÉRABLES (grade ≥ 2)",
        f"  Sensibilité : {metrics['global']['sensitivity_referable']:.1%}",
        f"  Spécificité : {metrics['global']['specificity_referable']:.1%}",
        "",
        f"  → Sur 1000 patients avec rétinopathie modérée+, le modèle en détecte "
        f"~{int(metrics['global']['sensitivity_referable'] * 1000)}.",
        f"  → Sur 1000 patients sains, ~{int((1 - metrics['global']['specificity_referable']) * 1000)} "
        f"seraient référés inutilement.",
        "",
        "FIABILITÉ DES SCORES DE CONFIANCE",
        f"  ECE (Expected Calibration Error) : {calibration['ece']:.3f}",
        f"  → {'Bonne' if calibration['ece'] < 0.05 else 'Calibration à améliorer' if calibration['ece'] < 0.1 else 'Mauvaise'} calibration",
        "",
        "SEUILS DE TRIAGE RECOMMANDÉS",
    ]

    for name, data in thresholds.items():
        lines.append(f"  {data['description']}:")
        lines.append(f"    Seuil : {data['threshold']:.3f}")
        lines.append(f"    Sensibilité : {data['sensitivity']:.1%}")
        lines.append(f"    Précision : {data['precision']:.1%}")
        lines.append("")

    lines.extend([
        "ANALYSE DES ERREURS",
        f"  Taux d'erreur global : {error_analysis['error_rate']:.1%}",
        f"  Erreurs à haute confiance (>80%) : {error_analysis['high_confidence_errors']['count']}",
        f"  Sous-estimations critiques (≥2 grades) : {error_analysis['critical_underestimations']['count']}",
        "",
        "  Confusions les plus fréquentes :",
    ])

    for pair, count in error_analysis["top_confusions"][:5]:
        lines.append(f"    {pair} : {count} cas")

    lines.extend([
        "",
        "=" * 70,
        f"Kappa quadratique pondéré : {metrics['global']['quadratic_weighted_kappa']:.3f}",
        f"AUC macro : {metrics['global']['auc_macro']:.3f}",
        "=" * 70,
    ])

    return "\n".join(lines)
