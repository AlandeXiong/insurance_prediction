"""
Model structure visualization for trained tree-based models.

Generates tree structure plots for LightGBM, XGBoost, and CatBoost.
Optional: feature importance bar chart per model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelStructureArtifacts:
    """Paths of generated model structure outputs."""
    output_dir: Path
    tree_plots: Dict[str, List[Path]] = field(default_factory=dict)  # model_name -> [tree_0.png, ...]
    feature_importance_plot: Optional[Path] = None
    errors: List[str] = field(default_factory=list)


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _plot_lightgbm_trees(
    model: Any,
    model_name: str,
    out_dir: Path,
    max_trees: int,
) -> List[Path]:
    """Plot first max_trees LightGBM trees. Returns list of saved paths."""
    try:
        import lightgbm as lgb
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        logger.warning(f"LightGBM tree plot skipped (missing deps): {e}")
        return []

    paths = []
    try:
        booster = model.booster_
        n_trees = booster.num_trees()
        to_plot = min(max_trees, max(1, n_trees))
        for i in range(to_plot):
            fig, ax = plt.subplots(figsize=(20, 12))
            lgb.plot_tree(booster, tree_index=i, ax=ax, show_info=["split_gain", "internal_count"])
            out_path = out_dir / f"tree_{i}.png"
            plt.savefig(out_path, bbox_inches="tight", dpi=100)
            plt.close()
            paths.append(out_path)
        logger.info(f"LightGBM '{model_name}': saved {len(paths)} tree plots to {out_dir}")
    except Exception as e:
        logger.warning(f"LightGBM tree plot failed for {model_name}: {e}")
    return paths


def _plot_xgboost_trees(
    model: Any,
    model_name: str,
    out_dir: Path,
    max_trees: int,
) -> List[Path]:
    """Plot first max_trees XGBoost trees. Returns list of saved paths."""
    try:
        import xgboost as xgb
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        logger.warning(f"XGBoost tree plot skipped (missing deps): {e}")
        return []

    paths = []
    try:
        booster = model.get_booster()
        # num_trees for multi-class might differ; we use first max_trees
        to_plot = min(max_trees, booster.num_boosted_rounds())
        if to_plot <= 0:
            to_plot = 1
        for i in range(to_plot):
            fig, ax = plt.subplots(figsize=(20, 12))
            xgb.plot_tree(booster, num_trees=i, ax=ax)
            out_path = out_dir / f"tree_{i}.png"
            plt.savefig(out_path, bbox_inches="tight", dpi=100)
            plt.close()
            paths.append(out_path)
        logger.info(f"XGBoost '{model_name}': saved {len(paths)} tree plots to {out_dir}")
    except Exception as e:
        logger.warning(f"XGBoost tree plot failed for {model_name}: {e}")
    return paths


def _plot_catboost_trees(
    model: Any,
    model_name: str,
    out_dir: Path,
    max_trees: int,
) -> List[Path]:
    """Plot first max_trees CatBoost trees (graphviz). Save as PNG via pipe() to avoid text-only output."""
    paths = []
    try:
        n_trees = model.tree_count_
        to_plot = min(max_trees, max(1, n_trees))
        for i in range(to_plot):
            graph = model.plot_tree(tree_idx=i)
            if graph is None:
                continue
            out_path = out_dir / f"tree_{i}.png"
            try:
                # Prefer pipe(): generates PNG in memory, then write to file (no intermediate .dot file)
                png_bytes = graph.pipe(format="png")
                out_path.write_bytes(png_bytes)
                paths.append(out_path)
            except Exception as pipe_err:
                # Fallback 1: render to PNG (needs graphviz in PATH)
                try:
                    stem = out_dir / f"tree_{i}_dot"
                    graph.render(str(stem), format="png", cleanup=True)
                    png_file = stem.with_suffix(".png")
                    if png_file.exists():
                        png_file.rename(out_path)
                        paths.append(out_path)
                    else:
                        raise FileNotFoundError("render did not produce PNG")
                except Exception as render_err:
                    # Fallback 2: save DOT source so user can run: dot -Tpng tree_i.dot -o tree_i.png
                    dot_path = out_dir / f"tree_{i}.dot"
                    with open(dot_path, "w", encoding="utf-8") as f:
                        f.write(getattr(graph, "source", str(graph)))
                    logger.warning(
                        f"CatBoost tree {i}: PNG not generated (install graphviz, e.g. brew install graphviz). "
                        f"DOT source saved to {dot_path}; run: dot -Tpng {dot_path} -o {out_path}"
                    )
        if paths:
            logger.info(f"CatBoost '{model_name}': saved {len(paths)} tree plots to {out_dir}")
    except Exception as e:
        logger.warning(f"CatBoost tree plot failed for {model_name}: {e}")
    return paths


def _plot_ensemble_trees(
    model: Any,
    model_name: str,
    out_dir: Path,
    max_trees: int,
    artifacts: ModelStructureArtifacts,
) -> None:
    """For VotingClassifier: plot first N trees of each base tree model."""
    try:
        from sklearn.ensemble import VotingClassifier
        if not isinstance(model, VotingClassifier):
            return
        for est_name, est_model in model.named_estimators_.items():
            sub_dir = out_dir / est_name
            _safe_mkdir(sub_dir)
            tree_paths = plot_single_model_trees(est_model, est_name, sub_dir, max_trees)
            if tree_paths:
                artifacts.tree_plots[f"{model_name}/{est_name}"] = tree_paths
    except Exception as e:
        logger.warning(f"Ensemble tree plot failed for {model_name}: {e}")
        artifacts.errors.append(f"ensemble {model_name}: {e}")


def plot_single_model_trees(
    model: Any,
    model_name: str,
    out_dir: Path,
    max_trees: int = 3,
) -> List[Path]:
    """
    Plot first max_trees trees for a single tree-based model.
    Returns list of saved file paths.
    """
    _safe_mkdir(out_dir)
    paths: List[Path] = []

    # LightGBM
    try:
        import lightgbm as lgb
        if isinstance(model, lgb.LGBMClassifier):
            paths = _plot_lightgbm_trees(model, model_name, out_dir, max_trees)
            return paths
    except ImportError:
        pass

    # XGBoost
    try:
        import xgboost as xgb
        if isinstance(model, xgb.XGBClassifier):
            paths = _plot_xgboost_trees(model, model_name, out_dir, max_trees)
            return paths
    except ImportError:
        pass

    # CatBoost
    try:
        from catboost import CatBoostClassifier
        if isinstance(model, CatBoostClassifier):
            paths = _plot_catboost_trees(model, model_name, out_dir, max_trees)
            return paths
    except ImportError:
        pass

    # Ensemble (VotingClassifier): handled by caller with _plot_ensemble_trees
    return paths


def run_model_structure_visualization(
    models: Dict[str, Any],
    output_dir: Path,
    max_trees_per_model: int = 3,
    feature_importance: Optional[Dict[str, Dict[str, float]]] = None,
    include_ensemble_trees: bool = True,
) -> ModelStructureArtifacts:
    """
    Generate tree structure plots for all tree-based models.

    Args:
        models: name -> fitted model (LGBMClassifier, XGBClassifier, CatBoostClassifier, VotingClassifier)
        output_dir: base directory for outputs (e.g. reports/model_structure)
        max_trees_per_model: max number of trees to plot per model (default 3)
        feature_importance: optional name -> {feature: importance} for bar chart
        include_ensemble_trees: if True, plot base estimators of VotingClassifier

    Returns:
        ModelStructureArtifacts with paths and any errors.
    """
    artifacts = ModelStructureArtifacts(output_dir=output_dir)
    _safe_mkdir(output_dir)

    try:
        from sklearn.ensemble import VotingClassifier
        has_voting = True
    except ImportError:
        VotingClassifier = None
        has_voting = False

    for name, model in models.items():
        if model is None:
            continue
        model_out = output_dir / name
        _safe_mkdir(model_out)

        if has_voting and isinstance(model, VotingClassifier) and include_ensemble_trees:
            _plot_ensemble_trees(model, name, model_out, max_trees_per_model, artifacts)
            # Also store summary for ensemble (no single tree file, but subdirs recorded)
            continue

        tree_paths = plot_single_model_trees(model, name, model_out, max_trees_per_model)
        if tree_paths:
            artifacts.tree_plots[name] = tree_paths

    # Optional: feature importance bar chart (one figure per model)
    if feature_importance:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            imp_dir = output_dir / "feature_importance"
            _safe_mkdir(imp_dir)
            for name, imp_dict in feature_importance.items():
                if not imp_dict:
                    continue
                items = sorted(imp_dict.items(), key=lambda x: -abs(x[1]))[:20]
                if not items:
                    continue
                labels, values = zip(*items)
                fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.25)))
                y_pos = np.arange(len(labels))
                ax.barh(y_pos, values, align="center")
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels, fontsize=9)
                ax.invert_yaxis()
                ax.set_xlabel("Importance")
                ax.set_title(f"Feature importance — {name}")
                plt.tight_layout()
                out_path = imp_dir / f"{name}_importance.png"
                plt.savefig(out_path, bbox_inches="tight", dpi=100)
                plt.close()
            if imp_dir.exists():
                artifacts.feature_importance_plot = imp_dir
            logger.info(f"Feature importance plots saved to {imp_dir}")
        except Exception as e:
            logger.warning(f"Feature importance plot failed: {e}")
            artifacts.errors.append(f"feature_importance: {e}")

    return artifacts
