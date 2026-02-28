import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
MTEB_PATH   = os.path.join(RESULTS_DIR, 'contriever_mrl.json')

DIMS  = [64, 128, 256, 512, 768]          
TASKS = ['NFCorpus', 'SciFact', 'ArguAna', 'SCIDOCS', 'FiQA2018']

BLUE   = '#2563EB'   # fine tuned
RED    = '#DC2626'   # baseline
LGRAY  = '#E5E7EB'

def load_ndcg(mteb: dict, model_key: str, metric: str = 'ndcg_at_10') -> dict:
    result = {}
    for dim in DIMS:
        dim_data = mteb[model_key].get(str(dim), {})
        result[dim] = {task: dim_data[task][metric]
                       for task in TASKS if task in dim_data}
    return result


def avg_across_tasks(ndcg_by_dim: dict) -> dict:
    return {dim: float(np.mean(list(task_scores.values())))
            for dim, task_scores in ndcg_by_dim.items()
            if task_scores}


def apply_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.4,
    })

def plot_avg_vs_dim(ft_avg: dict, bl_avg: dict, out_path: str):
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    dims = DIMS
    ft_y = [ft_avg[d] for d in dims]
    bl_y = [bl_avg[d] for d in dims]

    #shaded region between curves
    ax.fill_between(dims, bl_y, ft_y, alpha=0.10, color=BLUE, label='_nolegend_')

    ax.plot(dims, bl_y, color=RED,  linewidth=2.0, linestyle='--',
            marker='s', markersize=7, label='Baseline Contriever')
    ax.plot(dims, ft_y, color=BLUE, linewidth=2.0, linestyle='-',
            marker='o', markersize=7, label='Fine-tuned MRL Contriever')

    #annotate gain at each dim
    for d, fy, by in zip(dims, ft_y, bl_y):
        gain = fy - by
        mid  = (fy + by) / 2
        ax.annotate(
            f'+{gain:.3f}',
            xy=(d, mid),
            ha='center', va='center',
            fontsize=8, color='#1D4ED8',
            fontweight='bold',
        )

    ax.set_xticks(dims)
    ax.set_xlabel('Embedding Dimension', fontsize=12)
    ax.set_ylabel('Avg nDCG@10  (5 MTEB tasks)', fontsize=12)
    ax.set_title(
        'Accuracy vs Embedding Dimension\nFine-tuned MRL Contriever vs Baseline',
        fontsize=13, fontweight='bold', pad=12,
    )
    ax.legend(loc='lower right', framealpha=0.9, fontsize=10)

    task_str = ', '.join(TASKS)
    fig.text(0.5, 0.01, f'Tasks: {task_str}', ha='center', fontsize=8, color='gray')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved - {out_path}')

def plot_per_task(ft_ndcg: dict, bl_ndcg: dict, out_path: str):
    apply_style()
    n = len(TASKS)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 5), sharey=False)

    for ax, task in zip(axes, TASKS):
        ft_y = [ft_ndcg[d].get(task, 0.0) for d in DIMS]
        bl_y = [bl_ndcg[d].get(task, 0.0) for d in DIMS]

        ax.fill_between(DIMS, bl_y, ft_y, alpha=0.10, color=BLUE)
        ax.plot(DIMS, bl_y, color=RED,  linewidth=1.8, linestyle='--',
                marker='s', markersize=6, label='Baseline')
        ax.plot(DIMS, ft_y, color=BLUE, linewidth=1.8, linestyle='-',
                marker='o', markersize=6, label='Fine-tuned MRL')

        ax.set_xticks(DIMS)
        ax.set_xticklabels([str(d) for d in DIMS], rotation=45, fontsize=8)
        ax.set_title(task, fontsize=11, fontweight='bold')
        ax.set_xlabel('Dim', fontsize=9)
        ax.set_ylabel('nDCG@10', fontsize=9)

    # Shared legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2,
               framealpha=0.9, fontsize=10, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(
        'Per-task nDCG@10 — Fine-tuned MRL Contriever vs Baseline',
        fontsize=13, fontweight='bold', y=1.06,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved - {out_path}')

def main():
    if not os.path.exists(MTEB_PATH):
        print(f'ERROR: MTEB results not found at {MTEB_PATH}')
        sys.exit(1)

    with open(MTEB_PATH) as f:
        mteb = json.load(f)

    ft_ndcg = load_ndcg(mteb, 'fine_tuned')
    bl_ndcg = load_ndcg(mteb, 'baseline')
    ft_avg  = avg_across_tasks(ft_ndcg)
    bl_avg  = avg_across_tasks(bl_ndcg)

    #print summary table
    print(f'\n{"Dim":>6} | {"FT nDCG@10":>12} | {"BL nDCG@10":>12} | {"Gain":>8}')
    print('-' * 50)
    for d in DIMS:
        ft = ft_avg.get(d, 0.0)
        bl = bl_avg.get(d, 0.0)
        print(f'{d:>6} | {ft:>12.5f} | {bl:>12.5f} | {ft - bl:>+8.5f}')

    out1 = os.path.join(RESULTS_DIR, 'pareto_accuracy_vs_dim.png')
    out2 = os.path.join(RESULTS_DIR, 'pareto_per_task.png')

    plot_avg_vs_dim(ft_avg, bl_avg, out1)
    plot_per_task(ft_ndcg, bl_ndcg, out2)


if __name__ == '__main__':
    main()
