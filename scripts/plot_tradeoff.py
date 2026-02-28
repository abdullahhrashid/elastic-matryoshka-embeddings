import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
MTEB_PATH = os.path.join(RESULTS_DIR, 'contriever_mrl.json')
LATENCY_PATH = os.path.join(RESULTS_DIR, 'latency_benchmark.json')
OUT_PATH = os.path.join(RESULTS_DIR, 'accuracy_vs_speed.png')

DIMS = [768, 512, 256, 128, 64]
TASKS = ['NFCorpus', 'SciFact', 'ArguAna', 'SCIDOCS', 'FiQA2018']

def avg_ndcg_at_10(mteb_data, model_key):
    scores = {}
    for dim in DIMS:
        dim_data = mteb_data[model_key].get(str(dim), {})
        ndcg_values = [dim_data[task]['ndcg_at_10'] for task in TASKS if task in dim_data]
        scores[dim] = float(np.mean(ndcg_values)) if ndcg_values else 0.0
    return scores


def main():
    if not os.path.exists(MTEB_PATH):
        print(f'ERROR: MTEB results not found at {MTEB_PATH}')
        sys.exit(1)
    if not os.path.exists(LATENCY_PATH):
        print(f'ERROR: Latency results not found at {LATENCY_PATH}')
        print('Run `python scripts/benchmark.py` first.')
        sys.exit(1)

    with open(MTEB_PATH) as f:
        mteb = json.load(f)
    with open(LATENCY_PATH) as f:
        latency = json.load(f)

    ft_ndcg = avg_ndcg_at_10(mteb, 'fine_tuned')
    bl_ndcg = avg_ndcg_at_10(mteb, 'baseline')

    dims_sorted = sorted(DIMS)

    ft_x = [latency[str(d)]['p50_ms'] for d in dims_sorted]
    ft_y = [ft_ndcg[d] for d in dims_sorted]
    bl_x_768 = latency['768']['p50_ms']
    bl_y_768 = bl_ndcg[768]

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        ft_x, ft_y,
        color='#2563EB',
        linewidth=2.0,
        linestyle='-',
        zorder=2,
    )
    scatter_ft = ax.scatter(
        ft_x, ft_y,
        color='#2563EB',
        s=100,
        zorder=3,
        label='Fine-tuned MRL Contriever',
    )

    for d, x, y in zip(dims_sorted, ft_x, ft_y):
        ax.annotate(
            f'{d}d',
            xy=(x, y),
            xytext=(6, 5),
            textcoords='offset points',
            fontsize=10,
            color='#1E40AF',
            fontweight='bold',
        )

    ax.scatter(
        [bl_x_768], [bl_y_768],
        color='#DC2626',
        s=120,
        marker='D',
        zorder=4,
        label='Baseline Contriever (768d)',
    )
    ax.annotate(
        'Baseline\n768d',
        xy=(bl_x_768, bl_y_768),
        xytext=(8, -18),
        textcoords='offset points',
        fontsize=9,
        color='#991B1B',
    )

    ax.text(
        0.03, 0.95, '← Faster',
        transform=ax.transAxes,
        fontsize=9, color='gray', ha='left', va='top',
    )
    ax.text(
        0.97, 0.03, 'More Accurate →',
        transform=ax.transAxes,
        fontsize=9, color='gray', ha='right', va='bottom',
    )

    ax.set_xlabel('FAISS Search Latency — P50 (ms)', fontsize=12)
    ax.set_ylabel('Avg nDCG@10 (5 MTEB tasks)', fontsize=12)
    ax.set_title(
        'Accuracy vs. Speed Tradeoff\nMatryoshka MRL Contriever — Adaptive Embedding Dimensions',
        fontsize=13, fontweight='bold', pad=14,
    )
    ax.legend(loc='lower right', framealpha=0.9, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)

    task_str = ', '.join(TASKS)
    fig.text(
        0.5, 0.01,
        f'Tasks: {task_str}',
        ha='center', fontsize=8, color='gray',
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Plot saved → {OUT_PATH}')

    print(f'\n{"Dim":>6} | {"P50 Latency (ms)":>18} | {"Avg nDCG@10 (FT)":>18} | {"Avg nDCG@10 (BL)":>18}')
    for d in dims_sorted:
        lat_str = f"{latency[str(d)]['p50_ms']:.3f}"
        print(f'{d:>6} | {lat_str:>18} | {ft_ndcg[d]:>18.5f} | {bl_ndcg.get(d, 0.0):>18.5f}')

if __name__ == '__main__':
    main()
