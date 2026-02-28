from src.utils.config import load_config
from src.utils.logger import get_logger
from src.models.embedding_model import EmbeddingModel
from src.evaluation.evaluator import MRLModelWrapper
from transformers import AutoTokenizer
from dotenv import load_dotenv
import argparse
import torch
import mteb
import json
import os

load_dotenv()

logger = get_logger(__file__)

TASKS = [
    'NFCorpus',
    'SciFact',
    'ArguAna',
    'SCIDOCS',
    'FiQA2018',
]

DEFAULT_DIMS = [768, 512, 256, 128, 64]

parser = argparse.ArgumentParser(description='Evaluate the MRL model across dimensions using MTEB')
parser.add_argument('--config', type=str, default=None, help='path to config file')
parser.add_argument('--checkpoint', type=str, required=True, help='path to model checkpoint .pt file')
parser.add_argument('--experiment-name', type=str, required=True, help='name used for saving results')
parser.add_argument('--tasks', type=str, nargs='+', default=TASKS, help='MTEB tasks to run')
parser.add_argument('--dims', type=int, nargs='+', default=DEFAULT_DIMS, help='dimension slices to evaluate')
parser.add_argument('--batch-size', type=int, default=64, help='encoding batch size')

args = parser.parse_args()
config = load_config(args.config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')

tokenizer = AutoTokenizer.from_pretrained(config['model']['name'], trust_remote_code=True)

def load_model(checkpoint_path=None):
    m = EmbeddingModel(config['model']['name']).to(device)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        m.load_state_dict(ckpt['model_state_dict'])
        logger.info(f'Loaded checkpoint: {checkpoint_path}')
    m.eval()
    return m

TASK_TYPE_MAP = {
    'NFCorpus': 'Retrieval',
    'SciFact':  'Retrieval',
    'ArguAna':  'Retrieval',
    'SCIDOCS':  'Retrieval',
    'FiQA2018': 'Retrieval',
}

TASK_METRICS = {
    'Retrieval': [
        'ndcg_at_1', 'ndcg_at_10', 'ndcg_at_100', 'ndcg_at_1000',
        'recall_at_1', 'recall_at_10', 'recall_at_100', 'recall_at_1000',
        'precision_at_1', 'precision_at_10', 'precision_at_100', 'precision_at_1000',
        'mrr_at_10', 'mrr_at_100',
        'map_at_10', 'map_at_100', 'map_at_1000'
    ],
}

#top 3 most common headline metrics
TASK_PRIMARY = {
    'Retrieval': [
        'ndcg_at_10',  
        'mrr_at_10',    
        'map_at_100'   
    ]
}

def flatten_scores(scores):
    if isinstance(scores, dict):
        return scores
    if isinstance(scores, list):
        if len(scores) > 0 and isinstance(scores[0], dict):
            merged = {}
            for d in scores:
                for k, v in d.items():
                    if isinstance(v, (int, float)):
                        merged.setdefault(k, []).append(v)
            return {k: sum(v) / len(v) for k, v in merged.items()}
    return {}

def extract_metrics(task_result, scores):
    task_type = TASK_TYPE_MAP.get(task_result.task_name, 'Retrieval')
    wanted    = TASK_METRICS.get(task_type, TASK_METRICS['Retrieval'])
    primary   = TASK_PRIMARY.get(task_type, ['ndcg_at_10'])
    flat      = flatten_scores(scores)
    out = {'task_type': task_type, 'primary_metrics': primary}
    for m in wanted:
        out[m] = flat.get(m, None)
    return out

#runs evaluation on each dim
def run_evaluation(model, label, dims, tasks):
    mteb_tasks  = mteb.get_tasks(tasks=tasks)
    all_results = {}
    for dim in dims:
        logger.info(f'\n[{label}] dim={dim}')
        wrapper    = MRLModelWrapper(model, tokenizer, dim, device,
                                     batch_size=args.batch_size,
                                     max_length=config['data']['max_length'])
        evaluation = mteb.MTEB(tasks=mteb_tasks)
        results    = evaluation.run(wrapper, output_folder=None)

        dim_results = {}
        for r in results:
            scores = r.scores.get('test', r.scores.get('dev', {}))
            dim_results[r.task_name] = extract_metrics(r, scores)

        all_results[dim] = dim_results
    return all_results


#evaluating fine tuned model
logger.info('Evaluating Fine Tuned MRL model')
finetuned_model = load_model(args.checkpoint)
finetuned_results = run_evaluation(finetuned_model, 'fine-tuned', args.dims, args.tasks)

#evaluating baseline, same contriever model, no fine-tuning
logger.info('Evaluating Baseline')
baseline_model = load_model(checkpoint_path=None)
baseline_results = run_evaluation(baseline_model, 'baseline', args.dims, args.tasks)

#saving results
output = {
    'model':      config['model']['name'],
    'checkpoint': args.checkpoint,
    'tasks':      args.tasks,
    'dims':       args.dims,
    'fine_tuned': {str(k): v for k, v in finetuned_results.items()},
    'baseline':   {str(k): v for k, v in baseline_results.items()},
}

os.makedirs('results', exist_ok=True)
output_path = os.path.join('results', f'evaluation_{args.experiment_name}.json')
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

logger.info(f'Results saved to {output_path}')

#summary table
print(f'RESULTS SUMMARY — {args.experiment_name}')

header = f"{'Model':<12} {'Dim':>5}  " + "  ".join(f"{t[:10]:>10}" for t in args.tasks)
print(header)
print('-' * len(header))

for model_label, results in [('Fine-tuned', finetuned_results), ('Baseline', baseline_results)]:
    for dim in args.dims:
        row = f"{model_label:<12} {dim:>5}  "
        for task in args.tasks:
            score = results.get(dim, {}).get(task, {}).get('ndcg_at_10', None)
            row += f"  {score:>10.4f}" if score is not None else f"  {'N/A':>10}"
        print(row)
    print()
