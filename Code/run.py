import argparse
import os
import datetime
import math
import sys
import time
import platform
import subprocess

import numpy as np
import torch
try:
    import ujson as json
except ImportError:
    import json
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
try:
    from transformers.optimization import AdamW, get_linear_schedule_with_warmup
except ImportError:
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

from args import add_args
from model import HDERModel
from utils import set_seed, collate_fn, create_directory
from prepro import read_docred, get_srr_entity_types, SRR_ENTITY_TYPE_NAMES, SRR_ENTITY_TYPE_TO_ID
from evaluation import to_official, official_evaluate, merge_results
# [BUG-FIX-1] wandb replaced with offline stub to avoid login requirement
try:
    import wandb
except ImportError:
    class _WandbStub:
        def init(self, **kwargs): pass
        def log(self, d, **kwargs): pass
    wandb = _WandbStub()
from tqdm import tqdm

import pandas as pd
import pickle
def normalize_model_switches(args):
    if not args.use_hae:
        args.use_cross_layer_attention = False
        args.use_sentence_position_encoding = False
    return args

def print_model_config(args):
    print("===== Final Model Configuration =====")
    print(f"EXPERIMENT_NAME: {args.experiment_name}")
    print(f"RUN_MODE: {args.run_mode}")
    print(f"USE_HAE: {args.use_hae}")
    print(f"USE_CROSS_LAYER_ATTENTION: {args.use_cross_layer_attention}")
    print(f"USE_SENTENCE_POSITION_ENCODING: {args.use_sentence_position_encoding}")
    print(f"GLOBAL_CONTEXT_MODE: {args.global_context_mode}")
    print(f"GLOBAL_MEAN_PATH: {args.global_mean_path}")
    print(f"USE_SRR: {args.use_srr}")
    print(f"SRR_STEPS: {args.srr_steps}")
    print(f"SRR_TOKEN_ATTENTION_SCALE: {args.srr_token_attention_scale}")
    print(f"SRR_INIT: {args.srr_init}")
    print(f"USE_SPAN_BOUNDARY_HEAD: {args.use_span_boundary_head}")
    print(f"SPAN_AUX_WEIGHT: {args.span_aux_weight}")
    print(f"SPAN_DETACH_BACKBONE: {args.span_detach_backbone}")
    print(f"AUX_GRAPH_PATH: {args.aux_graph_path}")
    print(f"ENTITY_TYPE_CONFLICT_POLICY: {args.entity_type_conflict_policy}")
    print(f"PARAGRAPH_STRATEGY: {args.paragraph_strategy}")
    print(f"SENTENCES_PER_PARAGRAPH: {args.sentences_per_paragraph}")
    print(f"SEED: {args.seed}")
    print(f"TRAIN_BATCH_SIZE: {args.train_batch_size}")
    print(f"GRADIENT_ACCUMULATION_STEPS: {args.gradient_accumulation_steps}")
    print(f"EFFECTIVE_BATCH_SIZE: {getattr(args, 'effective_batch_size', 'pending')}")
    print(f"NUM_TRAIN_EPOCHS: {args.num_train_epochs}")
    print(f"LEARNING_RATE: {args.lr_transformer}")
    print(f"LR_ADDED: {args.lr_added}")
    print(f"WARMUP_RATIO: {args.warmup_ratio}")
    print(f"MAX_SEQ_LENGTH: {args.max_seq_length}")
    print(f"TRANSFORMER_TYPE: {args.transformer_type}")
    print(f"MODEL_NAME_OR_PATH: {args.model_name_or_path}")
    print("=====================================")


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def save_json(path, payload):
    create_directory(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)


def get_environment_info():
    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = "unknown"
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        git_commit = "unavailable"
    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "gpu_names": gpu_names,
        "gpu_count": torch.cuda.device_count(),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "transformers": transformers_version,
        "git_commit": git_commit,
        "code_package_version": "HDER_final_experiment_ready",
    }


def count_parameters(model):
    named = list(model.named_parameters())
    def count_where(pred):
        return sum(p.numel() for n, p in named if pred(n, p))
    total = sum(p.numel() for _, p in named)
    trainable = sum(p.numel() for _, p in named if p.requires_grad)
    counts = {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
        "encoder_params": count_where(lambda n, p: n.startswith("model.")),
        "hae_params": count_where(lambda n, p: n.startswith("hae.")),
        "srr_params": count_where(lambda n, p: n.startswith("srr.")),
        "classifier_params": count_where(lambda n, p: "classifier" in n or "bilinear" in n or "extractor" in n),
    }
    print("===== Parameter Counts =====")
    for key, value in counts.items():
        print(f"{key.upper()}: {value}")
    print("============================")
    return counts


def validate_experiment_configuration(args, model, optimizer_grouped_parameters=None):
    checks = {}
    param_names = [n for n, _ in model.named_parameters()]
    has_hae_params = any(n.startswith("hae.") for n in param_names)
    has_srr_params = any(n.startswith("srr.") for n in param_names)
    collect_srr_entity_types = args.use_srr
    checks["collect_srr_entity_types"] = collect_srr_entity_types

    if not args.use_hae:
        if getattr(model, "hae", None) is not None or has_hae_params:
            raise ValueError("w/o HAE must not instantiate HAE.")
        if args.use_srr and getattr(model, "srr", None) is None:
            raise ValueError("w/o HAE with use_srr=true must instantiate SRR.")

    if not args.use_srr:
        if getattr(model, "srr", None) is not None or has_srr_params:
            raise ValueError("w/o SRR must not instantiate SRR.")
        if collect_srr_entity_types:
            raise ValueError("w/o SRR must not collect entity_type.")
        if getattr(model, "span_boundary_head", None) is not None:
            raise ValueError("w/o SRR must not instantiate the SRR-conditioned span boundary head.")

    if args.use_srr and args.use_span_boundary_head:
        if getattr(model, "span_boundary_head", None) is None:
            raise ValueError("Span boundary head requested but not instantiated.")

    if args.use_hae and not args.use_cross_layer_attention:
        hae = getattr(model, "hae", None)
        if hae is None:
            raise ValueError("no_cross_layer requires HAE.")
        if any(hasattr(hae, name) for name in ["query", "key", "value"]):
            raise ValueError("no_cross_layer must not instantiate cross-layer Q/K/V.")
        if not hasattr(hae, "bottom_up_fusion"):
            raise ValueError("no_cross_layer must keep bottom-up fusion.")

    if args.use_hae and not args.use_sentence_position_encoding:
        hae = getattr(model, "hae", None)
        if hae is None:
            raise ValueError("no_position requires HAE.")
        if hasattr(hae, "sentence_position_embedding"):
            raise ValueError("no_position must not instantiate sentence_position_embedding.")

    if args.global_context_mode == "fixed_mean":
        if not hasattr(model, "fixed_global_mean"):
            raise ValueError("fixed_mean mode must register fixed_global_mean buffer.")
        buf = model.fixed_global_mean
        if tuple(buf.shape) != (model.hidden_size,):
            raise ValueError(f"fixed_global_mean shape mismatch: {tuple(buf.shape)}")
        if buf.requires_grad:
            raise ValueError("fixed_global_mean must have requires_grad=False.")
        if "fixed_global_mean" not in model.state_dict():
            raise ValueError("fixed_global_mean must be in state_dict.")
        metadata = getattr(model, "fixed_global_mean_metadata", None)
        if not metadata or metadata.get("source_split") != "train":
            raise ValueError("fixed_mean metadata must be train-only.")
    else:
        if hasattr(model, "fixed_global_mean"):
            raise ValueError("dynamic mode must not register fixed_global_mean.")

    if optimizer_grouped_parameters is not None:
        ids = []
        for group in optimizer_grouped_parameters:
            ids.extend(id(p) for p in group["params"])
        if len(ids) != len(set(ids)):
            raise ValueError("Optimizer groups contain duplicate parameters.")
        trainable_ids = {id(p) for _, p in model.named_parameters() if p.requires_grad}
        if set(ids) != trainable_ids:
            raise ValueError("Optimizer groups missing trainable parameters.")
        if hasattr(model, "fixed_global_mean") and id(model.fixed_global_mean) in set(ids):
            raise ValueError("fixed_global_mean buffer must not enter optimizer.")
    checks["status"] = "PASS"
    print("CONFIG_VALIDATION_PASS")
    return checks


def build_srr_cooccurrence_init(train_file, output_path, entity_type_conflict_policy="majority_first"):
    with open(train_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    counts = np.zeros((6, 6), dtype=np.float64)
    conflict_records = []
    total_entities = 0
    for doc_idx, sample in enumerate(data):
        total_entities += len(sample["vertexSet"])
        entity_types = get_srr_entity_types(
            sample["vertexSet"],
            sample.get("title", doc_idx),
            conflict_policy=entity_type_conflict_policy,
            conflict_records=conflict_records,
        )
        for head_idx, head_type in enumerate(entity_types):
            for tail_idx, tail_type in enumerate(entity_types):
                if head_idx != tail_idx:
                    counts[head_type, tail_type] += 1

    probabilities = (counts + 1.0) / (counts.sum(axis=1, keepdims=True) + 6.0)
    payload = {
        "type_order": SRR_ENTITY_TYPE_NAMES,
        "type_id": SRR_ENTITY_TYPE_TO_ID,
        "doc_count": len(data),
        "statistic_unit": "ordered_entity_pairs_within_each_train_document",
        "entity_type_conflict_policy": entity_type_conflict_policy,
        "total_entities": total_entities,
        "conflict_entity_count": len(conflict_records),
        "train_file": os.path.abspath(train_file),
        "counts": counts.astype(int).tolist(),
        "probabilities": probabilities.tolist(),
        "conflict_examples": conflict_records[:5],
    }
    create_directory(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"SRR_COOCCURRENCE_INIT_SAVED: {output_path}")
    print(f"SRR_COOCCURRENCE_ENTITY_TYPE_CONFLICT_POLICY: {entity_type_conflict_policy}")
    print(f"SRR_COOCCURRENCE_CONFLICT_ENTITY_COUNT: {len(conflict_records)}")
    return probabilities.astype(np.float32)


def load_srr_cooccurrence_init(path):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"SRR cooccurrence initialization file is required but missing: {path}"
        )
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("type_order") != SRR_ENTITY_TYPE_NAMES:
        raise ValueError(f"Unexpected SRR type order in {path}: {payload.get('type_order')}")
    probabilities = np.array(payload["probabilities"], dtype=np.float32)
    if probabilities.shape != (6, 6):
        raise ValueError(f"SRR cooccurrence matrix must be 6x6, got {probabilities.shape}")
    print(f"SRR_COOCCURRENCE_INIT_LOADED: {path}")
    print(f"SRR_COOCCURRENCE_STATISTIC_UNIT: {payload.get('statistic_unit')}")
    print(f"SRR_COOCCURRENCE_DOC_COUNT: {payload.get('doc_count')}")
    if "entity_type_conflict_policy" in payload:
        print(f"SRR_COOCCURRENCE_ENTITY_TYPE_CONFLICT_POLICY: {payload.get('entity_type_conflict_policy')}")
    if "conflict_entity_count" in payload:
        print(f"SRR_COOCCURRENCE_CONFLICT_ENTITY_COUNT: {payload.get('conflict_entity_count')}")
    return probabilities



def load_optional_aux_graph_prior(path):
    """Load optional precomputed auxiliary type-interaction graph statistics.

    The public package intentionally does not ship this local file. The loader
    accepts either {"adjacency": [[...]]} or {"probabilities": [[...]]}.
    Missing files are handled by the caller as a learned-initialization fallback.
    """
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        matrix = payload.get("adjacency", payload.get("probabilities"))
        type_order = payload.get("type_order")
        if type_order is not None and type_order != SRR_ENTITY_TYPE_NAMES:
            raise ValueError(f"Unexpected SRR type order in {path}: {type_order}")
    else:
        matrix = payload
    if matrix is None:
        raise ValueError(f"Auxiliary graph prior file {path} must contain 'adjacency' or 'probabilities'.")
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (6, 6):
        raise ValueError(f"Auxiliary graph prior matrix must be 6x6, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"Auxiliary graph prior matrix contains non-finite values: {path}")
    if (matrix < 0).any() or (matrix > 1).any():
        raise ValueError(f"SRR prior values must be in [0,1]: {path}")
    print(f"AUX_TYPE_GRAPH_LOADED: {path}")
    return matrix

def prepare_srr_adjacency_init(args, save_path):
    if not args.use_srr:
        return None
    if args.srr_steps <= 0:
        raise ValueError("--srr_steps must be positive when --use_srr true.")

    # Optional precomputed auxiliary graph statistics have highest priority when available.
    # HDER remains runnable when the file is absent.
    local_prior = load_optional_aux_graph_prior(args.aux_graph_path)
    if local_prior is not None:
        return local_prior
    if args.aux_graph_path:
        print(
            "[warn] Optional auxiliary type-interaction graph file is not present: "
            f"{args.aux_graph_path}. Falling back to learned SRR adjacency initialization."
        )

    if args.srr_init == "learned":
        return None

    if args.srr_cooccurrence_path:
        return load_srr_cooccurrence_init(args.srr_cooccurrence_path)

    if not args.do_train:
        raise FileNotFoundError("--srr_cooccurrence_path is required for cooccurrence init outside training.")
    train_file = os.path.join(args.data_dir, args.train_file)
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Cannot build SRR cooccurrence init; train file missing: {train_file}")
    output_path = os.path.join(save_path, "srr_type_cooccurrence.json")
    return build_srr_cooccurrence_init(train_file, output_path, args.entity_type_conflict_policy)


def build_optimizer_grouped_parameters(args, model, new_layer=None):
    if new_layer is None:
        new_layer = ["extractor", "bilinear", "hae", "srr", "span_boundary_head"]

    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    added_names = {n for n, _ in named_params if any(nd in n for nd in new_layer)}
    transformer_names = {n for n, _ in named_params if n not in added_names}

    added_params = [p for n, p in named_params if n in added_names]
    transformer_params = [p for n, p in named_params if n in transformer_names]
    all_param_ids = [id(p) for _, p in named_params]
    grouped_param_ids = [id(p) for p in transformer_params + added_params]

    if len(grouped_param_ids) != len(set(grouped_param_ids)):
        raise ValueError("Optimizer parameter grouping has duplicate parameters.")
    if set(grouped_param_ids) != set(all_param_ids):
        raise ValueError("Optimizer parameter grouping has missing parameters.")

    hae_names = {n for n, _ in named_params if n.startswith("hae.")}
    if hae_names and not hae_names.issubset(added_names):
        missing = sorted(hae_names - added_names)
        raise ValueError(f"HAE parameters are not in lr_added group: {missing[:5]}")

    srr_names = {n for n, _ in named_params if n.startswith("srr.")}
    if srr_names and not srr_names.issubset(added_names):
        missing = sorted(srr_names - added_names)
        raise ValueError(f"SRR parameters are not in lr_added group: {missing[:5]}")

    roberta_names = {n for n, _ in named_params if n.startswith("model.")}
    if roberta_names & added_names:
        overlap = sorted(roberta_names & added_names)
        raise ValueError(f"Transformer parameters leaked into lr_added group: {overlap[:5]}")

    print(
        "OPTIMIZER_GROUP_CHECK: "
        f"transformer_params={len(transformer_params)}, "
        f"added_params={len(added_params)}, "
        f"hae_params={len(hae_names)}, "
        f"srr_params={len(srr_names)}, "
        f"new_layer={new_layer}"
    )
    return [
        {"params": transformer_params},
        {"params": added_params, "lr": args.lr_added},
    ]


def load_input(batch, device, tag="dev"):

    input = {'input_ids': batch[0].to(device),
            'attention_mask': batch[1].to(device),
            'labels': batch[2].to(device),
            'entity_pos': batch[3],
            'hts': batch[4],
            'sent_pos': batch[5],
            'sent_labels': batch[6].to(device) if (not batch[6] is None) and (batch[7] is None) else None,
            'teacher_attns': batch[7].to(device) if not batch[7] is None else None,
            'tag': tag,
            'entity_type': batch[8],
            } 

    return input

def train(args, model, train_features, dev_features):

    def finetune(features, optimizer, num_epoch, num_steps):
        args._runtime_status = getattr(args, "_runtime_status", {})
        args._runtime_status.setdefault("epoch_seconds", [])
        args._runtime_status.setdefault("eval_seconds", [])
        args._runtime_status.setdefault("step_seconds", [])
        best_score = -1e9
        best_offi_results = []
        best_results = []
        best_output = []
        best_epoch = -1
        best_metrics = {}
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_iterator = range(int(num_epoch))
        updates_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        total_steps = int(updates_per_epoch * num_epoch)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        scaler = GradScaler(enabled=(args.device.type == "cuda"))
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        for epoch in tqdm(train_iterator, desc='Train epoch'):
            epoch_start = time.time()
            optimizer.zero_grad()
            for step, batch in enumerate(train_dataloader):
                step_start = time.time()
                model.train()

                inputs = load_input(batch, args.device)  
                outputs = model(**inputs)
                loss = [outputs["loss"]["rel_loss"]]

                if inputs["sent_labels"] is not None:
                    loss.append(outputs["loss"]["evi_loss"] * args.evi_lambda)
                                
                if inputs["teacher_attns"] is not None:
                    loss.append(outputs["loss"]["attn_loss"] * args.attn_lambda)
                
                loss = sum(loss) / args.gradient_accumulation_steps
                loss_is_finite = torch.isfinite(loss.detach()).all().item()
                args._runtime_status["loss_is_finite"] = bool(loss_is_finite)
                if not loss_is_finite:
                    raise ValueError("Training loss is NaN or Inf.")
                scaler.scale(loss).backward()
                args._runtime_status["backward_completed"] = True

                should_update = (
                    (step + 1) % args.gradient_accumulation_steps == 0
                    or (step + 1) == len(train_dataloader)
                )
                if should_update:
                    if args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    num_steps += 1
                    args._runtime_status["optimizer_step_completed"] = True
                    args._runtime_status["completed_optimizer_steps"] = num_steps
                    args._runtime_status["step_seconds"].append(time.time() - step_start)
                    
                log_losses = {k: float(v.detach().cpu()) for k, v in outputs["loss"].items()}
                wandb.log(log_losses, step=num_steps)
                if((step+1) % 10 == 0): print("loss:",loss.item())
                if args.max_train_batches > 0 and num_steps >= args.max_train_batches:
                    print(f"MAX_TRAIN_BATCHES_REACHED: stopped after {num_steps} optimizer steps.")
                    return num_steps
                if args.run_mode == "acceptance":
                    continue
                if (step + 1) == len(train_dataloader) or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    
                    eval_start = time.time()
                    dev_scores, dev_output, official_results, results = evaluate(args, model, dev_features, tag="dev")
                    args._runtime_status["eval_seconds"].append(time.time() - eval_start)
                    wandb.log(dev_scores, step=num_steps)
                    
                    print(dev_output)
                    if epoch >= 19:
                        ckpt_file = os.path.join(args.save_path, f"{epoch}.ckpt")
                        print(f"saving model checkpoint into {ckpt_file} ...")
                        torch.save(model.state_dict(), ckpt_file)
                    if dev_scores["dev_F1_ign"] > best_score:
                        best_score = dev_scores["dev_F1_ign"]
                        best_offi_results = official_results
                        best_results = results
                        best_output = dev_output
                        best_epoch = epoch
                        best_metrics = dict(dev_scores)

                        ckpt_file = os.path.join(args.save_path, "best.ckpt")
                        print(f"saving model checkpoint into {ckpt_file} ...")
                        torch.save(model.state_dict(), ckpt_file)
                        
                    if epoch == train_iterator[-1]: # last epoch

                        ckpt_file = os.path.join(args.save_path, "last.ckpt")
                        print(f"saving model checkpoint into {ckpt_file} ...")
                        torch.save(model.state_dict(), ckpt_file)
                        
                        pred_file = os.path.join(args.save_path, args.pred_file)
                        score_file = os.path.join(args.save_path, "scores.csv")
                        results_file = os.path.join(args.save_path, f"topk_{args.pred_file}")

                        dump_to_file(best_offi_results, pred_file, best_output, score_file, best_results, results_file)
                        metrics_payload = {
                            "run_mode": args.run_mode,
                            "best_epoch": best_epoch,
                            "best_step": num_steps,
                            "best_dev_scores": best_metrics,
                            "best_output": best_output,
                            "prediction_count": len(best_offi_results),
                            "best_checkpoint_path": os.path.join(args.save_path, "best.ckpt"),
                            "scores_csv": score_file,
                            "predictions": pred_file,
                        }
                        save_json(os.path.join(args.save_path, "metrics.json"), metrics_payload)
                     
            args._runtime_status["epoch_seconds"].append(time.time() - epoch_start)
        return num_steps

    optimizer_grouped_parameters = build_optimizer_grouped_parameters(args, model)
    validation = validate_experiment_configuration(args, model, optimizer_grouped_parameters)
    save_json(os.path.join(args.save_path, "config_validation.json"), validation)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr_transformer, eps=args.adam_epsilon)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    start_time = time.time()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps)
    runtime = getattr(args, "_runtime_status", {})
    runtime.update({
        "run_mode": args.run_mode,
        "train_total_seconds": time.time() - start_time,
        "avg_step_seconds": (
            float(np.mean(runtime.get("step_seconds", [])))
            if runtime.get("step_seconds") else None
        ),
        "total_eval_seconds": float(sum(runtime.get("eval_seconds", []))),
        "peak_gpu_memory": torch.cuda.max_memory_allocated(args.device) if args.device.type == "cuda" else 0,
        "exit_code": 0,
    })
    save_json(os.path.join(args.save_path, "runtime.json"), runtime)


def evaluate(args, model, features, tag="dev"):
    
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds, evi_preds = [], []
    scores, topks = [], []
    attns = []
    for batch in tqdm(dataloader, desc=f"Evaluating batches"):
        model.eval()

        if args.save_attn:
            tag = "infer"

        inputs = load_input(batch, args.device, tag)

        with torch.no_grad():
            outputs = model(**inputs)
            pred = outputs["rel_pred"]
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
            if "scores" in outputs:
                scores.append(outputs["scores"].cpu().numpy())  
                topks.append(outputs["topks"].cpu().numpy())   

            if "evi_pred" in outputs: # relation extraction and evidence extraction
                evi_pred = outputs["evi_pred"]
                evi_pred = evi_pred.cpu().numpy()
                evi_preds.append(evi_pred)   
            
            if "attns" in outputs: # attention recorded
                attn = outputs["attns"]
                attns.extend([a.cpu().numpy() for a in attn])


    preds = np.concatenate(preds, axis=0)
    
    if len(scores) != 0:
        scores = np.concatenate(scores, axis=0)
        topks =  np.concatenate(topks, axis=0)

    if len(evi_preds) != 0:
        evi_preds = np.concatenate(evi_preds, axis=0)
    
    official_results, results = to_official(preds, features, evi_preds = evi_preds, scores = scores, topks = topks)
    
    if len(official_results) > 0:
        if tag == "dev":
            best_re, best_evi, best_re_ign, _ = official_evaluate(official_results, args.data_dir, args.train_file, args.dev_file)
        else:
            best_re, best_evi, best_re_ign, _ = official_evaluate(official_results, args.data_dir, args.train_file, args.test_file)
    else:
        best_re = best_evi = best_re_ign = [-1, -1, -1]
    output = {
        tag + "_rel": [i * 100 for i in best_re],
        tag + "_rel_ign": [i * 100 for i in best_re_ign], 
        tag + "_evi": [i * 100 for i in best_evi],
    }
    scores = {
        "dev_F1": best_re[-1] * 100,
        "dev_evi_F1": best_evi[-1] * 100,
        "dev_F1_ign": best_re_ign[-1] * 100,
    }

    if args.save_attn:
        
        attns_path = os.path.join(args.load_path, f"{os.path.splitext(args.test_file)[0]}.attns")        
        print(f"saving attentions into {attns_path} ...")
        with open(attns_path, "wb") as f:
            pickle.dump(attns, f)

    return scores, output, official_results, results

def dump_to_file(offi:list, offi_path: str, scores: list, score_path: str, results: list = [], res_path: str = "", thresh: float = None):
    '''
    dump scores and (top-k) predictions to file.
    
    '''
    print(f"saving official predictions into {offi_path} ...")
    json.dump(offi, open(offi_path, "w"))
    
    print(f"saving evaluations into {score_path} ...")
    headers = ["precision", "recall", "F1"]
    scores_pd = pd.DataFrame.from_dict(scores, orient="index", columns = headers)
    print(scores_pd)
    scores_pd.to_csv(score_path, sep='\t')

    if len(results) != 0:
        assert res_path != ""
        print(f"saving topk results into {res_path} ...")
        json.dump(results, open(res_path, "w"))
    
    if thresh is not None:
        thresh_path = os.path.join(os.path.dirname(offi_path), "thresh")
        if not os.path.exists(thresh_path):
            print(f"saving threshold into {thresh_path} ...")
            json.dump(thresh, open(thresh_path, "w"))        

    return


def main():
    
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    args = normalize_model_switches(args)
    if args.run_mode == "acceptance":
        args.max_train_batches = 1
    print_model_config(args)
    if args.print_config_only:
        return
        
    # [BUG-FIX-3] disable wandb online sync to avoid needing API key
    import os as _os
    _os.environ.setdefault("WANDB_MODE", "offline")
    wandb.init(project="project", name="name")

    # create directory to save checkpoints and predicted files
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    save_path_ = os.path.join(args.save_path, args.experiment_name, run_timestamp)
    srr_adjacency_init = prepare_srr_adjacency_init(args, save_path_)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = "eager"
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )
    model_kwargs = {
        "from_tf": bool(".ckpt" in args.model_name_or_path),
        "config": config,
    }
    try:
        model = AutoModel.from_pretrained(
            args.model_name_or_path,
            attn_implementation="eager",
            **model_kwargs,
        )
    except TypeError:
        model = AutoModel.from_pretrained(
            args.model_name_or_path,
            **model_kwargs,
        )

    config.transformer_type = args.transformer_type

    set_seed(args)
    
    read = read_docred    
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id

    model = HDERModel(config, model, tokenizer,
                    num_labels=args.num_labels,
                    max_sent_num=args.max_sent_num, 
                    evi_thresh=args.evi_thresh,
                    use_hae=args.use_hae,
                    use_cross_layer_attention=args.use_cross_layer_attention,
                    global_context_mode=args.global_context_mode,
                    use_srr=args.use_srr,
                    srr_steps=args.srr_steps,
                    srr_init=args.srr_init,
                    srr_adjacency_init=srr_adjacency_init,
                    global_mean_path=args.global_mean_path,
                    max_seq_length=args.max_seq_length,
                    use_sentence_position_encoding=args.use_sentence_position_encoding,
                    paragraph_strategy=args.paragraph_strategy,
                    sentences_per_paragraph=args.sentences_per_paragraph,
                    use_span_boundary_head=args.use_span_boundary_head,
                    span_detach_backbone=args.span_detach_backbone,
                    srr_token_attention_scale=args.srr_token_attention_scale)
    model.to(args.device)
    create_directory(save_path_)
    args.save_path = save_path_
    args.effective_batch_size = args.train_batch_size * args.gradient_accumulation_steps * max(1, args.n_gpu)
    print_model_config(args)
    validation = validate_experiment_configuration(args, model)
    save_json(os.path.join(args.save_path, "args.json"), vars(args))
    save_json(os.path.join(args.save_path, "environment.json"), get_environment_info())
    save_json(os.path.join(args.save_path, "parameter_counts.json"), count_parameters(model))
    save_json(os.path.join(args.save_path, "config_validation.json"), validation)

    if args.load_path != "": # load model from existing checkpoint

        model_path = os.path.join(args.load_path, args.checkpoint_name)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        print(f"LOADING_CHECKPOINT: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))

    if args.do_train:  # Training
        
        collect_srr_entity_types = args.use_srr

        train_file = os.path.join(args.data_dir, args.train_file)
        dev_file = os.path.join(args.data_dir, args.dev_file)

        preprocess_start = time.time()
        train_features = read(
            train_file,
            tokenizer,
            transformer_type=args.transformer_type,
            max_seq_length=args.max_seq_length,
            teacher_sig_path=args.teacher_sig_path,
            collect_srr_entity_types=collect_srr_entity_types,
            entity_type_conflict_policy=args.entity_type_conflict_policy,
            entity_type_conflict_log_dir=args.save_path,
        )
        train_preprocess_seconds = time.time() - preprocess_start
        dev_preprocess_start = time.time()
        dev_features = read(
            dev_file,
            tokenizer,
            transformer_type=args.transformer_type,
            max_seq_length=args.max_seq_length,
            collect_srr_entity_types=collect_srr_entity_types,
            entity_type_conflict_policy=args.entity_type_conflict_policy,
            entity_type_conflict_log_dir=args.save_path,
        )
        dev_preprocess_seconds = time.time() - dev_preprocess_start
        save_json(
            os.path.join(args.save_path, "data_loading.json"),
            {
                "train_file": os.path.abspath(train_file),
                "dev_file": os.path.abspath(dev_file),
                "train_document_count": len(train_features),
                "dev_document_count": len(dev_features),
                "train_preprocess_seconds": train_preprocess_seconds,
                "dev_preprocess_seconds": dev_preprocess_seconds,
                "collect_srr_entity_types": collect_srr_entity_types,
            },
        )

        if args.dry_run:
            print("DRY_RUN_READY: model, tokenizer, train data, and dev data loaded successfully.")
            print(f"train_features={len(train_features)}, dev_features={len(dev_features)}, device={args.device}, n_gpu={args.n_gpu}")
            print_model_config(args)
            return

        train(args, model, train_features, dev_features)

    else:  # Testing

        basename = os.path.splitext(args.test_file)[0]
        test_file = os.path.join(args.data_dir, args.test_file)
        collect_srr_entity_types = args.use_srr
        
        test_features = read(
            test_file,
            tokenizer,
            transformer_type=args.transformer_type,
            max_seq_length=args.max_seq_length,
            collect_srr_entity_types=collect_srr_entity_types,
            entity_type_conflict_policy=args.entity_type_conflict_policy,
            entity_type_conflict_log_dir=args.load_path or args.results_path,
        )
        
        if args.eval_mode != "fushion":

            test_scores, test_output, official_results, results = evaluate(args, model, test_features, tag="test")   
            wandb.log(test_scores)

            offi_path = os.path.join(args.load_path, args.pred_file)
            score_path = os.path.join(args.load_path, f"{basename}_scores.csv")
            res_path = os.path.join(args.load_path, f"topk_{args.pred_file}")

            dump_to_file(official_results, offi_path, test_output, score_path, results, res_path)          

        else: # inference stage fusion

            results = json.load(open(os.path.join(args.results_path, f"topk_{args.pred_file}")))

            # formulate pseudo documents from top-k (k=num_labels in arguments) predictions
            pseudo_test_features = read(
                test_file,
                tokenizer,
                max_seq_length=args.max_seq_length,
                single_results=results,
                collect_srr_entity_types=collect_srr_entity_types,
                entity_type_conflict_policy=args.entity_type_conflict_policy,
                entity_type_conflict_log_dir=args.results_path,
                )
        
            pseudo_test_scores, pseudo_output, pseudo_official_results, pseudo_results = evaluate(args, model, pseudo_test_features, tag="test") 

            if 'thresh' in os.listdir(args.results_path):
                with open(os.path.join(args.results_path, "thresh")) as f:
                    thresh = json.load(f)
                print(f"Threshold loaded from file: {thresh}")
            else:
                thresh = None
            
            merged_offi, thresh = merge_results(results, pseudo_results, test_features, thresh)
            merged_re, merged_evi, merged_re_ign, _ = official_evaluate(merged_offi, args.data_dir, args.train_file, args.test_file)
            
            tag = args.test_file.split('.')[0]
            merged_output = {
                tag + "_rel": [i * 100 for i in merged_re],
                tag + "_rel_ign": [i * 100 for i in merged_re_ign], 
                tag + "_evi": [i * 100 for i in merged_evi],
            }
            
            wandb.log({"dev_F1": merged_re[-1] * 100, "dev_evi_F1": merged_evi[-1] * 100, "dev_F1_ign": merged_re_ign[-1] * 100})

            offi_path = os.path.join(args.results_path, f"fused_{args.pred_file}")
            score_path = os.path.join(args.results_path, f"{basename}_fused_scores.csv")
            dump_to_file(merged_offi, offi_path, merged_output, score_path, thresh = thresh)


if __name__ == "__main__":
    main()
