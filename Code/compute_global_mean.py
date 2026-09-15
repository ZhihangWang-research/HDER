import argparse
import datetime
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer

from long_seq import process_long_input
from prepro import read_docred
from utils import collate_fn
from validate_fixed_mean import (
    EXPECTED_FIXED_MEAN_DOCUMENT_COUNT,
    EXPECTED_FIXED_MEAN_HIDDEN_SIZE,
    FIXED_MEAN_STATISTIC_DEFINITION,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./dataset/docred")
    parser.add_argument("--train_file", default="train_annotated.json")
    parser.add_argument("--transformer_type", default="roberta")
    parser.add_argument("--model_name_or_path", default="roberta-large")
    parser.add_argument("--max_seq_length", default=1024, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    train_path = os.path.join(args.data_dir, args.train_file)
    if os.path.basename(train_path) != "train_annotated.json":
        raise ValueError("Global mean must be computed from train_annotated.json only.")

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    if int(config.hidden_size) != EXPECTED_FIXED_MEAN_HIDDEN_SIZE:
        raise ValueError(
            f"fixed_mean for the final HDER experiments must be 1024-d RoBERTa-large CLS, "
            f"got hidden_size={config.hidden_size} from {args.model_name_or_path}."
        )
    if getattr(config, "model_type", None) != "roberta":
        raise ValueError(f"fixed_mean must be generated with RoBERTa-large, got model_type={getattr(config, 'model_type', None)}")
    if int(getattr(config, "num_hidden_layers", -1)) != 24:
        raise ValueError(f"RoBERTa-large must have 24 hidden layers, got {getattr(config, 'num_hidden_layers', None)}")
    if int(getattr(config, "num_attention_heads", -1)) != 16:
        raise ValueError(f"RoBERTa-large must have 16 attention heads, got {getattr(config, 'num_attention_heads', None)}")
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = "eager"
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path, config=config)
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    config.transformer_type = args.transformer_type
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    if args.transformer_type == "roberta":
        start_tokens = [config.cls_token_id]
        end_tokens = [config.sep_token_id, config.sep_token_id]
    elif args.transformer_type == "bert":
        start_tokens = [config.cls_token_id]
        end_tokens = [config.sep_token_id]
    else:
        raise ValueError(f"Unsupported transformer_type: {args.transformer_type}")

    features = read_docred(
        train_path,
        tokenizer,
        transformer_type=args.transformer_type,
        max_seq_length=args.max_seq_length,
        collect_srr_entity_types=False,
    )
    dataloader = DataLoader(features, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    total = torch.zeros(config.hidden_size, dtype=torch.float64, device=device)
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            sequence_output, _ = process_long_input(
                model,
                input_ids,
                attention_mask,
                start_tokens,
                end_tokens,
                hidden_aggregation="last",
            )
            cls = sequence_output[:, 0, :].double()
            total += cls.sum(dim=0)
            count += cls.size(0)
    if count != EXPECTED_FIXED_MEAN_DOCUMENT_COUNT:
        raise ValueError(
            f"fixed_mean must use {EXPECTED_FIXED_MEAN_DOCUMENT_COUNT} train documents, got {count}."
        )
    mean = (total / max(count, 1)).float().cpu()
    if mean.shape != (config.hidden_size,):
        raise ValueError(f"mean shape mismatch: {tuple(mean.shape)}")
    if not torch.isfinite(mean).all():
        raise ValueError("Global mean contains NaN or Inf.")
    metadata = {
        "source_file": os.path.abspath(train_path),
        "source_split": "train",
        "document_count": count,
        "model_name_or_path": args.model_name_or_path,
        "transformer_type": args.transformer_type,
        "model_type": getattr(config, "model_type", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "max_seq_length": args.max_seq_length,
        "hidden_size": config.hidden_size,
        "dtype": str(mean.dtype),
        "generation_time": datetime.datetime.utcnow().isoformat() + "Z",
        "hidden_aggregation": "last",
        "statistic_definition": FIXED_MEAN_STATISTIC_DEFINITION,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({"mean": mean, "metadata": metadata}, args.output)
    print(f"GLOBAL_MEAN_SAVED: {args.output}")
    print(f"GLOBAL_MEAN_METADATA: {metadata}")


if __name__ == "__main__":
    main()
