import argparse
import os
import warnings

import torch


FIXED_MEAN_STATISTIC_DEFINITION = "mean of final-layer CLS hidden state over train_annotated documents"
EXPECTED_FIXED_MEAN_DOCUMENT_COUNT = 3053
EXPECTED_FIXED_MEAN_HIDDEN_SIZE = 1024
EXPECTED_FIXED_MEAN_SOURCE_BASENAME = "train_annotated.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--hidden_size", default=1024, type=int)
    parser.add_argument("--max_seq_length", default=1024, type=int)
    return parser.parse_args()


def _load_fixed_mean_payload(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"fixed_mean file not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "mean" not in payload or "metadata" not in payload:
        raise ValueError("fixed_mean file must contain {'mean': Tensor, 'metadata': dict}.")
    return payload


def _looks_like_non_roberta_large_name(name):
    name = str(name or "").lower().replace("\\", "/")
    if not name:
        return False
    if "roberta-base" in name or "bert-base" in name or "bert-large" in name:
        return True
    if "roberta" not in name and ("bert" in name or "deberta" in name or "electra" in name):
        return True
    return False


def _warn_if_model_name_differs(metadata_name, current_model_name):
    if not current_model_name:
        return
    metadata_name = str(metadata_name or "")
    current_model_name = str(current_model_name or "")
    if metadata_name and os.path.normcase(metadata_name) != os.path.normcase(current_model_name):
        warnings.warn(
            "fixed_mean model_name_or_path differs from current model_name_or_path, "
            "but RoBERTa-large-compatible metadata/config checks passed: "
            f"fixed_mean={metadata_name}, current={current_model_name}",
            RuntimeWarning,
        )


def _validate_roberta_large_compatibility(metadata, current_config=None, current_model_name=None):
    metadata_name = metadata.get("model_name_or_path", "")
    if _looks_like_non_roberta_large_name(metadata_name):
        raise ValueError(f"fixed_mean model_name_or_path is not RoBERTa-large compatible: {metadata_name}")
    if str(metadata.get("transformer_type", "")).lower() != "roberta":
        raise ValueError(f"transformer_type must be roberta, got {metadata.get('transformer_type')}")
    if int(metadata.get("hidden_size", -1)) != EXPECTED_FIXED_MEAN_HIDDEN_SIZE:
        raise ValueError(f"hidden_size must be 1024, got {metadata.get('hidden_size')}")

    model_type = metadata.get("model_type")
    if model_type is not None and str(model_type).lower() != "roberta":
        raise ValueError(f"metadata model_type must be roberta, got {model_type}")
    num_layers = metadata.get("num_hidden_layers")
    if num_layers is not None and int(num_layers) != 24:
        raise ValueError(f"RoBERTa-large must have 24 hidden layers, got {num_layers}")
    num_heads = metadata.get("num_attention_heads")
    if num_heads is not None and int(num_heads) != 16:
        raise ValueError(f"RoBERTa-large must have 16 attention heads, got {num_heads}")

    if current_config is not None:
        if int(getattr(current_config, "hidden_size", -1)) != EXPECTED_FIXED_MEAN_HIDDEN_SIZE:
            raise ValueError(f"current model hidden_size is not 1024: {getattr(current_config, 'hidden_size', None)}")
        current_model_type = getattr(current_config, "model_type", None)
        if current_model_type is not None and str(current_model_type).lower() != "roberta":
            raise ValueError(f"current model_type must be roberta, got {current_model_type}")
        current_layers = getattr(current_config, "num_hidden_layers", None)
        if current_layers is not None and int(current_layers) != 24:
            raise ValueError(f"current RoBERTa-large-compatible config must have 24 layers, got {current_layers}")
        current_heads = getattr(current_config, "num_attention_heads", None)
        if current_heads is not None and int(current_heads) != 16:
            raise ValueError(f"current RoBERTa-large-compatible config must have 16 heads, got {current_heads}")
    _warn_if_model_name_differs(metadata_name, current_model_name)


def validate_fixed_mean_payload(payload, hidden_size=1024, max_seq_length=1024,
                                current_config=None, current_model_name=None):
    mean = payload["mean"].detach().float().view(-1)
    metadata = payload["metadata"]
    if int(hidden_size) != EXPECTED_FIXED_MEAN_HIDDEN_SIZE:
        raise ValueError(f"fixed_mean expected hidden_size argument must be 1024, got {hidden_size}")
    if tuple(mean.shape) != (EXPECTED_FIXED_MEAN_HIDDEN_SIZE,):
        raise ValueError(f"mean shape mismatch: expected ({hidden_size},), got {tuple(mean.shape)}")
    if not torch.isfinite(mean).all():
        raise ValueError("fixed_mean contains NaN or Inf.")
    mean.requires_grad_(False)
    if metadata.get("source_split") != "train":
        raise ValueError(f"source_split must be train, got {metadata.get('source_split')}")
    source_file = str(metadata.get("source_file", ""))
    if os.path.basename(source_file) != EXPECTED_FIXED_MEAN_SOURCE_BASENAME:
        raise ValueError(
            f"source_file basename must be {EXPECTED_FIXED_MEAN_SOURCE_BASENAME}, got {metadata.get('source_file')}"
        )
    if int(metadata.get("document_count", -1)) != EXPECTED_FIXED_MEAN_DOCUMENT_COUNT:
        raise ValueError(
            f"document_count must be {EXPECTED_FIXED_MEAN_DOCUMENT_COUNT}, got {metadata.get('document_count')}"
        )
    if metadata.get("statistic_definition") != FIXED_MEAN_STATISTIC_DEFINITION:
        raise ValueError(
            "statistic_definition must be exactly "
            f"'{FIXED_MEAN_STATISTIC_DEFINITION}', got {metadata.get('statistic_definition')}"
        )
    _validate_roberta_large_compatibility(metadata, current_config=current_config, current_model_name=current_model_name)
    if int(metadata.get("max_seq_length", -1)) != int(max_seq_length):
        raise ValueError(
            f"max_seq_length mismatch: {metadata.get('max_seq_length')} vs {max_seq_length}"
        )
    return {"mean": mean, "metadata": metadata}


def validate_fixed_mean_file(path, hidden_size=1024, max_seq_length=1024,
                             current_config=None, current_model_name=None):
    payload = _load_fixed_mean_payload(path)
    result = validate_fixed_mean_payload(
        payload,
        hidden_size=hidden_size,
        max_seq_length=max_seq_length,
        current_config=current_config,
        current_model_name=current_model_name,
    )
    metadata = result["metadata"]
    print("FIXED_MEAN_VALIDATION_PASS")
    print(f"path={path}")
    print(f"document_count={metadata.get('document_count')}")
    print(f"source_file={metadata.get('source_file')}")
    return result


def main():
    args = parse_args()
    validate_fixed_mean_file(args.path, args.hidden_size, args.max_seq_length)


if __name__ == "__main__":
    main()
