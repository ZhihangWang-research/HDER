import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected: true/false")


def add_args(parser):
    parser.add_argument("--experiment_name", default="debug", type=str,
                        help="Experiment name used for output/<experiment_name>/<timestamp>.")
    parser.add_argument("--run_mode", default="formal", choices=["acceptance", "formal"], type=str,
                        help="acceptance runs minimal smoke checks; formal runs full training/evaluation.")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="roberta", type=str)
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str)
    parser.add_argument("--display_name", default=None, type=str)
    
    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="", type=str)
    parser.add_argument("--pred_file", default="results.json", type=str)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)
    parser.add_argument("--checkpoint_name", default="best.ckpt", type=str,
                        help="Checkpoint filename loaded from --load_path during evaluation. Defaults to best.ckpt, matching validation-based model selection.")
    parser.add_argument("--results_path", default="", type=str)
    parser.add_argument("--teacher_sig_path", default="", type=str)
    parser.add_argument("--save_attn", action="store_true", help="Whether store the evidence distribution or not")
    parser.add_argument("--dry_run", action="store_true",
                        help="Load config/model/tokenizer/data and then exit before training.")
    parser.add_argument("--print_config_only", action="store_true",
                        help="Normalize model switches, print the final config, and exit before loading data or models.")
    parser.add_argument("--max_train_batches", default=-1, type=int,
                        help="Stop after this many optimizer steps. Use 1 or 20 for cloud smoke tests.")
    parser.add_argument("--use_hae", default=False, type=str2bool,
                        help="Enable the HDER hierarchy-aware encoder.")
    parser.add_argument("--use_cross_layer_attention", default=False, type=str2bool,
                        help="Enable top-down cross-layer attention inside HAE. Requires --use_hae true in later stages.")
    parser.add_argument("--global_context_mode", default="dynamic", choices=["dynamic", "fixed_mean"], type=str,
                        help="Document-level context source for HAE ablations in later stages.")
    parser.add_argument("--use_srr", default=False, type=str2bool,
                        help="Enable the HDER structured relational reasoning module.")
    parser.add_argument("--srr_steps", default=2, type=int,
                        help="Number of type-level SRR message passing steps.")
    parser.add_argument("--srr_token_attention_scale", default=0.05, type=float,
                        help="Residual scale for token-level type-aware attention inside SRR.")
    parser.add_argument("--use_span_boundary_head", default=True, type=str2bool,
                        help="Enable the typed start/end entity-span boundary decoder.")
    parser.add_argument("--span_aux_weight", default=0.01, type=float,
                        help="Weight of the auxiliary typed span-boundary loss.")
    parser.add_argument("--span_detach_backbone", default=True, type=str2bool,
                        help="Detach HAE/SRR features for the auxiliary span loss to keep the relation path stable.")
    parser.add_argument("--srr_init", default="learned", choices=["learned", "cooccurrence"], type=str,
                        help="SRR type adjacency initialization method.")
    parser.add_argument("--srr_cooccurrence_path", default="", type=str,
                        help="Path to a saved 6x6 SRR type cooccurrence initialization JSON.")
    parser.add_argument("--aux_graph_path", default="./auxiliary_graph_data/graph_statistics.json", type=str,
                        help=("Optional local auxiliary type-interaction graph statistics used only to initialize SRR. "
                              "The HDER/SRR implementation remains runnable when this file is absent."))
    parser.add_argument("--entity_type_conflict_policy", default="majority_first",
                        choices=["strict", "majority_first"], type=str,
                        help="How to resolve inconsistent mention types inside a DocRED entity cluster for SRR.")
    parser.add_argument("--use_sentence_position_encoding", default=True, type=str2bool,
                        help="Enable learnable sentence position encoding inside HAE.")
    parser.add_argument("--paragraph_strategy", default="sentence_window",
                        choices=["single_document", "sentence_window"], type=str,
                        help="Paragraph construction strategy for HAE. DocRED has no natural paragraph boundary.")
    parser.add_argument("--sentences_per_paragraph", default=4, type=int,
                        help="Number of consecutive sentences per pseudo paragraph when using sentence_window.")
    parser.add_argument("--global_mean_path", default="", type=str,
                        help="Path to a fixed global mean document vector for --global_context_mode fixed_mean.")
    parser.add_argument("--max_mean_batches", default=-1, type=int,
                        help="Limit batches when estimating fixed global mean; -1 uses the full train set.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--eval_mode", default="single", type=str,
                        choices=["single", "fushion"], 
                        help="Single-pass evaluation or evaluation with inference-stage fusion.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--max_sent_num", default=25, type=int,
                        help="Max number of sentences in each document.")
    parser.add_argument("--evi_thresh", default=0.2, type=float,
                        help="Evidence Threshold. ")
    parser.add_argument("--evi_lambda", default=0.03, type=float,
                        help="Weight of relation-agnostic evidence loss during training. ")
    parser.add_argument("--attn_lambda", default=1.0, type=float,
                        help="Weight of knowledge distillation loss for attentions during training. ")
    parser.add_argument("--lr_transformer", default=3e-5, type=float,
                        help="The initial learning rate for transformer.")
    parser.add_argument("--lr_added", default=1e-4, type=float,
                        help="The initial learning rate for added modules.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=20.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")

    return parser
