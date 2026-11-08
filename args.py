"""Command-line arguments and YAML configuration for SIFT."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def add_common(parser):
    parser.add_argument("--source_cities", nargs="+", default=["bj", "cd"])
    parser.add_argument("--target_city", choices=("bj", "cd", "xa"), default="xa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--result_dir", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--role_count", type=int, default=8)
    parser.add_argument("--role_dim", type=int, default=96)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--embedding_dim", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.06)
    parser.add_argument("--direction_scale", type=float, default=0.75)
    parser.add_argument("--grammar_attention_temperature", type=float, default=1.0)
    parser.add_argument("--mobility_prior_scale", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--config", help="YAML configuration file")
    parser.add_argument(
        "--data_root",
        default=str(ROOT),
        help="Directory containing dataset/ and tasks/",
    )
    parser.add_argument(
        "--output_dir",
        default=str(ROOT.parent / "sift_runs"),
        help="Save generated files outside the code directory",
    )
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    parser.set_defaults(
        poi_alignment="frequency_rank",
        poi_slots=162,
        poi_alignment_temperature=0.05,
        poi_alignment_balance=0.5,
        poi_alignment_iterations=30,
        poi_city_adapter=False,
        poi_city_adapter_hidden=32,
        poi_city_adapter_scale=0.5,
    )
    return parser


def stage1_parser():
    parser = add_common(
        argparse.ArgumentParser(description="SIFT Stage-I source branch training")
    )
    parser.add_argument("--poi_warmup_epochs", type=int, default=12)
    parser.add_argument("--grammar_epochs", type=int, default=24)
    parser.add_argument("--joint_epochs", type=int, default=36)
    parser.add_argument("--poi_learning_rate", type=float, default=0.0005)
    parser.add_argument("--grammar_learning_rate", type=float, default=0.003)
    parser.add_argument("--joint_learning_rate", type=float, default=0.0002)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--grad_clip", type=float, default=2.0)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--w_poi_hidden", type=float, default=0.5)
    parser.add_argument("--w_poi", type=float, default=0.2)
    parser.add_argument("--w_dynamic", type=float, default=0.2)
    parser.add_argument("--w_grammar", type=float, default=1.0)
    parser.add_argument("--w_residual", type=float, default=0.5)
    parser.add_argument("--w_directional_anchor", type=float, default=0.2)
    parser.add_argument("--w_balance", type=float, default=0.05)
    parser.add_argument("--w_sharpness", type=float, default=0.05)
    parser.add_argument("--w_variance", type=float, default=0.1)
    parser.add_argument("--w_covariance", type=float, default=0.001)

    parser.add_argument("--city_delta_rank", type=int, default=1)
    parser.add_argument("--city_delta_scale", type=float, default=0.15)
    parser.add_argument("--city_delta_penalty", type=float, default=0.2)
    parser.add_argument("--city_center_penalty", type=float, default=0.5)
    parser.set_defaults(city_decomposition="bounded")
    return parser


def stage2_parser():
    parser = add_common(
        argparse.ArgumentParser(description="SIFT Stage-II deployment branch training")
    )
    parser.add_argument("--stage1_checkpoint", required=False)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=0.00085)
    parser.add_argument("--min_learning_rate", type=float, default=0.00013)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--grad_clip", type=float, default=2.0)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument(
        "--selection_metric",
        choices=("validation_loss",),
        default="validation_loss",
        help="Save the minimum source-city validation objective; no downstream labels are used.",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "structured_only"),
        default="full",
        help="full includes embedding distillation; structured_only uses structural supervision.",
    )
    parser.add_argument(
        "--output_variant",
        choices=(
            "full",
            "local",
            "local_role",
            "local_grammar",
            "grammar",
            "raw_poi",
            "raw_poi_grammar",
            "raw_poi_role_grammar",
        ),
        default="local_role",
    )
    parser.add_argument("--w_role", type=float, default=1.0)
    parser.add_argument("--w_dynamic", type=float, default=0.5)
    parser.add_argument("--w_confidence", type=float, default=0.05)
    parser.add_argument("--w_context", type=float, default=1.0)
    parser.add_argument("--w_grammar_embedding", type=float, default=0.5)
    parser.add_argument("--w_embedding", type=float, default=0.1)
    parser.add_argument("--w_poi", type=float, default=0.2)
    parser.add_argument("--w_sharpness", type=float, default=0.03)
    parser.add_argument("--w_balance", type=float, default=0.03)
    parser.add_argument("--w_variance", type=float, default=0.1)
    parser.add_argument("--w_covariance", type=float, default=0.001)
    return parser


def model_config(args):
    poi_category_dim = int(getattr(args, "poi_category_dim", 162))
    config = {
        "role_count": int(args.role_count),
        "role_dim": int(args.role_dim),
        "hidden_dim": int(args.hidden_dim),
        "embedding_dim": int(args.embedding_dim),
        "temperature": float(args.temperature),
        "direction_scale": float(args.direction_scale),
        "grammar_attention_temperature": float(args.grammar_attention_temperature),
        "mobility_prior_scale": float(args.mobility_prior_scale),
        "dropout": float(args.dropout),
        "poi_category_dim": poi_category_dim,
        "poi_feature_dim": poi_category_dim + 1,
        "poi_city_adapter": bool(args.poi_city_adapter),
        "poi_city_adapter_hidden": int(args.poi_city_adapter_hidden),
        "poi_city_adapter_scale": float(args.poi_city_adapter_scale),
    }
    config.update(
        source_cities=list(args.source_cities),
        city_decomposition="bounded",
        city_delta_rank=args.city_delta_rank,
        city_delta_scale=args.city_delta_scale,
        city_delta_penalty=args.city_delta_penalty,
        city_center_penalty=args.city_center_penalty,
    )
    return config


def apply_smoke(args):
    if args.smoke:
        if hasattr(args, "epochs"):
            args.epochs = min(args.epochs, 3)
        if hasattr(args, "poi_warmup_epochs"):
            args.poi_warmup_epochs = min(args.poi_warmup_epochs, 2)
            args.grammar_epochs = min(args.grammar_epochs, 2)
            args.joint_epochs = min(args.joint_epochs, 2)
        args.log_every = 1
    return args


def parse_config(parser, stage, argv=None):
    """Load a stage's YAML settings, then apply explicit command-line overrides."""
    import sys
    import torch
    import yaml

    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args(argv)
    if known.config:
        with open(known.config, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict) or stage not in config:
            parser.error(
                "Configuration must contain the requested stage1/stage2 mapping"
            )
        if not isinstance(config[stage], dict):
            parser.error("Stage configuration must be a mapping")
        values = {
            key: config[key]
            for key in ("source_cities", "target_city")
            if key in config
        }
        values.update(config[stage])
        allowed = {action.dest for action in parser._actions} | set(parser._defaults)
        unknown = set(values) - allowed
        if unknown:
            parser.error("Unknown configuration keys: " + ", ".join(sorted(unknown)))
        for action in parser._actions:
            if action.dest in values:
                value = values[action.dest]
                items = value if isinstance(value, list) else [value]
                if action.type:
                    try:
                        items = [action.type(item) for item in items]
                    except (TypeError, ValueError) as exc:
                        parser.error(str(exc))
                if action.choices and any(item not in action.choices for item in items):
                    parser.error("Invalid value for " + action.dest)
                values[action.dest] = items if isinstance(value, list) else items[0]
        parser.set_defaults(**values)
    args = parser.parse_args(argv)
    sources = args.source_cities
    if (
        not isinstance(sources, list)
        or len(sources) not in (1, 2)
        or len(set(sources)) != len(sources)
    ):
        parser.error("source_cities must contain one or two distinct cities")
    if (
        any(city not in ("bj", "cd", "xa") for city in sources)
        or args.target_city in sources
    ):
        parser.error("Use bj/cd/xa with the target excluded from source_cities")
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    scenario = "_".join(sources) + "_to_" + args.target_city
    run_root = Path(args.output_dir).expanduser().resolve() / scenario
    if args.smoke:
        run_root = run_root / "smoke"
    stage_root = run_root / stage
    if stage == "stage2" and args.config:
        stage_root = stage_root / Path(args.config).stem
    args.checkpoint_dir = args.checkpoint_dir or str(stage_root)
    args.result_dir = args.result_dir or str(stage_root)
    args.data_root = str(Path(args.data_root).expanduser().resolve())
    if stage == "stage2" and not args.stage1_checkpoint:
        args.stage1_checkpoint = str(
            run_root / "stage1" / ("stage1_" + scenario + ".pt")
        )
    return args
