"""Train the SIFT teacher using one or two source cities."""

import json
from pathlib import Path
import numpy as np
import torch
from torch import optim
import utils
from args import apply_smoke, model_config, stage1_parser
from model import SourceGrammarTeacher


def weight_config(args):
    return {
        "poi_hidden": args.w_poi_hidden,
        "poi": args.w_poi,
        "dynamic": args.w_dynamic,
        "grammar": args.w_grammar,
        "residual": args.w_residual,
        "directional_anchor": args.w_directional_anchor,
        "balance": args.w_balance,
        "sharpness": args.w_sharpness,
        "variance": args.w_variance,
        "covariance": args.w_covariance,
    }


def weighted_loss(terms, weights, included):
    total = next(iter(terms.values())).new_tensor(0.0)
    for name in included:
        total = total + float(weights[name]) * terms[name]
    return total


def set_trainable(model, mode):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if mode == "poi":
        modules = [model.poi_encoder, model.poi_hidden_decoder]
        if model.city_context_adapter is not None:
            modules.append(model.city_context_adapter)
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    elif mode == "grammar":
        model.shared_grammar.requires_grad_(True)
    elif mode == "representation":
        for name, parameter in model.named_parameters():
            if name != "shared_grammar":
                parameter.requires_grad_(True)
    else:
        raise ValueError("Unknown training mode {}".format(mode))


def forward_city(model, data, direction_enabled):
    return model(
        data["poi_features"],
        data["poi_composition"],
        data["a_out"],
        data["a_in"],
        data["dynamic"],
        data["residual"],
        direction_enabled=direction_enabled,
        city=data["city"],
        city_context=data["poi_city_context"],
    )


def optimize_cities(
    model, source_data, optimizer, weights, included, direction_enabled, grad_clip
):
    optimizer.zero_grad()
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    detail = {}
    for city, data in source_data.items():
        output = forward_city(model, data, direction_enabled)
        terms = model.loss_terms(
            output, data["poi_composition"], data["dynamic"], data["residual"]
        )
        city_loss = weighted_loss(terms, weights, included)
        total = total + city_loss / len(source_data)
        detail[city] = {
            name: float(value.detach().cpu()) for (name, value) in terms.items()
        }
    total.backward()
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        grad_clip,
    )
    optimizer.step()
    return (float(total.detach().cpu()), detail)


def model_diagnostics(model, source_data):
    model.eval()
    embeddings = []
    roles = {"semantic": [], "outgoing": [], "incoming": []}
    with torch.no_grad():
        for data in source_data.values():
            output = forward_city(model, data, True)
            embeddings.append(output["embedding"].cpu().numpy())
            roles["semantic"].append(output["q_semantic"].cpu().numpy())
            roles["outgoing"].append(output["q_out"].cpu().numpy())
            roles["incoming"].append(output["q_in"].cpu().numpy())
    return utils.representation_diagnostics(
        np.concatenate(embeddings, axis=0),
        {name: np.concatenate(values, axis=0) for (name, values) in roles.items()},
        model.grammar().detach().cpu().numpy(),
    )


def evaluation_loss(model, source_data, weights):
    included = tuple(weights.keys())
    model.eval()
    values = []
    with torch.no_grad():
        for data in source_data.values():
            output = forward_city(model, data, True)
            terms = model.loss_terms(
                output, data["poi_composition"], data["dynamic"], data["residual"]
            )
            values.append(float(weighted_loss(terms, weights, included).cpu()))
    return float(np.mean(values))


def save_checkpoint(
    path,
    model,
    config,
    initializers,
    poi_statistics,
    poi_alignment,
    args,
    weights,
    epoch,
    score,
    diagnostics,
):
    checkpoint = {
        "format": "SIFT-stage1",
        "model_config": config,
        "model_state": {
            key: value.detach().cpu() for (key, value) in model.state_dict().items()
        },
        "poi_initializers": initializers,
        "poi_statistics": poi_statistics,
        "poi_alignment": poi_alignment,
        "source_cities": list(args.source_cities),
        "target_city": args.target_city,
        "epoch": int(epoch),
        "selection_score": float(score),
        "diagnostics": diagnostics,
        "loss_weights": weights,
        "phase_epochs": {
            "poi_warmup": int(args.poi_warmup_epochs),
            "grammar": int(args.grammar_epochs),
            "joint": int(args.joint_epochs),
        },
        "data_access": {
            "source": ["poi_f.npy", "source_matrix_norm.npy", "source_sum.npy"],
            "target": [],
        },
    }
    checkpoint["parameters"] = vars(args).copy()
    checkpoint["decomposition"] = {
        "shared_at_target": True,
        "source_city_deviations_at_target": False,
    }
    torch.save(checkpoint, path)


def run(args):
    args = apply_smoke(args)
    if min(args.poi_warmup_epochs, args.grammar_epochs) < 0 or args.joint_epochs < 1:
        raise ValueError(
            "Warmup/grammar epochs must be nonnegative; joint_epochs must be positive"
        )
    utils.set_seed(args.seed)
    device = utils.resolve_device(args.device)
    if args.target_city in args.source_cities:
        raise ValueError("target_city must not appear in source_cities")
    poi_alignment = utils.fit_poi_alignment(
        args.source_cities,
        Path(args.data_root),
        mode=args.poi_alignment,
        output_dim=utils.POI_CATEGORY_DIM,
    )
    args.poi_category_dim = int(poi_alignment["output_dim"])
    poi_statistics = utils.compute_poi_statistics(
        args.source_cities, Path(args.data_root), poi_alignment
    )
    source_data = {
        city: utils.load_source_city(
            city, Path(args.data_root), device, poi_statistics, poi_alignment
        )
        for city in args.source_cities
    }
    initializers = utils.compute_poi_initializers(
        source_data, args.role_count, args.seed
    )
    config = model_config(args)
    model = SourceGrammarTeacher(config, initializers).to(device)
    weights = weight_config(args)
    save_root = Path(args.checkpoint_dir)
    result_root = Path(args.result_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_root / "stage1_{}_to_{}.pt".format(
        "_".join(args.source_cities), args.target_city
    )
    print(
        "[Stage1] sources={} target={} device={} roles={} embedding={} poi={}({}) params={:,}".format(
            args.source_cities,
            args.target_city,
            device,
            args.role_count,
            args.embedding_dim,
            poi_alignment["mode"],
            poi_alignment["output_dim"],
            sum((parameter.numel() for parameter in model.parameters())),
        ),
        flush=True,
    )
    set_trainable(model, "poi")
    poi_optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.poi_learning_rate,
        weight_decay=args.weight_decay,
    )
    for epoch in range(args.poi_warmup_epochs):
        model.train()
        (loss, _) = optimize_cities(
            model,
            source_data,
            poi_optimizer,
            weights,
            ("poi_hidden",),
            False,
            args.grad_clip,
        )
        if epoch % args.log_every == 0 or epoch + 1 == args.poi_warmup_epochs:
            print("[POI {:4d}] loss={:.6f}".format(epoch, loss), flush=True)
    set_trainable(model, "grammar")
    grammar_optimizer = optim.Adam(
        [model.shared_grammar],
        lr=args.grammar_learning_rate,
        weight_decay=args.weight_decay,
    )
    for epoch in range(args.grammar_epochs):
        model.train()
        (loss, _) = optimize_cities(
            model,
            source_data,
            grammar_optimizer,
            weights,
            ("grammar", "residual"),
            False,
            args.grad_clip,
        )
        if epoch % args.log_every == 0 or epoch + 1 == args.grammar_epochs:
            print("[Grammar {:4d}] loss={:.6f}".format(epoch, loss), flush=True)
    set_trainable(model, "representation")
    representation_parameters = [
        parameter
        for (name, parameter) in model.named_parameters()
        if name != "shared_grammar"
    ]
    representation_optimizer = optim.Adam(
        representation_parameters,
        lr=args.joint_learning_rate,
        weight_decay=args.weight_decay,
    )
    joint_grammar_optimizer = optim.Adam(
        [model.shared_grammar],
        lr=args.grammar_learning_rate * 0.1,
        weight_decay=args.weight_decay,
    )
    representation_terms = (
        "poi_hidden",
        "poi",
        "dynamic",
        "residual",
        "directional_anchor",
        "balance",
        "sharpness",
        "variance",
        "covariance",
    )
    best_score = float("inf")
    best_epoch = -1
    best_diagnostics = None
    for epoch in range(args.joint_epochs):
        set_trainable(model, "representation")
        model.train()
        (representation_loss, _) = optimize_cities(
            model,
            source_data,
            representation_optimizer,
            weights,
            representation_terms,
            True,
            args.grad_clip,
        )
        set_trainable(model, "grammar")
        model.train()
        (grammar_loss, _) = optimize_cities(
            model,
            source_data,
            joint_grammar_optimizer,
            weights,
            ("grammar", "residual"),
            True,
            args.grad_clip,
        )
        diagnostics = model_diagnostics(model, source_data)
        base_score = evaluation_loss(model, source_data, weights)
        rank_shortfall = max(0.0, 10.0 - diagnostics["embedding_effective_rank"]) / 10.0
        entropy_excess = max(
            0.0,
            max(diagnostics["outgoing_entropy"], diagnostics["incoming_entropy"]) - 0.9,
        )
        score = base_score + 0.1 * rank_shortfall + 0.5 * entropy_excess
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_diagnostics = diagnostics
            save_checkpoint(
                checkpoint_path,
                model,
                config,
                initializers,
                poi_statistics,
                poi_alignment,
                args,
                weights,
                epoch,
                score,
                diagnostics,
            )
        if epoch % args.log_every == 0 or epoch + 1 == args.joint_epochs:
            print(
                "[Joint {:4d}] repr={:.6f} grammar={:.6f} score={:.6f} rank={:.2f} entropy={:.3f}/{:.3f} best={:.6f}@{}".format(
                    epoch,
                    representation_loss,
                    grammar_loss,
                    score,
                    diagnostics["embedding_effective_rank"],
                    diagnostics["outgoing_entropy"],
                    diagnostics["incoming_entropy"],
                    best_score,
                    best_epoch,
                ),
                flush=True,
            )
    result = {
        "default": -best_score,
        "stage": "1",
        "source_cities": list(args.source_cities),
        "target_city": args.target_city,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "checkpoint": str(checkpoint_path),
        "model_config": config,
        "poi_alignment": utils.poi_alignment_summary(poi_alignment),
        "poi_alignment_diagnostics": {
            city: data["poi_alignment_diagnostics"]
            for (city, data) in source_data.items()
        },
        "diagnostics": best_diagnostics,
        "phase_epochs": {
            "poi_warmup": args.poi_warmup_epochs,
            "grammar": args.grammar_epochs,
            "joint": args.joint_epochs,
        },
        "data_access": {
            "source": ["poi_f.npy", "source_matrix_norm.npy", "source_sum.npy"],
            "target": [],
        },
    }
    result_path = result_root / "stage1_{}_to_{}.json".format(
        "_".join(args.source_cities), args.target_city
    )
    utils.atomic_json(result_path, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    from args import parse_config

    run(parse_config(stage1_parser(), "stage1"))


if __name__ == "__main__":
    main()
