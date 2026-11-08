"""Distill the source teacher into a POI-only SIFT student."""

import json
from pathlib import Path
import numpy as np
import torch
from torch import optim
import utils
from args import apply_smoke, stage2_parser
from model import POIOnlyStudent, SourceGrammarTeacher


def load_teacher(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") not in {"SIFT-stage1", "FIGUR-2026-07-27-stage1-v6"}:
        raise ValueError("Expected a SIFT Stage-I checkpoint")
    teacher = SourceGrammarTeacher(
        checkpoint["model_config"], checkpoint["poi_initializers"]
    )
    teacher.load_state_dict(checkpoint["model_state"])
    teacher.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return (checkpoint, teacher)


def weight_config(args):
    names = (
        "role",
        "dynamic",
        "confidence",
        "context",
        "grammar_embedding",
        "embedding",
        "poi",
        "sharpness",
        "balance",
        "variance",
        "covariance",
    )
    weights = {name: getattr(args, "w_" + name) for name in names}
    if args.mode == "structured_only":
        weights["embedding"] = 0.0
    return weights


def weighted_loss(terms, weights):
    total = next(iter(terms.values())).new_tensor(0.0)
    for name, value in terms.items():
        total = total + float(weights[name]) * value
    return total


def teacher_targets(teacher, data):
    with torch.no_grad():
        output = teacher(
            data["poi_features"],
            data["poi_composition"],
            data["a_out"],
            data["a_in"],
            data["dynamic"],
            data["residual"],
            direction_enabled=True,
            city=data["city"],
            city_context=data["poi_city_context"],
        )
    return {
        "embedding": output["embedding"].detach(),
        "local_embedding": output["local_embedding"].detach(),
        "role_embedding": output["role_embedding"].detach(),
        "grammar_embedding": output["grammar_embedding"].detach(),
        "q_out": output["q_out"].detach(),
        "q_in": output["q_in"].detach(),
        "outgoing_context": output["outgoing_context"].detach(),
        "incoming_context": output["incoming_context"].detach(),
        "outgoing_message": output["outgoing_message"].detach(),
        "incoming_message": output["incoming_message"].detach(),
        "grammar_reliability": output["grammar_reliability"].detach(),
        "dynamic_target": data["dynamic"].detach(),
    }


def student_diagnostics(student, cities_data):
    student.eval()
    embeddings = []
    roles = {"semantic": [], "outgoing": [], "incoming": []}
    with torch.no_grad():
        for data in cities_data.values():
            output = student(
                data["poi_features"], data["poi_composition"], data["poi_city_context"]
            )
            embeddings.append(output["embedding"].cpu().numpy())
            roles["semantic"].append(output["q_semantic"].cpu().numpy())
            roles["outgoing"].append(output["q_out"].cpu().numpy())
            roles["incoming"].append(output["q_in"].cpu().numpy())
    return utils.representation_diagnostics(
        np.concatenate(embeddings, axis=0),
        {name: np.concatenate(values, axis=0) for (name, values) in roles.items()},
        student.shared_grammar.detach().cpu().numpy(),
    )


def target_output(student, target_data):
    student.eval()
    with torch.no_grad():
        output = student(
            target_data["poi_features"],
            target_data["poi_composition"],
            target_data["poi_city_context"],
        )
    embedding = output["embedding"].cpu().numpy()
    diagnostics = utils.representation_diagnostics(
        embedding,
        {
            "semantic": output["q_semantic"].cpu().numpy(),
            "outgoing": output["q_out"].cpu().numpy(),
            "incoming": output["q_in"].cpu().numpy(),
        },
        student.shared_grammar.detach().cpu().numpy(),
    )
    return (output, embedding, diagnostics)


def run(args):
    args = apply_smoke(args)
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if not 0.0 <= args.min_learning_rate <= args.learning_rate:
        raise ValueError("Require 0 <= min_learning_rate <= learning_rate")
    utils.set_seed(args.seed)
    device = utils.resolve_device(args.device)
    (checkpoint, teacher) = load_teacher(args.stage1_checkpoint, device)
    if list(args.source_cities) != list(checkpoint["source_cities"]):
        raise ValueError("source_cities do not match Stage 1 checkpoint")
    if args.target_city != checkpoint["target_city"]:
        raise ValueError("target_city does not match Stage 1 checkpoint")
    poi_statistics = checkpoint["poi_statistics"]
    poi_alignment = utils.checkpoint_poi_alignment(checkpoint)
    args.poi_alignment = poi_alignment["mode"]
    args.poi_slots = int(poi_alignment["output_dim"])
    source_data = {
        city: utils.load_source_city(
            city, Path(args.data_root), device, poi_statistics, poi_alignment
        )
        for city in args.source_cities
    }
    target_data = utils.load_target_city(
        args.target_city,
        Path(args.data_root),
        device,
        poi_statistics,
        poi_alignment,
        load_labels=False,
    )
    if target_data["data_access"]["representation_input"] != ["poi_f.npy"]:
        raise AssertionError("Target representation accessed forbidden data")
    teacher_by_city = {
        city: teacher_targets(teacher, data) for (city, data) in source_data.items()
    }
    grammar = teacher.grammar()
    student_config = dict(checkpoint["model_config"])
    student_config.update(
        {
            "direction_scale": float(args.direction_scale),
            "grammar_attention_temperature": float(args.grammar_attention_temperature),
            "mobility_prior_scale": float(args.mobility_prior_scale),
            "dropout": float(args.dropout),
        }
    )
    student = POIOnlyStudent(
        student_config, grammar, checkpoint["poi_initializers"]
    ).to(device)
    student.initialize_from_teacher(teacher)
    masks = utils.split_region_masks(
        source_data, args.validation_fraction, args.seed, device
    )
    weights = weight_config(args)
    optimizer = optim.Adam(
        student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_learning_rate
    )
    save_root = Path(args.checkpoint_dir)
    result_root = Path(args.result_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_root / "stage2_{}_{}_{}_to_{}.pt".format(
        args.mode, args.output_variant, "_".join(args.source_cities), args.target_city
    )
    embedding_path = save_root / "embedding_{}_{}_{}.npy".format(
        args.mode, args.output_variant, args.target_city
    )
    best_score = float("inf")
    best_epoch = -1
    print(
        "[Stage2] mode={} grammar_control={} sources={} target={} device={} target_input={} params={:,}".format(
            args.mode,
            "learned",
            args.source_cities,
            args.target_city,
            device,
            target_data["data_access"]["representation_input"],
            sum((parameter.numel() for parameter in student.parameters())),
        ),
        flush=True,
    )
    for epoch in range(args.epochs):
        student.train()
        optimizer.zero_grad()
        train_total = torch.tensor(0.0, device=device)
        for city, data in source_data.items():
            output = student(
                data["poi_features"], data["poi_composition"], data["poi_city_context"]
            )
            terms = student.loss_terms(
                output,
                teacher_by_city[city],
                data["poi_composition"],
                masks[city]["train"],
            )
            train_total = train_total + weighted_loss(terms, weights) / len(source_data)
        train_total.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        student.eval()
        with torch.no_grad():
            validation_total = torch.tensor(0.0, device=device)
            for city, data in source_data.items():
                output = student(
                    data["poi_features"],
                    data["poi_composition"],
                    data["poi_city_context"],
                )
                terms = student.loss_terms(
                    output,
                    teacher_by_city[city],
                    data["poi_composition"],
                    masks[city]["validation"],
                )
                validation_total = validation_total + weighted_loss(
                    terms, weights
                ) / len(source_data)
        train_value = float(train_total.detach().cpu())
        validation_value = float(validation_total.detach().cpu())
        diagnostics = student_diagnostics(student, source_data)
        role_entropy = max(
            diagnostics["outgoing_entropy"], diagnostics["incoming_entropy"]
        )
        # Use the existing weighted objective on source validation regions.
        # Representation diagnostics are logged, not added to the selection score.
        selection_score = validation_value
        if selection_score < best_score:
            best_score = selection_score
            best_epoch = epoch
            stage2_source_supervision = ["source_matrix_norm.npy", "source_sum.npy"]
            bundle = {
                "format": "SIFT-stage2",
                "model_config": student_config,
                "poi_initializers": checkpoint["poi_initializers"],
                "student_state": {
                    key: value.detach().cpu()
                    for (key, value) in student.state_dict().items()
                },
                "shared_grammar": student.shared_grammar.detach().cpu(),
                "poi_statistics": poi_statistics,
                "poi_alignment": poi_alignment,
                "source_cities": list(args.source_cities),
                "target_city": args.target_city,
                "mode": args.mode,
                "output_variant": args.output_variant,
                "grammar_control": "learned",
                "stage1_checkpoint": str(Path(args.stage1_checkpoint).resolve()),
                "epoch": epoch,
                "selection_score": selection_score,
                "selection_metric": args.selection_metric,
                "validation_loss": validation_value,
                "diagnostics": diagnostics,
                "loss_weights": weights,
                "data_access": {
                    "stage1_source": [
                        "poi_f.npy",
                        "source_matrix_norm.npy",
                        "source_sum.npy",
                    ],
                    "stage2_source_input": ["poi_f.npy"],
                    "stage2_source_supervision": stage2_source_supervision,
                    "target_representation_input": ["poi_f.npy"],
                    "target_downstream_evaluation": list(
                        (utils.TASK_FILES[task] for task in target_data["labels"])
                    ),
                },
            }
            torch.save(bundle, checkpoint_path)
        if epoch % args.log_every == 0 or epoch + 1 == args.epochs:
            print(
                "[Stage2 {:4d}] train={:.6f} val={:.6f} score={:.6f} rank={:.2f} entropy={:.3f} best={:.6f}@{}".format(
                    epoch,
                    train_value,
                    validation_value,
                    selection_score,
                    diagnostics["embedding_effective_rank"],
                    role_entropy,
                    best_score,
                    best_epoch,
                ),
                flush=True,
            )
    best_bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    student.load_state_dict(best_bundle["student_state"])
    student.to(device)
    (output, full_embedding, target_diagnostics) = target_output(student, target_data)
    embedding_variants = {
        "full": full_embedding,
        "local": output["local_embedding"].cpu().numpy(),
        "local_role": torch.cat(
            [output["local_embedding"], output["role_embedding"]], dim=-1
        )
        .cpu()
        .numpy(),
        "local_grammar": torch.cat(
            [output["local_embedding"], output["grammar_embedding"]], dim=-1
        )
        .cpu()
        .numpy(),
        "grammar": output["grammar_embedding"].cpu().numpy(),
        "raw_poi": output["raw_poi_embedding"].cpu().numpy(),
        "raw_poi_grammar": torch.cat(
            [output["raw_poi_embedding"], output["grammar_embedding"]], dim=-1
        )
        .cpu()
        .numpy(),
        "raw_poi_role_grammar": torch.cat(
            [
                output["raw_poi_embedding"],
                output["role_embedding"],
                output["grammar_embedding"],
            ],
            dim=-1,
        )
        .cpu()
        .numpy(),
    }
    embedding = embedding_variants[args.output_variant]
    np.save(embedding_path, embedding)
    result = {
        "stage": "2",
        "source_cities": list(args.source_cities),
        "target_city": args.target_city,
        "mode": args.mode,
        "output_variant": args.output_variant,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "selection_metric": args.selection_metric,
        "checkpoint": str(checkpoint_path),
        "embedding": str(embedding_path),
        "embedding_shape": list(embedding.shape),
        "parameters": vars(args).copy(),
        "data_access": best_bundle["data_access"],
    }
    result_path = result_root / "stage2_{}_{}_{}_to_{}.json".format(
        args.mode, args.output_variant, "_".join(args.source_cities), args.target_city
    )
    utils.atomic_json(result_path, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    from args import parse_config

    run(parse_config(stage2_parser(), "stage2"))


if __name__ == "__main__":
    main()
