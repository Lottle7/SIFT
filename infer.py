"""Generate target-city SIFT embeddings from POI data only."""

import argparse
from pathlib import Path

import numpy as np
import torch

from args import ROOT
from model import POIOnlyStudent
import utils


def select_embedding(output, variant):
    blocks = {
        "full": ("embedding",),
        "local": ("local_embedding",),
        "local_role": ("local_embedding", "role_embedding"),
        "local_grammar": ("local_embedding", "grammar_embedding"),
        "grammar": ("grammar_embedding",),
        "raw_poi": ("raw_poi_embedding",),
        "raw_poi_grammar": ("raw_poi_embedding", "grammar_embedding"),
        "raw_poi_role_grammar": (
            "raw_poi_embedding",
            "role_embedding",
            "grammar_embedding",
        ),
    }
    if variant not in blocks:
        raise ValueError("Unknown output variant: " + str(variant))
    return torch.cat([output[key] for key in blocks[variant]], dim=-1)


def generate(checkpoint_path, city, data_root=ROOT, device="cpu", output_variant=None):
    bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if bundle.get("format") not in {"SIFT-stage2", "FIGUR-2026-07-27-stage2-v6"}:
        raise ValueError("Expected a SIFT Stage-II checkpoint")
    if city != bundle["target_city"]:
        raise ValueError("The city must match the checkpoint's target_city")
    device = utils.resolve_device(device)
    student = POIOnlyStudent(
        bundle["model_config"], bundle["shared_grammar"], bundle["poi_initializers"]
    )
    student.load_state_dict(bundle["student_state"])
    student.to(device).eval()
    data = utils.load_target_city(
        city,
        Path(data_root),
        device,
        bundle["poi_statistics"],
        utils.checkpoint_poi_alignment(bundle),
        load_labels=False,
    )
    variant = output_variant or bundle.get("output_variant")
    if variant is None:
        raise ValueError(
            "Specify --output_variant for a checkpoint without this metadata"
        )
    with torch.no_grad():
        output = student(
            data["poi_features"], data["poi_composition"], data["poi_city_context"]
        )
        embedding = select_embedding(output, variant).cpu().numpy()
    return embedding


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--city", choices=utils.CITIES, required=True)
    parser.add_argument("--data_root", default=str(ROOT))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True, help="Output .npy file")
    parser.add_argument("--output_variant", default=None)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    if output.suffix != ".npy":
        parser.error("--output must end in .npy")
    embedding = generate(
        args.checkpoint, args.city, args.data_root, args.device, args.output_variant
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, embedding)
    print("Saved {} embeddings to {}".format(embedding.shape, output))


if __name__ == "__main__":
    main()
