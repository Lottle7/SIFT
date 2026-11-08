"""SIFT source teacher, POI-only student, and loss components."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(input_dim, hidden_dim, output_dim, dropout):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


def categorical_kl(target, prediction):
    target = target.detach().clamp_min(1e-09)
    prediction = prediction.clamp_min(1e-09)
    return (target * (torch.log(target) - torch.log(prediction))).sum(-1).mean()


def normalized_assignment_entropy(assignments):
    entropy = -(assignments * torch.log(assignments.clamp_min(1e-09))).sum(dim=-1)
    return entropy.mean() / torch.log(
        assignments.new_tensor(float(assignments.size(1)))
    )


def role_balance_loss(assignments):
    usage = assignments.mean(dim=0)
    uniform = torch.full_like(usage, 1.0 / assignments.size(1))
    return F.kl_div(torch.log(usage.clamp_min(1e-09)), uniform, reduction="sum")


def variance_covariance_losses(embedding, target_std=0.5):
    centered = embedding - embedding.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + 0.0001)
    variance = F.relu(float(target_std) - std).mean()
    denominator = max(embedding.size(0) - 1, 1)
    covariance = torch.matmul(centered.t(), centered) / denominator
    covariance = covariance - torch.diag(torch.diagonal(covariance))
    covariance_loss = covariance.pow(2).sum() / max(embedding.size(1), 1)
    return (variance, covariance_loss)


def per_region_kl(target, prediction):
    target = target.detach().clamp_min(1e-09)
    prediction = prediction.clamp_min(1e-09)
    return (target * (torch.log(target) - torch.log(prediction))).sum(dim=-1)


def grammar_induced_messages(
    hidden, q_out, q_in, grammar, dynamic, temperature, mobility_prior_scale
):
    scores = torch.matmul(torch.matmul(q_out, grammar), q_in.t())
    prior_scale = float(mobility_prior_scale)
    outgoing_scores = scores + prior_scale * dynamic[:, 1].unsqueeze(0)
    incoming_scores = scores.t() + prior_scale * dynamic[:, 0].unsqueeze(0)
    if scores.size(0) > 1:
        eye = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)
        outgoing_scores = outgoing_scores.masked_fill(eye, -10000.0)
        incoming_scores = incoming_scores.masked_fill(eye, -10000.0)
    scale = max(float(temperature), 0.001)
    outgoing_attention = torch.softmax(outgoing_scores / scale, dim=-1)
    incoming_attention = torch.softmax(incoming_scores / scale, dim=-1)
    return (
        torch.matmul(outgoing_attention, hidden),
        torch.matmul(incoming_attention, hidden),
        outgoing_attention,
        incoming_attention,
    )


class DisentangledEmbeddingHead(nn.Module):
    """Keep local, role, and grammar information in separate subspaces."""

    def __init__(
        self,
        local_input_dim,
        role_input_dim,
        grammar_input_dim,
        hidden_dim,
        embedding_dim,
        dropout,
    ):
        super(DisentangledEmbeddingHead, self).__init__()
        local_dim = max(embedding_dim // 2, 8)
        role_dim = max(embedding_dim // 4, 4)
        grammar_dim = embedding_dim - local_dim - role_dim
        if grammar_dim < 4:
            raise ValueError("embedding_dim is too small for three blocks")
        self.block_dims = {"local": local_dim, "role": role_dim, "grammar": grammar_dim}
        self.local = build_mlp(local_input_dim, hidden_dim, local_dim, dropout)
        self.role = build_mlp(role_input_dim, hidden_dim, role_dim, dropout)
        self.grammar = build_mlp(grammar_input_dim, hidden_dim, grammar_dim, dropout)
        self.local_norm = nn.LayerNorm(local_dim)
        self.role_norm = nn.LayerNorm(role_dim)
        self.grammar_norm = nn.LayerNorm(grammar_dim)

    def forward(
        self, local_input, role_input, grammar_input, grammar_gate, grammar_enabled=True
    ):
        local = self.local_norm(self.local(local_input))
        role = self.role_norm(self.role(role_input))
        grammar = self.grammar_norm(self.grammar(grammar_input))
        if not grammar_enabled:
            grammar = torch.zeros_like(grammar)
        else:
            grammar = grammar * grammar_gate
        embedding = torch.cat([local, role, grammar], dim=-1)
        return {
            "embedding": embedding,
            "local_embedding": local,
            "role_embedding": role,
            "grammar_embedding": grammar,
        }


class POIEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, role_dim, dropout):
        super(POIEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, role_dim),
            nn.LayerNorm(role_dim),
            nn.GELU(),
        )

    def forward(self, poi_features):
        return self.network(poi_features)


class FixedPOIRoles(nn.Module):
    """Cross-city anchors in the taxonomy-free POI frequency-rank space."""

    def __init__(self, poi_initializers, temperature):
        super(FixedPOIRoles, self).__init__()
        profiles = torch.tensor(poi_initializers, dtype=torch.float32)
        profiles = profiles / profiles.sum(dim=-1, keepdim=True).clamp_min(1e-09)
        self.register_buffer("profiles", profiles)
        self.temperature = float(temperature)

    def forward(self, poi_composition):
        logits = torch.matmul(
            F.normalize(poi_composition, dim=-1), F.normalize(self.profiles, dim=-1).t()
        )
        return torch.softmax(logits / max(self.temperature, 0.001), dim=-1)


class _TeacherBase(nn.Module):
    """Shared components of the source-city teacher."""

    def __init__(self, config, poi_initializers):
        super(_TeacherBase, self).__init__()
        self.config = dict(config)
        self.role_count = int(config["role_count"])
        self.direction_scale = float(config["direction_scale"])
        self.grammar_attention_temperature = float(
            config.get("grammar_attention_temperature", 0.5)
        )
        self.mobility_prior_scale = float(config.get("mobility_prior_scale", 0.5))
        role_dim = int(config["role_dim"])
        hidden_dim = int(config["hidden_dim"])
        embedding_dim = int(config["embedding_dim"])
        poi_feature_dim = int(config.get("poi_feature_dim", 27))
        poi_category_dim = int(config.get("poi_category_dim", 26))
        self.poi_feature_dim = poi_feature_dim
        self.poi_category_dim = poi_category_dim
        dropout = float(config["dropout"])
        self.city_context_adapter = None
        if config.get("poi_city_adapter", False):
            raise ValueError("City-adapter checkpoints are outside this release")
        self.roles = FixedPOIRoles(poi_initializers, config["temperature"])
        self.poi_encoder = POIEncoder(poi_feature_dim, hidden_dim, role_dim, dropout)
        self.poi_hidden_decoder = nn.Linear(role_dim, poi_category_dim)
        context_dim = role_dim * 2 + 6
        self.out_context = build_mlp(context_dim, hidden_dim, role_dim, dropout)
        self.in_context = build_mlp(context_dim, hidden_dim, role_dim, dropout)
        self.out_adjustment = nn.Linear(role_dim, self.role_count)
        self.in_adjustment = nn.Linear(role_dim, self.role_count)
        nn.init.zeros_(self.out_adjustment.weight)
        nn.init.zeros_(self.out_adjustment.bias)
        nn.init.zeros_(self.in_adjustment.weight)
        nn.init.zeros_(self.in_adjustment.bias)
        self.shared_grammar = nn.Parameter(
            torch.zeros(self.role_count, self.role_count)
        )
        self.embedding_head = DisentangledEmbeddingHead(
            local_input_dim=poi_feature_dim + role_dim,
            role_input_dim=3 * self.role_count + 6,
            grammar_input_dim=2 * role_dim + 2 * self.role_count,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.poi_decoder = nn.Linear(embedding_dim, poi_category_dim)
        self.dynamic_decoder = nn.Linear(embedding_dim, 6)

    def adapt_poi(self, poi_features, poi_composition, city_context):
        encoder_features = poi_features[:, : self.poi_feature_dim]
        gate = poi_composition.new_ones(self.poi_category_dim)
        return (encoder_features, poi_composition, gate)

    @staticmethod
    def empirical_grammar(q_out, q_in, residual):
        detached_out = q_out.detach()
        detached_in = q_in.detach()
        numerator = torch.matmul(torch.matmul(detached_out.t(), residual), detached_in)
        denominator = torch.outer(
            detached_out.sum(dim=0), detached_in.sum(dim=0)
        ).clamp_min(1e-06)
        return (numerator / denominator).clamp(-1.0, 1.0)

    def directed_roles(
        self, q_semantic, poi_hidden, a_out, a_in, dynamic, direction_enabled
    ):
        if not direction_enabled:
            return (q_semantic, q_semantic)
        outgoing_neighbors = torch.matmul(a_out, poi_hidden)
        incoming_neighbors = torch.matmul(a_in, poi_hidden)
        outgoing_hidden = self.out_context(
            torch.cat([poi_hidden, outgoing_neighbors, dynamic], dim=-1)
        )
        incoming_hidden = self.in_context(
            torch.cat([poi_hidden, incoming_neighbors, dynamic], dim=-1)
        )
        outgoing_delta = self.direction_scale * torch.tanh(
            self.out_adjustment(outgoing_hidden)
        )
        incoming_delta = self.direction_scale * torch.tanh(
            self.in_adjustment(incoming_hidden)
        )
        semantic_log = torch.log(q_semantic.clamp_min(1e-09))
        q_out = torch.softmax(semantic_log + outgoing_delta, dim=-1)
        q_in = torch.softmax(semantic_log + incoming_delta, dim=-1)
        return (q_out, q_in)

    def loss_terms(self, output, poi_composition, dynamic, residual):
        return {
            "poi_hidden": F.mse_loss(
                output["poi_hidden_reconstruction"], poi_composition
            ),
            "poi": F.mse_loss(output["poi_reconstruction"], poi_composition),
            "dynamic": F.smooth_l1_loss(output["dynamic_reconstruction"], dynamic),
            "grammar": F.mse_loss(output["grammar"], output["empirical_grammar"]),
            "residual": F.smooth_l1_loss(output["predicted_residual"], residual),
            "directional_anchor": 0.5
            * (
                categorical_kl(output["q_semantic"], output["q_out"])
                + categorical_kl(output["q_semantic"], output["q_in"])
            ),
            "balance": 0.5
            * (role_balance_loss(output["q_out"]) + role_balance_loss(output["q_in"])),
            "sharpness": 0.5
            * (
                normalized_assignment_entropy(output["q_out"])
                + normalized_assignment_entropy(output["q_in"])
            ),
            "variance": output["variance_loss"],
            "covariance": output["covariance_loss"],
        }


class POIOnlyStudent(nn.Module):
    """Target-deployable SIFT student; forward accepts POI only."""

    def __init__(self, config, grammar, poi_initializers):
        super(POIOnlyStudent, self).__init__()
        self.config = dict(config)
        self.role_count = int(config["role_count"])
        self.direction_scale = float(config["direction_scale"])
        self.grammar_attention_temperature = float(
            config.get("grammar_attention_temperature", 0.5)
        )
        self.mobility_prior_scale = float(config.get("mobility_prior_scale", 0.5))
        hidden_dim = int(config["hidden_dim"])
        role_dim = int(config["role_dim"])
        embedding_dim = int(config["embedding_dim"])
        poi_feature_dim = int(config.get("poi_feature_dim", 27))
        poi_category_dim = int(config.get("poi_category_dim", 26))
        self.poi_feature_dim = poi_feature_dim
        self.poi_category_dim = poi_category_dim
        dropout = float(config["dropout"])
        self.city_context_adapter = None
        if config.get("poi_city_adapter", False):
            raise ValueError("City-adapter checkpoints are outside this release")
        self.roles = FixedPOIRoles(poi_initializers, config["temperature"])
        self.poi_encoder = POIEncoder(poi_feature_dim, hidden_dim, role_dim, dropout)
        self.out_head = build_mlp(role_dim, hidden_dim, self.role_count, dropout)
        self.in_head = build_mlp(role_dim, hidden_dim, self.role_count, dropout)
        self.dynamic_head = build_mlp(role_dim, hidden_dim, 6, dropout)
        self.confidence_head = build_mlp(role_dim, hidden_dim, 1, dropout)
        self.register_buffer("shared_grammar", grammar.detach().clone())
        self.embedding_head = DisentangledEmbeddingHead(
            local_input_dim=poi_feature_dim + role_dim,
            role_input_dim=3 * self.role_count + 6,
            grammar_input_dim=2 * role_dim + 2 * self.role_count,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.poi_decoder = nn.Linear(embedding_dim, poi_category_dim)

    def initialize_from_teacher(self, teacher):
        if self.city_context_adapter is not None:
            if teacher.city_context_adapter is None:
                raise ValueError("Teacher is missing the POI city adapter")
            self.city_context_adapter.load_state_dict(
                teacher.city_context_adapter.state_dict()
            )
        self.poi_encoder.load_state_dict(teacher.poi_encoder.state_dict())
        self.embedding_head.load_state_dict(teacher.embedding_head.state_dict())
        self.poi_decoder.load_state_dict(teacher.poi_decoder.state_dict())

    def adapt_poi(self, poi_features, poi_composition, city_context):
        encoder_features = poi_features[:, : self.poi_feature_dim]
        gate = poi_composition.new_ones(self.poi_category_dim)
        return (encoder_features, poi_composition, gate)

    def forward(self, poi_features, poi_composition, city_context=None):
        (encoder_features, adapted_composition, adapter_gate) = self.adapt_poi(
            poi_features, poi_composition, city_context
        )
        hidden = self.poi_encoder(encoder_features)
        q_semantic = self.roles(adapted_composition)
        semantic_log = torch.log(q_semantic.clamp_min(1e-09))
        q_out = torch.softmax(
            semantic_log + self.direction_scale * torch.tanh(self.out_head(hidden)),
            dim=-1,
        )
        q_in = torch.softmax(
            semantic_log + self.direction_scale * torch.tanh(self.in_head(hidden)),
            dim=-1,
        )
        dynamic = self.dynamic_head(hidden)
        confidence = torch.sigmoid(self.confidence_head(hidden))
        outgoing_context = torch.matmul(q_out, self.shared_grammar)
        incoming_context = torch.matmul(q_in, self.shared_grammar.t())
        (outgoing_message, incoming_message, outgoing_attention, incoming_attention) = (
            grammar_induced_messages(
                hidden,
                q_out,
                q_in,
                self.shared_grammar,
                dynamic,
                self.grammar_attention_temperature,
                self.mobility_prior_scale,
            )
        )
        grammar_enabled = bool(
            torch.count_nonzero(self.shared_grammar).detach().cpu().item()
        )
        blocks = self.embedding_head(
            torch.cat([encoder_features, hidden], dim=-1),
            torch.cat([q_semantic, q_out, q_in, dynamic], dim=-1),
            torch.cat(
                [
                    outgoing_message,
                    incoming_message,
                    outgoing_context,
                    incoming_context,
                ],
                dim=-1,
            ),
            confidence,
            grammar_enabled=grammar_enabled,
        )
        embedding = blocks["embedding"]
        (variance, covariance) = variance_covariance_losses(embedding)
        result = {
            "embedding": embedding,
            "raw_poi_embedding": poi_features[
                :, self.poi_feature_dim : self.poi_feature_dim + self.poi_category_dim
            ]
            * 4.0,
            "adapted_poi_composition": adapted_composition,
            "poi_adapter_gate": adapter_gate,
            "q_semantic": q_semantic,
            "q_out": q_out,
            "q_in": q_in,
            "dynamic": dynamic,
            "confidence": confidence,
            "outgoing_context": outgoing_context,
            "incoming_context": incoming_context,
            "outgoing_message": outgoing_message,
            "incoming_message": incoming_message,
            "outgoing_attention": outgoing_attention,
            "incoming_attention": incoming_attention,
            "poi_reconstruction": torch.softmax(self.poi_decoder(embedding), dim=-1),
            "variance_loss": variance,
            "covariance_loss": covariance,
        }
        result.update(blocks)
        return result

    @staticmethod
    def masked_mean(values, mask):
        selected = values[mask]
        if selected.numel() == 0:
            return values.new_tensor(0.0)
        return selected.mean()

    def loss_terms(self, output, teacher, poi_composition, mask):
        dynamic_error = (
            (output["dynamic"] - teacher["dynamic_target"]).pow(2).mean(dim=-1)
        )
        confidence_target = teacher["grammar_reliability"].squeeze(-1)
        confidence_error = (output["confidence"].squeeze(-1) - confidence_target).pow(2)
        embedding_error = (
            (
                F.normalize(output["embedding"], dim=-1)
                - F.normalize(teacher["embedding"], dim=-1)
            )
            .pow(2)
            .mean(dim=-1)
        )
        grammar_embedding_error = (
            (
                F.normalize(output["grammar_embedding"], dim=-1)
                - F.normalize(teacher["grammar_embedding"], dim=-1)
            )
            .pow(2)
            .mean(dim=-1)
        )
        role_context_error = (
            (output["outgoing_context"] - teacher["outgoing_context"]).pow(2)
            + (output["incoming_context"] - teacher["incoming_context"]).pow(2)
        ).mean(dim=-1)
        message_context_error = 0.5 * (
            (
                F.normalize(output["outgoing_message"], dim=-1)
                - F.normalize(teacher["outgoing_message"], dim=-1)
            )
            .pow(2)
            .mean(dim=-1)
            + (
                F.normalize(output["incoming_message"], dim=-1)
                - F.normalize(teacher["incoming_message"], dim=-1)
            )
            .pow(2)
            .mean(dim=-1)
        )
        context_error = role_context_error + message_context_error
        poi_error = (output["poi_reconstruction"] - poi_composition).pow(2).mean(dim=-1)
        role_error = 0.5 * (
            (
                teacher["q_out"].detach()
                * (
                    torch.log(teacher["q_out"].detach().clamp_min(1e-09))
                    - torch.log(output["q_out"].clamp_min(1e-09))
                )
            ).sum(dim=-1)
            + (
                teacher["q_in"].detach()
                * (
                    torch.log(teacher["q_in"].detach().clamp_min(1e-09))
                    - torch.log(output["q_in"].clamp_min(1e-09))
                )
            ).sum(dim=-1)
        )
        sharpness = 0.5 * (
            normalized_assignment_entropy(output["q_out"])
            + normalized_assignment_entropy(output["q_in"])
        )
        balance = 0.5 * (
            role_balance_loss(output["q_out"]) + role_balance_loss(output["q_in"])
        )
        return {
            "role": self.masked_mean(role_error, mask),
            "dynamic": self.masked_mean(dynamic_error, mask),
            "confidence": self.masked_mean(confidence_error, mask),
            "context": self.masked_mean(context_error, mask),
            "grammar_embedding": self.masked_mean(grammar_embedding_error, mask),
            "embedding": self.masked_mean(embedding_error, mask),
            "poi": self.masked_mean(poi_error, mask),
            "sharpness": sharpness,
            "balance": balance,
            "variance": output["variance_loss"],
            "covariance": output["covariance_loss"],
        }


class SourceGrammarTeacher(_TeacherBase):
    """SIFT source teacher with shared interactions and bounded city-specific deviations."""

    def __init__(self, config, poi_initializers):
        super(SourceGrammarTeacher, self).__init__(config, poi_initializers)
        self.source_cities = tuple(config["source_cities"])
        self.city_decomposition = config.get("city_decomposition", "bounded")
        if self.city_decomposition != "bounded":
            raise ValueError("This release uses the bounded SIFT decomposition")
        self.city_delta_rank = int(config.get("city_delta_rank", 2))
        self.city_delta_scale = float(config.get("city_delta_scale", 0.25))
        self.city_delta_penalty = float(config.get("city_delta_penalty", 0.1))
        self.city_center_penalty = float(config.get("city_center_penalty", 0.2))
        if self.city_delta_rank < 0:
            raise ValueError("city_delta_rank must be non-negative")
        width = self.role_count + 2 * self.city_delta_rank * len(self.source_cities)
        packed = torch.zeros(self.role_count, width)
        for index in range(len(self.source_cities)):
            (left, _) = self._factor_slices(index)
            packed[:, left].normal_(mean=0.0, std=0.02)
        self.shared_grammar = torch.nn.Parameter(packed)

    def _factor_slices(self, city_index):
        start = self.role_count + city_index * 2 * self.city_delta_rank
        left = slice(start, start + self.city_delta_rank)
        right = slice(start + self.city_delta_rank, start + 2 * self.city_delta_rank)
        return (left, right)

    def city_factors(self, city):
        if city not in self.source_cities:
            raise KeyError("Unknown source city {}".format(city))
        city_index = self.source_cities.index(city)
        (left_slice, right_slice) = self._factor_slices(city_index)
        return (self.shared_grammar[:, left_slice], self.shared_grammar[:, right_slice])

    def raw_city_deviation(self, city):
        (left, right) = self.city_factors(city)
        return torch.matmul(left, right.t())

    def city_deviation(self, city):
        return self.city_delta_scale * torch.tanh(self.raw_city_deviation(city))

    def all_city_deviations(self):
        return torch.stack(
            [self.city_deviation(city) for city in self.source_cities], dim=0
        )

    def grammar(self, city=None):
        shared = torch.tanh(self.shared_grammar[:, : self.role_count])
        if city is None:
            return shared
        return (shared + self.city_deviation(city)).clamp(-1.0, 1.0)

    def context_grammar(self, city):
        return self.grammar()

    def deviation_regularizer(self):
        deviations = self.all_city_deviations()
        magnitude = deviations.pow(2).mean()
        center = deviations.mean(dim=0).pow(2).mean()
        return self.city_delta_penalty * magnitude + self.city_center_penalty * center

    def deviation_diagnostics(self):
        with torch.no_grad():
            deviations = self.all_city_deviations()
            shared_norm = torch.linalg.norm(self.grammar())
            deviation_norm = torch.linalg.norm(deviations, dim=(-2, -1)).mean()
            result = {
                "city_decomposition": self.city_decomposition,
                "shared_frobenius": float(shared_norm.cpu()),
                "center_frobenius": float(
                    torch.linalg.norm(deviations.mean(dim=0)).cpu()
                ),
                "deviation_to_shared_ratio": float(
                    (deviation_norm / shared_norm.clamp_min(1e-09)).cpu()
                ),
                "mean_abs": float(deviations.abs().mean().cpu()),
                "max_abs": float(deviations.abs().max().cpu()),
            }
            for index, city in enumerate(self.source_cities):
                result[city + "_frobenius"] = float(
                    torch.linalg.norm(deviations[index]).cpu()
                )
            return result

    def forward(
        self,
        poi_features,
        poi_composition,
        a_out,
        a_in,
        dynamic,
        residual,
        direction_enabled=True,
        city=None,
        city_context=None,
    ):
        if city is None:
            raise ValueError("Source forward requires a source city")
        (encoder_features, adapted_composition, adapter_gate) = self.adapt_poi(
            poi_features, poi_composition, city_context
        )
        poi_hidden = self.poi_encoder(encoder_features)
        q_semantic = self.roles(adapted_composition)
        (q_out, q_in) = self.directed_roles(
            q_semantic, poi_hidden, a_out, a_in, dynamic, direction_enabled
        )
        shared_grammar = self.grammar()
        effective_grammar = self.grammar(city)
        context_grammar = self.context_grammar(city)
        outgoing_context = torch.matmul(q_out, context_grammar)
        incoming_context = torch.matmul(q_in, context_grammar.t())
        outgoing_message = torch.matmul(a_out, poi_hidden)
        incoming_message = torch.matmul(a_in, poi_hidden)
        (
            induced_outgoing_message,
            induced_incoming_message,
            induced_outgoing_attention,
            induced_incoming_attention,
        ) = grammar_induced_messages(
            poi_hidden,
            q_out,
            q_in,
            context_grammar,
            dynamic,
            self.grammar_attention_temperature,
            self.mobility_prior_scale,
        )
        empirical = self.empirical_grammar(q_out, q_in, residual)
        predicted_residual = torch.matmul(
            torch.matmul(q_out, effective_grammar), q_in.t()
        )
        message_error = 0.5 * (
            (
                F.normalize(outgoing_message, dim=-1)
                - F.normalize(induced_outgoing_message, dim=-1)
            )
            .pow(2)
            .mean(dim=-1)
            + (
                F.normalize(incoming_message, dim=-1)
                - F.normalize(induced_incoming_message, dim=-1)
            )
            .pow(2)
            .mean(dim=-1)
        )
        grammar_reliability = (
            torch.exp(-10.0 * message_error).clamp(0.0, 1.0).unsqueeze(-1)
        )
        blocks = self.embedding_head(
            torch.cat([encoder_features, poi_hidden], dim=-1),
            torch.cat([q_semantic, q_out, q_in, dynamic], dim=-1),
            torch.cat(
                [
                    outgoing_message,
                    incoming_message,
                    outgoing_context,
                    incoming_context,
                ],
                dim=-1,
            ),
            grammar_reliability,
            grammar_enabled=True,
        )
        embedding = blocks["embedding"]
        (variance, covariance) = variance_covariance_losses(embedding)
        result = {
            "embedding": embedding,
            "raw_poi_embedding": poi_features[
                :, self.poi_feature_dim : self.poi_feature_dim + self.poi_category_dim
            ]
            * 4.0,
            "poi_hidden": poi_hidden,
            "adapted_poi_composition": adapted_composition,
            "poi_adapter_gate": adapter_gate,
            "q_semantic": q_semantic,
            "q_out": q_out,
            "q_in": q_in,
            "outgoing_context": outgoing_context,
            "incoming_context": incoming_context,
            "outgoing_message": outgoing_message,
            "incoming_message": incoming_message,
            "induced_outgoing_message": induced_outgoing_message,
            "induced_incoming_message": induced_incoming_message,
            "induced_outgoing_attention": induced_outgoing_attention,
            "induced_incoming_attention": induced_incoming_attention,
            "grammar_reliability": grammar_reliability,
            "grammar": effective_grammar,
            "shared_grammar": shared_grammar,
            "context_grammar": context_grammar,
            "city_deviation": self.city_deviation(city),
            "empirical_grammar": empirical,
            "predicted_residual": predicted_residual,
            "poi_hidden_reconstruction": torch.softmax(
                self.poi_hidden_decoder(poi_hidden), dim=-1
            ),
            "poi_reconstruction": torch.softmax(self.poi_decoder(embedding), dim=-1),
            "dynamic_reconstruction": self.dynamic_decoder(embedding),
            "variance_loss": variance,
            "covariance_loss": covariance,
        }
        result.update(blocks)
        return result

    def loss_terms(self, output, poi_composition, dynamic, residual):
        terms = super(SourceGrammarTeacher, self).loss_terms(
            output, poi_composition, dynamic, residual
        )
        terms["residual"] = terms["residual"] + self.deviation_regularizer()
        return terms
