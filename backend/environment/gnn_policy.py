"""
EduPath AI — Graph Neural Network (GNN) Policy
Team KRIYA | Meta Hackathon 2026

Implements a GNN-based policy network that operates on the curriculum
prerequisite graph. Uses GCN message passing to learn topic-level
representations, then selects actions via an MLP policy head.
Compatible with Stable-Baselines3 PPO via GNNGymWrapper.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

from environment.curriculum import TOPIC_GRAPH


# Domain to one-hot index mapping
DOMAIN_MAP = {
    "tech": 0,
    "healthcare": 1,
    "business": 2,
    "law": 3,
    "design": 4,
}

# Sorted topic IDs for consistent indexing
ALL_TOPIC_IDS = sorted(TOPIC_GRAPH.keys())
TOPIC_TO_IDX = {tid: i for i, tid in enumerate(ALL_TOPIC_IDS)}
NUM_TOPICS = len(ALL_TOPIC_IDS)

# Build static edge index from prerequisite graph
def _build_edge_index() -> np.ndarray:
    """Build directed edges from prerequisite → dependent topic."""
    src, dst = [], []
    for topic_id, topic in TOPIC_GRAPH.items():
        dst_idx = TOPIC_TO_IDX[topic_id]
        for prereq_id in topic.prerequisites:
            if prereq_id in TOPIC_TO_IDX:
                src_idx = TOPIC_TO_IDX[prereq_id]
                src.append(src_idx)
                dst.append(dst_idx)
    if not src:
        # Add self-loop to avoid empty graph
        src.append(0)
        dst.append(0)
    return np.array([src, dst], dtype=np.int64)

STATIC_EDGE_INDEX = _build_edge_index()


def build_node_features(
    completed_topics: List[str],
    available_topics: List[str],
    mastery_probabilities: Dict[str, float],
) -> np.ndarray:
    """
    Build per-topic node features (dim=9):
      [is_completed, is_available, mastery_probability, difficulty_norm,
       domain_onehot (5 dims)]
    """
    features = np.zeros((NUM_TOPICS, 9), dtype=np.float32)
    completed_set = set(completed_topics)
    available_set = set(available_topics)

    for i, topic_id in enumerate(ALL_TOPIC_IDS):
        topic = TOPIC_GRAPH[topic_id]
        features[i, 0] = 1 if topic_id in completed_set else 0
        features[i, 1] = 1 if topic_id in available_set else 0
        features[i, 2] = mastery_probabilities.get(topic_id, 0.1)
        features[i, 3] = topic.difficulty / 5.0  # Normalize to 0-1
        # Domain one-hot (indices 4-8)
        domain_idx = DOMAIN_MAP.get(topic.field, 0)
        features[i, 4 + domain_idx] = 1

    return features


def build_scalar_features(
    job_readiness_score: float,
    badges_earned: int,
    total_steps: int,
    weekly_hours: int,
) -> np.ndarray:
    """Build scalar features (dim=4), all normalized to ~[0, 1]."""
    return np.array([
        job_readiness_score,
        badges_earned / 10.0,
        total_steps / 100.0,
        weekly_hours / 40.0,
    ], dtype=np.float32)


def build_topic_mask(
    completed_topics: List[str],
    available_topics: List[str],
) -> np.ndarray:
    """Create mask for valid topic selection (1=valid, 0=invalid)."""
    mask = np.zeros(NUM_TOPICS, dtype=np.float32)
    available_set = set(available_topics)
    for i, topic_id in enumerate(ALL_TOPIC_IDS):
        if topic_id in available_set:
            mask[i] = 1
    # If no topics available, allow all (fallback)
    if mask.sum() == 0:
        mask[:] = 1
    return mask


if HAS_TORCH_GEOMETRIC:
    class GnnTutoringPolicy(nn.Module):
        """
        GNN-based tutoring policy with dual action heads.

        Architecture:
          Input: Graph(node_features=Nx8, edge_index=2xE) + scalars(4)
          → 2-layer GATConv (8→64→64)
          → Global mean pool → 64d graph embedding
          → Concat with scalars → 68d
          → MLP → 128 → 64
          → Action head 1: 7 logits (action type)
          → Action head 2: N logits (topic selection, masked)
        """

        def __init__(self, node_feature_dim: int = 8, scalar_dim: int = 4,
                     hidden_dim: int = 64, num_action_types: int = 7,
                     num_topics: int = NUM_TOPICS):
            super().__init__()

            # GNN layers
            self.gat1 = GATConv(node_feature_dim, hidden_dim, heads=2, concat=False)
            self.gat2 = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)

            # MLP after concat
            combined_dim = hidden_dim + scalar_dim  # 64 + 4 = 68
            self.mlp = nn.Sequential(
                nn.Linear(combined_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
            )

            # Dual action heads
            self.action_type_head = nn.Linear(64, num_action_types)
            self.topic_head = nn.Linear(64, num_topics)

            # Value head (for PPO)
            self.value_head = nn.Linear(64, 1)

        def forward(self, node_features, edge_index, scalar_features,
                    topic_mask=None, batch=None):
            """
            Forward pass.

            Args:
                node_features: (N, 8) node feature tensor
                edge_index: (2, E) edge index tensor
                scalar_features: (B, 4) scalar features
                topic_mask: (B, num_topics) mask for valid topics
                batch: (N,) batch assignment vector for batched graphs

            Returns:
                action_type_logits: (B, 7)
                topic_logits: (B, num_topics) — masked
                value: (B, 1)
            """
            # GNN encoding
            x = F.relu(self.gat1(node_features, edge_index))
            x = F.relu(self.gat2(x, edge_index))

            # Global mean pooling
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            graph_embedding = global_mean_pool(x, batch)  # (B, 64)

            # Concatenate with scalar features
            combined = torch.cat([graph_embedding, scalar_features], dim=-1)  # (B, 68)

            # MLP
            hidden = self.mlp(combined)  # (B, 64)

            # Action heads
            action_type_logits = self.action_type_head(hidden)  # (B, 7)
            topic_logits = self.topic_head(hidden)  # (B, num_topics)

            # Apply topic mask (set invalid topics to -inf)
            if topic_mask is not None:
                topic_logits = topic_logits.masked_fill(topic_mask == 0, float('-inf'))

            # Value
            value = self.value_head(hidden)  # (B, 1)

            return action_type_logits, topic_logits, value

        def get_action(self, node_features, edge_index, scalar_features,
                       topic_mask=None, deterministic=False):
            """Sample or select actions from the policy."""
            action_type_logits, topic_logits, value = self.forward(
                node_features, edge_index, scalar_features, topic_mask
            )

            # Sample action type
            action_type_probs = F.softmax(action_type_logits, dim=-1)
            if deterministic:
                action_type = torch.argmax(action_type_probs, dim=-1)
            else:
                action_type_dist = torch.distributions.Categorical(action_type_probs)
                action_type = action_type_dist.sample()

            # Sample topic
            topic_probs = F.softmax(topic_logits, dim=-1)
            if deterministic:
                topic_idx = torch.argmax(topic_probs, dim=-1)
            else:
                topic_dist = torch.distributions.Categorical(topic_probs)
                topic_idx = topic_dist.sample()

            return action_type, topic_idx, value

else:
    # Stub when torch_geometric is not available
    class GnnTutoringPolicy:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "GnnTutoringPolicy requires torch and torch_geometric. "
                "Install with: pip install torch torch_geometric"
            )
