"""
Losses and diagnostics for shaping the flow embedding space.

The sequence autoencoder's only objective is reconstruction, so nothing in it says
where a flow should sit relative to other flows. What is here adds that: a supervised
contrastive term applied to the L2-normalised bottleneck vector, plus the measurements
needed to tell a working term from one that is silently a no-op or is quietly
collapsing the space.

Why one level and not a coarse/fine hierarchy: on IIoTset-Ferrag there is nothing to
nest. 96% of train flows and 99.7% of test flows are MQTT over TCP, and four of the
five largest endpoint-pair groups carry exactly one port and one protocol, so a
"protocol within endpoint" level partitions nothing (InspectFlowGroups.py reports this
as subs/parent). A second level belongs here only once a capture is available that has
one; adding it back means a second supcon call at a lower temperature, since fine
labels are built from coarse ones and therefore already nest.

Nothing here holds state or parameters -- these are functions over a batch of
embeddings, so they compose with whatever the training script already computes.
"""
import torch


def supcon(u: torch.Tensor, labels: torch.Tensor, temperature: float,
           balance_groups: bool = True) -> tuple[torch.Tensor, dict]:
    """
    Supervised contrastive loss, the L_out form of Khosla et al. (2020).

    Args:
        u:           (B, D) embeddings, already L2-normalised. Pass float32 -- a
                     512-dim dot product in bf16 carries about three significant
                     digits, which is not enough for a small temperature.
        labels:      (B,) integer group id per row.
        temperature: lower means a sharper softmax, so a tighter cluster.
        balance_groups: average per group before averaging across groups, so every
                     group in the batch counts equally regardless of how many anchors
                     it contributed. This matters here: one endpoint pair holds 27% of
                     train windows, and a plain anchor mean would hand it 27% of the
                     loss and let it dictate the geometry. Set False for the textbook
                     form.

    Returns:
        (loss, diagnostics). The loss is a scalar tensor; diagnostics carries
        `positives_per_anchor`, `usable_anchor_frac` and `groups_in_batch`.

    Anchors with no other member of their group in the batch have no positives and are
    excluded rather than contributing zero -- including them would silently scale the
    loss down by the fraction of singleton anchors. If no anchor has a positive the
    term returns 0 and `usable_anchor_frac` is 0, which is the signal that the batch
    sampler is not putting groups together.
    """
    B = u.shape[0]
    zero = u.new_zeros(())
    empty = {"positives_per_anchor": 0.0, "usable_anchor_frac": 0.0, "groups_in_batch": 0.0}
    if B < 2:
        return zero, empty

    self_mask = torch.eye(B, dtype=torch.bool, device=u.device)
    sim = (u @ u.T) / temperature
    sim = sim.masked_fill(self_mask, float("-inf"))

    # torch.logsumexp subtracts the row max internally, so this is the stable form of
    # log softmax over the non-self entries of each row.
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    positives = (labels.view(-1, 1) == labels.view(1, -1)) & ~self_mask
    counts = positives.sum(dim=1)
    usable = counts > 0
    if not bool(usable.any()):
        empty["groups_in_batch"] = float(labels.unique().numel())
        return zero, empty

    # masked_fill rather than a multiply: log_prob holds -inf on the diagonal, and
    # -inf * 0.0 is nan.
    summed = log_prob.masked_fill(~positives, 0.0).sum(dim=1)
    per_anchor = -(summed[usable] / counts[usable])
    anchor_labels = labels[usable]

    if balance_groups:
        groups, inverse = anchor_labels.unique(return_inverse=True)
        totals = torch.zeros(groups.numel(), device=u.device, dtype=per_anchor.dtype)
        totals = totals.index_add(0, inverse, per_anchor)
        sizes = torch.zeros_like(totals).index_add(
            0, inverse, torch.ones_like(per_anchor))
        loss = (totals / sizes).mean()
    else:
        loss = per_anchor.mean()

    diagnostics = {
        "positives_per_anchor": float(counts[usable].float().mean()),
        "usable_anchor_frac": float(usable.float().mean()),
        "groups_in_batch": float(labels.unique().numel()),
    }
    return loss, diagnostics


@torch.no_grad()
def effective_rank(z: torch.Tensor) -> float:
    """
    Participation ratio of the embedding covariance spectrum, (sum L)^2 / sum L^2.

    The collapse alarm. A contrastive term over a modest number of groups can flatten
    the embedding onto a low-dimensional simplex -- which reads as beautiful cluster
    separation while destroying exactly the local geometry the kNN, Mahalanobis and
    LOF detectors depend on. That risk is concrete here: 18 endpoint-pair groups carry
    98% of the training windows, so the coarse term is close to an 18-way
    classification in a 512-dimensional space.

    Sits between 1 (one direction carries everything) and D. Computed from the singular
    values of the centred batch rather than by forming the covariance, so it stays well
    conditioned at D = 512.
    """
    x = z.float()
    x = x - x.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(x)
    eig = sv ** 2
    total = eig.sum()
    if float(total) <= 0.0:
        return 0.0
    return float((total ** 2) / (eig ** 2).sum())


@torch.no_grad()
def group_distance_stats(u: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    Mean cosine distance within a group vs. between groups, and their ratio.

    The plainest reading of "are the clusters real": `ratio` below 1 means same-group
    pairs are closer than different-group pairs, and smaller is tighter. Unlike a
    silhouette it needs no per-point nearest-cluster search, so it is cheap enough to
    run every epoch on a large sample.

    Args:
        u:      (N, D) L2-normalised embeddings.
        labels: (N,) integer group ids.
    """
    N = u.shape[0]
    dist = 1.0 - (u @ u.T)
    same = labels.view(-1, 1) == labels.view(1, -1)
    off_diagonal = ~torch.eye(N, dtype=torch.bool, device=u.device)

    intra_mask = same & off_diagonal
    inter_mask = (~same) & off_diagonal
    intra = float(dist[intra_mask].mean()) if bool(intra_mask.any()) else float("nan")
    inter = float(dist[inter_mask].mean()) if bool(inter_mask.any()) else float("nan")
    return {"intra": intra, "inter": inter,
            "ratio": intra / inter if inter == inter else float("nan")}


@torch.no_grad()
def knn_group_agreement(u: torch.Tensor, labels: torch.Tensor, k: int = 10) -> float:
    """
    Fraction of each point's k nearest neighbours (cosine) that share its group.

    Local where the silhouette and the distance ratio are global: it answers "is my
    immediate neighbourhood made of my own kind", which is the property a
    nearest-neighbour anomaly detector actually consumes. Points whose group has no
    other member present are skipped, since their score would be 0 by construction.
    """
    N = u.shape[0]
    if N <= 1:
        return float("nan")
    sim = u @ u.T
    sim.fill_diagonal_(float("-inf"))
    k = min(k, N - 1)
    neighbours = sim.topk(k, dim=1).indices
    agree = (labels[neighbours] == labels.view(-1, 1)).float().mean(dim=1)

    counts = torch.bincount(labels, minlength=int(labels.max()) + 1)
    has_peer = counts[labels] > 1
    if not bool(has_peer.any()):
        return float("nan")
    return float(agree[has_peer].mean())
