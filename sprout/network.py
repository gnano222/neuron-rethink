"""The SPROUT network: neurons + synapses as explicit objects (§3, §4.1, §4.4).

We deliberately do *not* use dense weight matrices. The per-synapse state
(confidence, eligibility, age) is where the novelty lives and what we want to
watch, so the graph is stored as:

  * ``neurons``  - list of :class:`Neuron`, indexed by id.
  * ``synapses`` - dict keyed by ``(pre, post)`` -> :class:`Synapse`.
  * ``incoming`` / ``outgoing`` - per-neuron adjacency (lists of pre / post ids).

Forward and backward are hand-rolled over the adjacency lists so the irregular,
mutating sparse topology is handled directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Neuron:
    id: int
    layer: int
    pos: tuple              # fixed 2D coordinate, for drawing
    activation: float = 0.0  # a_j from the most recent forward pass
    firing_rate: float = 0.0  # r_j, EMA of activation
    bias: float = 0.0
    grow_attempts: int = 0    # times growth has targeted this neuron (budget)


@dataclass
class Synapse:
    pre: int
    post: int
    weight: float = 0.0
    confidence: float = 0.0   # c >= 0, novel
    eligibility: float = 0.0  # e >= 0, novel ("fired together recently")
    age: int = 0              # steps since birth, for the grace period
    grad_mag: float = 0.0     # EMA of |dL/dw|  - the "currency" magnitude meter
    grad_signed: float = 0.0  # EMA of  dL/dw   - the signed meter (for consistency)


class Network:
    def __init__(self, layer_sizes):
        self.layer_sizes = list(layer_sizes)
        self.neurons: list[Neuron] = []
        self.synapses: dict[tuple[int, int], Synapse] = {}
        self.incoming: dict[int, list[int]] = {}   # post -> [pre, ...]
        self.outgoing: dict[int, list[int]] = {}   # pre  -> [post, ...]
        self.layers: list[list[int]] = []

        nid = 0
        for layer_idx, size in enumerate(self.layer_sizes):
            ids = []
            for k in range(size):
                # column = layer, row = position within layer (centred)
                y = k - (size - 1) / 2.0
                self.neurons.append(Neuron(id=nid, layer=layer_idx, pos=(float(layer_idx), float(y))))
                self.incoming[nid] = []
                self.outgoing[nid] = []
                ids.append(nid)
                nid += 1
            self.layers.append(ids)

    # -- structure ----------------------------------------------------------
    @property
    def num_neurons(self) -> int:
        return len(self.neurons)

    def add_synapse(self, pre, post, weight=0.0, confidence=0.0, eligibility=0.0, age=0):
        if self.neurons[pre].layer >= self.neurons[post].layer:
            raise ValueError(f"synapse {pre}->{post} must go to a later layer")
        if (pre, post) in self.synapses:
            return self.synapses[(pre, post)]
        syn = Synapse(pre, post, weight, confidence, eligibility, age)
        self.synapses[(pre, post)] = syn
        self.incoming[post].append(pre)
        self.outgoing[pre].append(post)
        return syn

    def remove_synapse(self, pre, post):
        if (pre, post) not in self.synapses:
            return
        del self.synapses[(pre, post)]
        self.incoming[post].remove(pre)
        self.outgoing[pre].remove(post)

    # -- forward (§4.1) -----------------------------------------------------
    def forward(self, x):
        x = np.asarray(x, dtype=float)
        # input neurons hold the raw coordinates
        for k, nid in enumerate(self.layers[0]):
            self.neurons[nid].activation = float(x[k])

        # hidden layers: z = bias + sum(w * a_pre); a = ReLU(z)
        last = len(self.layers) - 1
        for layer_idx in range(1, last):
            for nid in self.layers[layer_idx]:
                z = self.neurons[nid].bias
                for pre in self.incoming[nid]:
                    z += self.synapses[(pre, nid)].weight * self.neurons[pre].activation
                self.neurons[nid].activation = z if z > 0.0 else 0.0

        # output layer: collect z, then softmax over the (2) outputs
        out_ids = self.layers[last]
        zs = np.empty(len(out_ids))
        for i, nid in enumerate(out_ids):
            z = self.neurons[nid].bias
            for pre in self.incoming[nid]:
                z += self.synapses[(pre, nid)].weight * self.neurons[pre].activation
            zs[i] = z
        probs = _softmax(zs)
        for i, nid in enumerate(out_ids):
            self.neurons[nid].activation = float(probs[i])
        return probs

    # -- backward (§4.4) ----------------------------------------------------
    def backward(self, y_true):
        """Cross-entropy + softmax backprop over live edges.

        Assumes :meth:`forward` has just been run (activations are current).
        Returns ``(loss, grad_w, grad_b)`` where grads only cover live synapses
        and non-input neurons.
        """
        last = len(self.layers) - 1
        out_ids = self.layers[last]
        probs = np.array([self.neurons[nid].activation for nid in out_ids])
        loss = float(-np.log(probs[y_true] + 1e-12))

        delta: dict[int, float] = {}
        # output pre-activation gradient: softmax + cross-entropy => p - onehot
        for i, nid in enumerate(out_ids):
            delta[nid] = probs[i] - (1.0 if i == y_true else 0.0)

        # hidden layers in reverse: delta_j = (sum w_jk delta_k) * relu'(z_j)
        for layer_idx in range(last - 1, 0, -1):
            for nid in self.layers[layer_idx]:
                acc = 0.0
                for post in self.outgoing[nid]:
                    acc += self.synapses[(nid, post)].weight * delta[post]
                relu_grad = 1.0 if self.neurons[nid].activation > 0.0 else 0.0
                delta[nid] = acc * relu_grad

        grad_w = {}
        for (pre, post), syn in self.synapses.items():
            grad_w[(pre, post)] = delta[post] * self.neurons[pre].activation

        grad_b = {nid: delta[nid] for nid in delta}  # all non-input neurons
        return loss, grad_w, grad_b


def _softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


# ----------------------------------------------------------------------------
# Graph construction (§5.1) and weight init (§5.2)
# ----------------------------------------------------------------------------

def build_graph(layer_sizes, density=0.5, seed=0):
    """Layered sparse DAG. Each neuron connects to a random subset of the
    previous layer at ``density``, with guaranteed min fan-in >= 2 and
    min fan-out >= 1 (pure random sampling strands neurons at high sparsity)."""
    net = Network(layer_sizes)
    rng = np.random.default_rng(seed)

    for layer_idx in range(1, len(net.layers)):
        prev = net.layers[layer_idx - 1]
        cur = net.layers[layer_idx]
        k = max(2, min(len(prev), int(round(density * len(prev)))))
        for post in cur:
            sources = rng.choice(prev, size=k, replace=False)
            for pre in sources:
                net.add_synapse(int(pre), int(post))
        # guarantee every neuron in the previous layer has >= 1 outgoing edge
        for pre in prev:
            if not net.outgoing[pre]:
                post = int(rng.choice(cur))
                net.add_synapse(int(pre), post)

    return net


def init_weights(net, seed=0):
    """He init using the *real* sparse in-degree per neuron (§5.2).

    Masking a dense He init would assume ``n`` inputs while only ``k`` remain,
    collapsing per-layer variance. We scale to the actual ``k_j`` instead.
    """
    rng = np.random.default_rng(seed)
    for nid, neuron in enumerate(net.neurons):
        if neuron.layer == 0:
            continue
        k = len(net.incoming[nid])
        if k == 0:
            continue
        std = np.sqrt(2.0 / k)
        for pre in net.incoming[nid]:
            net.synapses[(pre, nid)].weight = float(rng.normal(0.0, std))
