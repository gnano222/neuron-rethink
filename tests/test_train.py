import numpy as np

from sprout.data import generate_blobs
from sprout.network import build_graph, init_weights
from sprout.train import Config, Trainer, accuracy, predict


def test_accuracy_matches_manual_argmax():
    net = build_graph([2, 6, 4, 2], density=0.6, seed=0)
    init_weights(net, seed=0)
    X, y = generate_blobs(n=100, seed=0)
    preds = predict(net, X)
    assert accuracy(net, X, y) == (preds == y).mean()


def test_deep_signal_alive_after_init():
    # §5.2 / pitfall: with proper per-neuron fan-in scaling, signal must still
    # be alive in the *last* hidden layer, not collapsed to ~0.
    net = build_graph([2, 8, 8, 6, 2], density=0.5, seed=0)
    init_weights(net, seed=0)
    X, _ = generate_blobs(n=200, seed=0)
    last_hidden = net.layers[-2]
    acts = []
    for x in X:
        net.forward(x)
        acts.append(np.mean([abs(net.neurons[n].activation) for n in last_hidden]))
    assert np.mean(acts) > 1e-3  # signal survived to depth


def test_backprop_only_learns_blobs():
    # §10 step 1 success criterion: accuracy climbs on the easy task.
    net = build_graph([2, 8, 8, 6, 2], density=0.5, seed=0)
    init_weights(net, seed=0)
    X, y = generate_blobs(n=400, seed=0)
    cfg = Config()  # all advanced mechanisms off by default => plain backprop
    trainer = Trainer(cfg, net, X, y, seed=0)
    for _ in range(3000):
        trainer.step()
    assert accuracy(net, X, y) > 0.9


def test_default_config_has_advanced_mechanisms_off():
    cfg = Config()
    assert cfg.enable_eligibility is False
    assert cfg.enable_confidence is False
    assert cfg.enable_prune is False
    assert cfg.enable_grow is False
    assert cfg.enable_homeostasis is False


def test_default_config_softened_cliff_knobs():
    # the calibrated 2D confidence rule with the smooth (sigmoid) settled cliff is
    # the DEFAULT currency confidence — the promoted baseline architecture
    cfg = Config()
    assert cfg.confidence_mode == "twod"
    assert cfg.settled_mode == "sigmoid"
    assert cfg.conf_k == 3.0


def test_default_config_ghost_meter_off():
    # A2 (grow on a persistent EMA of the virtual gradient) is OPT-IN, so the
    # promoted baseline currency keeps its instantaneous-batch growth unchanged.
    cfg = Config()
    assert cfg.ghost_meter is False
    assert cfg.beta_ghost == 0.8


def test_default_config_uses_selective_grow_bar():
    # B1 promoted: the selective hiring bar (grow_bar_frac=3.0) is the default,
    # not the prior eager 1.5 — it fixes the grow<->prune oscillation at source.
    assert Config().grow_bar_frac == 3.0


def test_default_config_has_no_init_density_override():
    # init_density is an eval-harness build hint, not a training knob: None means
    # "use the suite's --density". The Trainer never reads it; it only tells the
    # eval runner how densely to wire the *initial* graph, letting a variant pin
    # its own connectivity (e.g. a fully-connected control).
    assert Config().init_density is None


def test_default_config_has_no_init_layers_override():
    # init_layers is the sibling build hint to init_density: None means "use the
    # suite's --layers". The Trainer never reads it; it only lets a variant pin
    # its own neuron counts (e.g. a width-sweep arm) within one suite.
    assert Config().init_layers is None


def test_prune_warmup_delays_pruning():
    # No synapse should be pruned before prune_warmup, even with prune enabled.
    net = build_graph([2, 8, 8, 6, 2], density=0.5, seed=0)
    init_weights(net, seed=0)
    X, y = generate_blobs(n=200, seed=0)
    cfg = Config(enable_prune=True, prune_warmup=1000, t_struct=100,
                 t_grace=50, theta_prune=10.0)  # huge theta => everything prunable
    trainer = Trainer(cfg, net, X, y, seed=0)
    for _ in range(900):
        trainer.step()
    assert all(e["type"] != "prune" for e in trainer.events)  # nothing pruned yet
    for _ in range(300):  # cross the warmup
        trainer.step()
    assert any(e["type"] == "prune" for e in trainer.events)   # now it prunes


def test_trainer_records_history():
    net = build_graph([2, 8, 8, 6, 2], density=0.5, seed=0)
    init_weights(net, seed=0)
    X, y = generate_blobs(n=200, seed=0)
    trainer = Trainer(Config(), net, X, y, seed=0)
    for _ in range(50):
        trainer.step()
    assert len(trainer.history["step"]) > 0
    assert len(trainer.history["synapse_count"]) == len(trainer.history["step"])
