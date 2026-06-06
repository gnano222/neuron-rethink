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
    assert cfg.enable_confidence is False
    assert cfg.enable_prune is False
    assert cfg.enable_grow is False


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


def test_default_config_has_no_grow_demand_k():
    # grow_demand_k=None => exact-sparse grow scan over all active posts (the
    # bit-identical default); an int k restricts to the top-k highest-|delta|.
    assert Config().grow_demand_k is None


def test_default_config_has_sleep_on_floor1_nocap():
    # PROMOTED DEFAULT: settledness-gated sleep consolidation is now ON by
    # default at floor 1.0 with NO per-burst cap (sleep_max_prune=None) — the
    # floor-0-to-2 sweep found this the deepest preserved-accuracy operating
    # point (~-46% synapses, acc preserved). validate.py and the eval `currency`
    # baseline are pinned no-sleep as stable references.
    cfg = Config()
    assert cfg.enable_sleep is True
    assert cfg.sleep_prune_floor == 1.0
    assert cfg.sleep_max_prune is None          # None = no cap (prune all eligible)
    # detector knobs unchanged from the measured loss-settling trace
    assert cfg.sleep_warmup == 2500
    assert cfg.sleep_loss_beta == 0.01
    assert cfg.sleep_loss_tol == 0.03
    assert cfg.sleep_patience == 1500


def test_sleep_no_cap_prunes_more_than_wake_cap_in_one_burst():
    # sleep_max_prune=None => a consolidation burst removes EVERY eligible wire,
    # not a fixed few — so a single burst exceeds the gentle wake cap (max_prune).
    from collections import Counter
    from sprout.data import generate_spirals
    X, y = generate_spirals(n=400, seed=0)
    net = build_graph([2, 12, 12, 8, 2], density=0.6, seed=0)
    init_weights(net, seed=0)
    cfg = Config(eta_base=0.02, grad_currency=True, enable_confidence=True,
                 enable_prune=True, enable_grow=True, gamma_dec=0.001, t_struct=100,
                 enable_sleep=True, sleep_warmup=500, sleep_patience=300,
                 sleep_loss_tol=0.05, sleep_prune_floor=1.0, sleep_max_prune=None)
    tr = Trainer(cfg, net, X, y, seed=0)
    for _ in range(3000):
        tr.step()
    sleep_steps = [e["step"] for e in tr.events if e["type"] == "sleep"]
    prune_by_step = Counter(e["step"] for e in tr.events if e["type"] == "prune")
    bursts = [prune_by_step.get(s, 0) for s in sleep_steps]
    assert sleep_steps                          # it slept
    assert max(bursts) > cfg.max_prune          # uncapped: a burst beat the wake cap (2)


def test_sleep_consolidation_fires_and_sparsifies():
    # An aggressive-sleep trainer should (a) fire at least one consolidation burst
    # once settled and (b) end sparser than an otherwise-identical no-sleep twin.
    from sprout.data import generate_spirals
    base = dict(eta_base=0.02, grad_currency=True, enable_confidence=True,
                enable_prune=True, enable_grow=True, gamma_dec=0.001, t_struct=100)
    X, y = generate_spirals(n=400, seed=0)

    def run(sleep):
        net = build_graph([2, 12, 12, 8, 2], density=0.5, seed=0)
        init_weights(net, seed=0)
        cfg = Config(**base, enable_sleep=sleep, sleep_warmup=500,
                     sleep_patience=300, sleep_loss_tol=0.05, sleep_max_prune=8)
        tr = Trainer(cfg, net, X, y, seed=0)
        for _ in range(4000):
            tr.step()
        return tr, len(net.synapses)

    tr_sleep, n_sleep = run(True)
    _, n_wake = run(False)
    assert any(e["type"] == "sleep" for e in tr_sleep.events)   # it slept
    assert n_sleep < n_wake                                      # and ended sparser


def test_sleep_off_leaves_no_sleep_events():
    # the default (sleep off) path must never record a sleep event.
    from sprout.data import generate_spirals
    net = build_graph([2, 10, 10, 8, 2], density=0.5, seed=0)
    init_weights(net, seed=0)
    X, y = generate_spirals(n=300, seed=0)
    cfg = Config(eta_base=0.02, grad_currency=True, enable_confidence=True,
                 enable_prune=True, enable_grow=True, gamma_dec=0.001, t_struct=100,
                 enable_sleep=False)          # explicit off (default is now on)
    tr = Trainer(cfg, net, X, y, seed=0)
    for _ in range(2000):
        tr.step()
    assert not any(e["type"] == "sleep" for e in tr.events)


def test_trainer_records_history():
    net = build_graph([2, 8, 8, 6, 2], density=0.5, seed=0)
    init_weights(net, seed=0)
    X, y = generate_blobs(n=200, seed=0)
    trainer = Trainer(Config(), net, X, y, seed=0)
    for _ in range(50):
        trainer.step()
    assert len(trainer.history["step"]) > 0
    assert len(trainer.history["synapse_count"]) == len(trainer.history["step"])
