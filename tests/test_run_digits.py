from run_digits import train_digits


def test_train_digits_beats_chance_quickly():
    res = train_digits(steps=1500, seed=0, layers=(64, 32, 16, 10),
                       density=0.4)
    assert res["test_acc"] > 0.30          # chance = 0.10; 1.5k steps clears it
    assert res["synapses"] < 64 * 32 + 32 * 16 + 16 * 10   # sparser than dense
    assert res["edge_steps"] > 0 and res["wall_time"] >= 0.0
