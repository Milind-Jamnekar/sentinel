from ml.train import train

def test_training():
    final = train()
    assert final < 0.01

def test_training_with_reproducible():
    assert train(steps=5) == train(steps=5)

def test_zero_steps_does_not_crash():
    assert train(steps=0) == float("inf")