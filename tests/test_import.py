def test_import_nano_train() -> None:
    import nanotrain

    assert nanotrain.__version__ == "0.1.0"


def test_import_training_engine() -> None:
    from nanotrain.runtime import DataBuilder, ModelBuilder, OptimizerBuilder, TrainingEngine

    assert DataBuilder.__name__ == "DataBuilder"
    assert ModelBuilder.__name__ == "ModelBuilder"
    assert OptimizerBuilder.__name__ == "OptimizerBuilder"
    assert TrainingEngine.__name__ == "TrainingEngine"
