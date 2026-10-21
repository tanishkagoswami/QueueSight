"""On startup QueueSight only simulates and trains what is missing, so a restart takes seconds."""
import pytest
from scripts.prepare import DATA_FILES, MODEL_FILES, steps_needed


def touch(folder, names):
    for n in names:
        (folder / n).parent.mkdir(parents=True, exist_ok=True)
        (folder / n).write_text("x")


@pytest.fixture
def dirs(tmp_path):
    return tmp_path / "data", tmp_path / "artifacts"


def test_first_start_of_the_simulated_hospital_simulates_then_trains(dirs):
    assert steps_needed("sim", *dirs) == ["simulate", "train"]


def test_restart_with_everything_ready_does_nothing(dirs):
    touch(dirs[0], DATA_FILES)
    touch(dirs[1], MODEL_FILES)
    assert steps_needed("sim", *dirs) == []
    assert steps_needed("datathon", *dirs) == []


def test_data_without_models_only_trains(dirs):
    touch(dirs[0], DATA_FILES)
    assert steps_needed("sim", *dirs) == ["train"]


def test_new_simulated_data_always_retrains_even_if_old_models_exist(dirs):
    touch(dirs[1], MODEL_FILES)
    assert steps_needed("sim", *dirs) == ["simulate", "train"]


def test_missing_datathon_csvs_stop_with_a_clear_message(dirs):
    with pytest.raises(SystemExit, match="DATATHON CSVs"):
        steps_needed("datathon", *dirs)