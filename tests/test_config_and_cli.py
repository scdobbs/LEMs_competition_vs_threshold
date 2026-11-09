"""Config validation, unit-loader refusal, run recording and a CLI smoke test on a
100 x 100 synthetic tile (directive Phase 1, requirement 6)."""

import json

import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from lemcvt.config import ConfigError, load_site, load_tile, load_tile_config
from synthetic_tile import make_synthetic_tile, restore_env, set_env


@pytest.fixture
def tile_env(tmp_path):
    info = make_synthetic_tile(tmp_path, tile="GM1", site="GM")
    prev = set_env(info)
    yield info
    restore_env(prev)


@pytest.fixture
def feet_tile_env(tmp_path):
    info = make_synthetic_tile(tmp_path, tile="AP1", site="AP", z_units="ftUS", reproduce_v1_units=False)
    prev = set_env(info)
    yield info
    restore_env(prev)


class TestConfig:

    def test_site_and_tile_load(self, tile_env):
        site = load_site("GM")
        assert site.U == 1.47e-4
        cfg = load_tile_config("GM1")
        assert cfg.shape == (100, 100)
        assert cfg.model.K0_ad == pytest.approx(1.47e-4 / 5.0)
        assert cfg.model.D0 == pytest.approx(1.47e-4 / 1.5e-2)
        assert cfg.model.K_bounds == (7e-7, 2e-2)

    def test_unknown_tile(self, tile_env):
        with pytest.raises(ConfigError):
            load_tile_config("GM9")

    def test_bad_bounds_rejected(self, tile_env):
        p = tile_env["configs"] / "tiles" / "GM1.yaml"
        d = yaml.safe_load(p.read_text())
        d["model"]["K_bounds"] = [1e-2, 1e-4]
        p.write_text(yaml.safe_dump(d))
        with pytest.raises(ConfigError, match="K_bounds"):
            load_tile_config("GM1")

    def test_initial_outside_bounds_rejected(self, tile_env):
        p = tile_env["configs"] / "tiles" / "GM1.yaml"
        d = yaml.safe_load(p.read_text())
        d["model"]["A_crit_0"] = 1e7
        p.write_text(yaml.safe_dump(d))
        with pytest.raises(ConfigError, match="outside bounds"):
            load_tile_config("GM1")

    def test_missing_key_rejected(self, tile_env):
        p = tile_env["configs"] / "tiles" / "GM1.yaml"
        d = yaml.safe_load(p.read_text())
        del d["units"]["z_units"]
        p.write_text(yaml.safe_dump(d))
        with pytest.raises(ConfigError, match="missing keys"):
            load_tile_config("GM1")


class TestLoader:

    def test_loads_and_verifies(self, tile_env):
        td = load_tile("GM1")
        np.testing.assert_array_equal(td.Z, tile_env["zf"])
        np.testing.assert_array_equal(td.FD, tile_env["fd"])
        assert td.FD.dtype == np.uint8
        assert td.dx == 1.0
        assert td.z_factor == 1.0
        assert td.sanity["median_slope"] < 0.6

    def test_sha_mismatch_refused(self, tile_env):
        p = tile_env["configs"] / "tiles" / "GM1.yaml"
        d = yaml.safe_load(p.read_text())
        d["grids"]["sha256"]["area"] = "0" * 64
        p.write_text(yaml.safe_dump(d))
        with pytest.raises(ConfigError, match="sha256 mismatch"):
            load_tile("GM1")

    def test_feet_converted_to_metres(self, feet_tile_env):
        td = load_tile("AP1")
        assert td.z_units_working == "m"
        assert td.z_factor == pytest.approx(0.3048006, rel=1e-6)
        assert 200 < np.nanmin(td.Z) < np.nanmax(td.Z) < 600

    def test_wrong_declared_units_refused(self, feet_tile_env):
        """Feet stored but declared as metres: the sanity check must refuse."""
        p = feet_tile_env["configs"] / "tiles" / "AP1.yaml"
        d = yaml.safe_load(p.read_text())
        d["units"]["z_units"] = "m"
        p.write_text(yaml.safe_dump(d))
        with pytest.raises(Exception, match="declared units disagree"):
            load_tile("AP1")

    def test_reproduce_v1_units_keeps_feet(self, feet_tile_env, caplog):
        p = feet_tile_env["configs"] / "tiles" / "AP1.yaml"
        d = yaml.safe_load(p.read_text())
        d["units"]["reproduce_v1_units"] = True
        p.write_text(yaml.safe_dump(d))
        with caplog.at_level("WARNING"):
            td = load_tile("AP1")
        assert td.z_factor == 1.0
        assert td.z_units_working == "ftUS"
        assert np.nanmax(td.Z) > 900
        assert any("NOT converted" in r.message for r in caplog.records)


class TestCLISmoke:

    def test_tile_run_and_forward(self, tile_env):
        from lemcvt.cli import main

        runner = CliRunner()
        r = runner.invoke(main, ["tile", "run", "--tile", "GM1", "--stages", "ad,acrit,nonlinear,joint",
                                 "--crop", "0,60,0,60", "--no-derived", "--run-id", "smoke_run"])
        assert r.exit_code == 0, r.output
        run_dir = tile_env["results"] / "smoke_run"
        params = json.loads((run_dir / "params.json").read_text())
        assert set(params["stages_results"]) == {"ad", "acrit", "nonlinear", "joint"}
        for st in ("ad", "acrit", "nonlinear", "joint"):
            assert np.isfinite(params["stages_results"][st]["mse"])
        run = json.loads((run_dir / "run.json").read_text())
        assert run["git_hash"] and "config" in run and run["config"]["tile"]["tile"] == "GM1"
        assert (run_dir / "trajectory.parquet").exists() or (run_dir / "trajectory.csv").exists()
        tifs = sorted(p.name for p in run_dir.glob("*.tif"))
        assert "GM_1m_best_tile_AD-opt_elevation.tif" in tifs
        assert "GM_1m_best_tile_Partitioned-model-a_crit-k-d-opt_elevation.tif" in tifs

        r2 = runner.invoke(main, ["tile", "forward", "--tile", "GM1", "--params", str(run_dir / "params.json"),
                                  "--crop", "0,60,0,60", "--run-id", "smoke_fwd"])
        assert r2.exit_code == 0, r2.output
        fwd = json.loads((tile_env["results"] / "smoke_fwd" / "params.json").read_text())
        keys = {fr["key"]: fr for fr in fwd["forward_runs"]}
        assert keys["joint_opt"]["mse"] == pytest.approx(params["stages_results"]["joint"]["mse"], rel=1e-12)

    def test_data_verify_and_info(self, tile_env):
        from lemcvt.cli import main

        runner = CliRunner()
        r = runner.invoke(main, ["data", "manifest"])
        assert r.exit_code == 0, r.output
        r = runner.invoke(main, ["info"])
        assert r.exit_code == 0 and "LEMCVT_DATA" in r.output

    def test_slurm_dry_run(self, tile_env):
        from lemcvt.cli import main

        runner = CliRunner()
        r = runner.invoke(main, ["slurm", "submit", "tile_run", "--tiles", "GM1", "--dry-run"])
        assert r.exit_code == 0, r.output
        assert "#SBATCH --time=02:00:00" in r.output and "--array=0-0" in r.output
        assert "lemcvt -v tile run --tile $LABEL" in r.output
