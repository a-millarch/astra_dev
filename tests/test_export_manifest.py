"""Export/validate round-trip on synthetic tiny artifacts.

Runs entirely on generated objects — no real data, models or calibrators are
needed (EBM/calibrator paths are simply absent from the tiny artifacts and
must be skipped gracefully).
"""

import hashlib
import json
import shutil

import pytest

from astra.inference.export_artifacts import (
    run_export,
    run_validate,
    write_minimal_support_files,
)
from astra.inference.synthetic import save_tiny_artifacts

MODEL_NAME = 'tinytest'


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _export_tiny(root, model_name=MODEL_NAME, **bundle_kwargs):
    """Generate tiny artifacts under *root* and export them to a handoff dir."""
    artifacts = save_tiny_artifacts(str(root / 'artifacts'),
                                    model_name=model_name, **bundle_kwargs)
    support = write_minimal_support_files(root / 'support')
    out_dir = root / 'handoff'
    manifest = run_export(
        model_name,
        artifacts_dir=artifacts['artifacts_dir'],
        out_dir=str(out_dir),
        config_path=support['config'],
        metadata_csv=support['metadata_csv'],
        handoff_doc=str(root / 'HANDOFF.md'),  # absent → skipped gracefully
    )
    return out_dir, manifest


@pytest.fixture(scope='module')
def exported(tmp_path_factory):
    """One pristine exported handoff dir shared by the read-only tests.

    Mutation tests copy it into their own tmp_path first.
    """
    root = tmp_path_factory.mktemp('export_roundtrip')
    out_dir, manifest = _export_tiny(root)
    return {'out_dir': out_dir, 'manifest': manifest}


def test_export_creates_manifest_and_hashes(exported):
    out_dir = exported['out_dir']
    manifest_path = out_dir / 'manifest.json'
    assert manifest_path.is_file(), 'export must write manifest.json'

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['schema_version'] == 1
    assert manifest['model_name'] == MODEL_NAME
    assert manifest['files'], 'manifest must list exported files'

    # Every listed file exists with matching size and recomputed sha256.
    for entry in manifest['files']:
        p = out_dir / entry['path']
        assert p.is_file(), f"listed file missing: {entry['path']}"
        assert p.stat().st_size == entry['bytes'], entry['path']
        assert _sha256(p) == entry['sha256'], f"hash mismatch: {entry['path']}"
        assert '\\' not in entry['path'], 'manifest paths must use forward slashes'

    # Core artifacts must be listed.
    listed = {e['path'] for e in manifest['files']}
    assert f'models/{MODEL_NAME}.pth' in listed
    assert f'models/deployment/deployment_{MODEL_NAME}.pkl' in listed
    assert 'examples/synthetic_patient.json' in listed
    assert 'data/external/metadata.csv' in listed

    # Data-protection gate: sign_off must start out as null.
    assert manifest['shap_background']['sign_off'] is None
    assert manifest['shap_background']['n_samples'] > 0

    # Tiny artifacts have no calibrators / EBM — recorded as absent, not error.
    assert manifest['calibration'] is None
    assert manifest['ebm'] == {'included': False, 'files': []}


def test_validate_passes_on_intact_bundle(exported):
    # Full acceptance path: hash check → model load → synthetic forward pass.
    rc = run_validate(str(exported['out_dir']))
    assert rc == 0


def test_predictor_loads_from_handoff_bundle_root(exported):
    """Regression (Azure 2026-07-11): the HANDOFF quickstart points
    AstraPredictor.load at the exported bundle root, whose artifacts live
    under <root>/models/ — the facade must resolve the nested layout instead
    of looking for <root>/deployment/."""
    from astra.inference.api import AstraPredictor

    predictor = AstraPredictor.load(
        MODEL_NAME, artifacts_dir=str(exported['out_dir']), device='cpu')
    info = predictor.model_info()
    assert info['model_name'] == MODEL_NAME
    assert info['seq_len'] == predictor.seq_len
    assert len(info['channels']) > 0


def test_validate_fails_on_corruption(exported, tmp_path):
    corrupted = tmp_path / 'handoff_corrupt'
    shutil.copytree(exported['out_dir'], corrupted)

    pth = corrupted / 'models' / f'{MODEL_NAME}.pth'
    blob = bytearray(pth.read_bytes())
    blob[len(blob) // 2] ^= 0xFF  # flip one byte
    pth.write_bytes(bytes(blob))

    rc = run_validate(str(corrupted))
    assert rc != 0, 'byte-flipped weights must fail the hash check'


def test_validate_fails_on_missing_file(exported, tmp_path):
    broken = tmp_path / 'handoff_missing'
    shutil.copytree(exported['out_dir'], broken)
    (broken / 'examples' / 'synthetic_patient.json').unlink()

    rc = run_validate(str(broken))
    assert rc != 0, 'a deleted manifest-listed file must fail validation'


def test_validate_temporal_bundle(tmp_path):
    # Temporal-head model: validate must also check the per-timestep curve.
    out_dir, _ = _export_tiny(tmp_path, model_name='tinytemp', temporal_head=True)
    rc = run_validate(str(out_dir))
    assert rc == 0


class TestModelNameFromConfig:
    """Config-first CLI: model_name resolvable from a config YAML."""

    def test_resolves_from_yaml(self, tmp_path):
        from astra.inference.export_artifacts import _model_name_from_config
        cfg = tmp_path / "exp.yaml"
        cfg.write_text("model_name: exp_model\nother: 1\n", encoding="utf-8")
        assert _model_name_from_config(str(cfg)) == "exp_model"

    def test_missing_key_exits(self, tmp_path):
        import pytest
        from astra.inference.export_artifacts import _model_name_from_config
        cfg = tmp_path / "empty.yaml"
        cfg.write_text("other: 1\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            _model_name_from_config(str(cfg))

    def test_unreadable_exits(self, tmp_path):
        import pytest
        from astra.inference.export_artifacts import _model_name_from_config
        with pytest.raises(SystemExit):
            _model_name_from_config(str(tmp_path / "missing.yaml"))
