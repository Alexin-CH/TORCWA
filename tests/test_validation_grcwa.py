"""Fast subset of the external validation, skipped if grcwa is not installed."""

import pytest

grcwa = pytest.importorskip("grcwa")

from validation.validate_grcwa import case_grating, case_slab


def test_slab_normal_analytic_and_grcwa():
    assert case_slab(0.0, "p", "pytest-slab-normal")


def test_slab_oblique_p():
    assert case_slab(30.0, "p", "pytest-slab-oblique-p")


def test_slab_oblique_s():
    assert case_slab(30.0, "s", "pytest-slab-oblique-s")


def test_grating_normal_per_order():
    assert case_grating(0.0, "pytest-grating-normal")
