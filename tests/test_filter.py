from __future__ import annotations

import numpy as np

from pfvol import SVParams, particle_filter, simulate
from pfvol.benchmarks import rolling_std
from pfvol.filter import normalize_log_weights, systematic_resample, weighted_quantiles


def test_weights_normalize() -> None:
    rng = np.random.default_rng(0)
    for scale in (1.0, 50.0, 1000.0):  # large log-weights must not overflow
        w, _ = normalize_log_weights(rng.normal(0, scale, size=500))
        assert np.isclose(w.sum(), 1.0)
        assert (w >= 0).all()


def test_log_weight_normalizer_matches_direct_sum() -> None:
    logw = np.log(np.array([0.1, 0.2, 0.3]))
    _, lse = normalize_log_weights(logw)
    assert np.isclose(lse, np.log(0.6))


def test_resampling_preserves_particle_count() -> None:
    rng = np.random.default_rng(1)
    for n in (10, 100, 1000):
        w = rng.random(n)
        w /= w.sum()
        idx = systematic_resample(w, rng)
        assert len(idx) == n
        assert idx.min() >= 0 and idx.max() < n


def test_systematic_resampling_counts() -> None:
    w = np.array([0.5, 0.25, 0.25, 0.0])
    counts = np.bincount(systematic_resample(w, np.random.default_rng(2)), minlength=4)
    assert counts.tolist() == [2, 1, 1, 0]  # N * w is integral, so systematic gives it exactly


def test_weighted_quantiles_are_ordered_and_sensible() -> None:
    x = np.linspace(0, 1, 1001)
    w = np.full(1001, 1 / 1001)
    q = weighted_quantiles(x, w, (0.05, 0.5, 0.95))
    assert np.all(np.diff(q) > 0)
    assert np.allclose(q, [0.05, 0.5, 0.95], atol=0.01)


def test_same_seed_same_output() -> None:
    r = simulate(150, seed=3).returns
    a = particle_filter(r, n_particles=300, seed=7)
    b = particle_filter(r, n_particles=300, seed=7)
    assert np.array_equal(a.mean_vol, b.mean_vol)
    assert np.array_equal(a.quantiles, b.quantiles)
    c = particle_filter(r, n_particles=300, seed=8)
    assert not np.array_equal(a.mean_vol, c.mean_vol)


def test_output_shapes_and_ordering() -> None:
    r = simulate(120, seed=4).returns
    f = particle_filter(r, n_particles=400, seed=0)
    assert f.mean_vol.shape == f.ess.shape == f.pred_var.shape == (120,)
    assert f.quantiles.shape == (120, 5)
    assert np.all(np.diff(f.quantiles, axis=1) >= 0)
    assert np.all((f.ess > 0) & (f.ess <= 400 + 1e-6))
    assert np.all(f.ess[f.resampled] < 200)  # resample only below N/2


def test_true_parameters_score_higher_loglik_than_wrong_ones() -> None:
    path = simulate(600, seed=5)
    good = np.mean([particle_filter(path.returns, n_particles=1000, seed=s).loglik for s in range(4)])
    bad = np.mean(
        [particle_filter(path.returns, SVParams(phi=0.5, sigma_eta=0.6), 1000, seed=s).loglik
         for s in range(4)]
    )
    assert good > bad


def test_error_decreases_as_n_grows() -> None:
    # Monte Carlo error: distance from a big-N reference run of the same data
    r = simulate(150, seed=6).returns
    ref = particle_filter(r, n_particles=40_000, seed=999, compute_quantiles=False).mean_vol
    errs = []
    for n in (50, 400, 3200):
        e = [
            np.sqrt(np.mean((particle_filter(r, n_particles=n, seed=s, compute_quantiles=False)
                             .mean_vol - ref) ** 2))
            for s in range(12)
        ]
        errs.append(np.mean(e))
    assert errs[0] > errs[1] > errs[2]


def test_filter_beats_rolling_std_after_a_moderate_shock() -> None:
    # Paired comparison over many simulated paths with a +0.5 jump in log-variance.
    # Not claimed for huge jumps (+1.5 and up): the filter's prior makes those
    # nearly impossible, so it adapts slowly (see docs/results/shock_response.json).
    day, size, n_paths = 150, 0.5, 40
    window = slice(day, day + 15)
    pf_err, roll_err = [], []
    for seed in range(n_paths):
        path = simulate(250, seed=2000 + seed, shocks={day: size})
        pf = particle_filter(path.returns, n_particles=1000, seed=seed, compute_quantiles=False)
        roll = rolling_std(path.returns, 20)
        pf_err.append(np.mean(np.abs(pf.mean_vol[window] - path.vol[window])))
        roll_err.append(np.mean(np.abs(roll[window] - path.vol[window])))
    pf_err, roll_err = np.array(pf_err), np.array(roll_err)
    assert pf_err.mean() < roll_err.mean()
    assert np.mean(pf_err < roll_err) > 0.5  # and it wins on most individual paths


# ---- variants, estimation helpers -------------------------------------------------
def test_student_t_logpdf_is_a_density_with_unit_variance() -> None:
    from pfvol.filter import student_t_logpdf

    xs = np.linspace(-60, 60, 240_001)
    h = np.array([0.0])
    dens = np.array([np.exp(student_t_logpdf(x, h, 5.0))[0] for x in xs[::40]])
    dx = (xs[1] - xs[0]) * 40
    assert abs(dens.sum() * dx - 1.0) < 1e-3
    assert abs((dens * xs[::40] ** 2).sum() * dx - 1.0) < 2e-2  # variance 1


def test_t_likelihood_tends_to_gaussian_for_large_dof() -> None:
    from pfvol.filter import gaussian_logpdf, student_t_logpdf

    h = np.array([-1.0, 0.5])
    assert np.allclose(student_t_logpdf(0.7, h, 1e6), gaussian_logpdf(0.7, h), atol=1e-4)


def test_variants_run_reproducibly_and_report_sensible_output() -> None:
    from pfvol import JumpParams

    r = simulate(200, seed=11).returns
    for kw in ({"obs_dof": 6.0}, {"jump": JumpParams()}):
        a = particle_filter(r, n_particles=400, seed=2, **kw)  # type: ignore[arg-type]
        b = particle_filter(r, n_particles=400, seed=2, **kw)  # type: ignore[arg-type]
        assert np.array_equal(a.mean_vol, b.mean_vol)
        assert np.isfinite(a.loglik) and np.isclose(a.loglik, a.loglik_t.sum())
        assert np.all(a.mean_vol > 0)


def test_t_filter_has_better_loglik_on_heavy_tailed_returns() -> None:
    rng = np.random.default_rng(0)
    r = 0.01 * rng.standard_t(4, size=1500) / np.sqrt(2.0)  # unit-variance t(4) noise, constant vol
    base = np.mean([particle_filter(r, n_particles=600, seed=s).loglik for s in range(3)])
    robust = np.mean([particle_filter(r, n_particles=600, seed=s, obs_dof=4.0).loglik
                      for s in range(3)])
    assert robust > base


def test_prior_mean_is_a_forecast_not_the_posterior() -> None:
    r = simulate(120, seed=12).returns
    f = particle_filter(r, n_particles=500, seed=0)
    assert f.prior_mean.shape == f.mean_vol.shape and np.all(f.prior_mean > 0)
    assert not np.allclose(f.prior_mean, f.mean_vol)


def test_garch_recursion_matches_hand_calculation() -> None:
    from pfvol.benchmarks import garch_next_vol

    r = np.array([0.01, -0.02, 0.0])
    omega, alpha, beta = 1e-6, 0.1, 0.8
    v = omega / (1 - alpha - beta)
    expected = []
    for x in r:
        v = omega + alpha * x * x + beta * v
        expected.append(np.sqrt(v))
    assert np.allclose(garch_next_vol(r, omega, alpha, beta), expected)


def test_diebold_mariano_detects_a_real_difference_and_not_a_null() -> None:
    from pfvol.metrics import diebold_mariano

    rng = np.random.default_rng(3)
    base = rng.normal(size=800)
    mean_d, z, p = diebold_mariano(base + 0.25, base + rng.normal(0, 0.3, 800))
    assert mean_d > 0 and z > 0 and p < 0.01  # A has higher loss
    _, _, p_null = diebold_mariano(base, base + rng.normal(0, 0.3, 800))
    assert p_null > 0.01
