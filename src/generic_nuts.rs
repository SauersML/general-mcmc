//! Backend-agnostic No-U-Turn Sampler (NUTS) core.

use crate::euclidean::EuclideanVector;
use crate::generic_hmc::HamiltonianTarget;
use crate::stats::{
    ChainStats, ChainTracker, RunStats, collect_rhat, max_skipnan, split_rhat_mean_ess,
};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use ndarray::{Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis, s};
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use rand::distr::Distribution as RandDistribution;
// rand_distr provides the distributions, but we rely on rand's Distribution trait for compatibility.
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Exp1, StandardNormal, StandardUniform};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

/// Backend-agnostic No-U-Turn Sampler (NUTS) spanning multiple chains.
pub struct GenericNUTS<V, Target>
where
    V: EuclideanVector,
    Target: HamiltonianTarget<V>,
{
    chains: Vec<GenericNUTSChain<V, Target>>,
}

type RunResult<T> = Result<(Array3<T>, RunStats), Box<dyn Error>>;

/// Shortest open-warmup window. Split R-hat and ESS halve every chain, and each
/// half needs two draws before a within-half variance exists.
const MIN_WARMUP_WINDOW: usize = 4;

/// When the chains of [`GenericNUTS::run_adaptive`] count as mixing: in one
/// warmup window, every coordinate's split R-hat is below `max_rhat` and its
/// ESS is above `min_ess`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MixingTargets {
    pub max_rhat: f64,
    pub min_ess: f64,
}

/// What an open-ended warmup spent before the sampler collected draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaptiveWarmup {
    /// Warmup transitions per chain.
    pub transitions: usize,
    /// Doubling windows run.
    pub windows: usize,
}

/// Why [`GenericNUTS::run_adaptive`] refused to collect draws.
#[derive(Clone, Debug, PartialEq)]
pub enum AdaptiveWarmupError {
    /// No chains, no requested draws, or targets that no window can meet.
    InvalidRequest(String),
    /// A window's split R-hat or ESS was not a number, so no test on it decides
    /// anything.
    NonFiniteDiagnostics { windows: usize, transitions: usize },
    /// Step size and metric had stabilized and the window met the ESS target,
    /// yet split R-hat stayed at or above its target in two consecutive windows:
    /// the chains sample different regions, and a longer warmup cannot join them.
    ChainsDisagree {
        windows: usize,
        transitions: usize,
        max_rhat: f64,
    },
    /// The chains agreed and met the ESS target, yet one coordinate's posterior
    /// variance moved beyond its sampling error in the same direction in two
    /// consecutive windows: the variance does not settle as the window grows.
    VarianceDoesNotSettle {
        windows: usize,
        transitions: usize,
        coordinate: usize,
    },
    /// Doubling the window again would overflow `usize`.
    WindowOverflow { windows: usize, transitions: usize },
}

impl fmt::Display for AdaptiveWarmupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(reason) => write!(f, "invalid open warmup request: {reason}"),
            Self::NonFiniteDiagnostics {
                windows,
                transitions,
            } => write!(
                f,
                "warmup window {windows} (after {transitions} transitions per chain) produced a split R-hat or ESS that is not a number"
            ),
            Self::ChainsDisagree {
                windows,
                transitions,
                max_rhat,
            } => write!(
                f,
                "chains disagree after {transitions} warmup transitions per chain ({windows} windows): split R-hat {max_rhat} stayed at or above its target in two consecutive windows whose step size and metric had stabilized"
            ),
            Self::VarianceDoesNotSettle {
                windows,
                transitions,
                coordinate,
            } => write!(
                f,
                "the posterior variance of coordinate {coordinate} kept moving the same way in two consecutive warmup windows while the chains agreed, after {transitions} transitions per chain ({windows} windows)"
            ),
            Self::WindowOverflow {
                windows,
                transitions,
            } => write!(
                f,
                "warmup window {windows} cannot double again after {transitions} transitions per chain"
            ),
        }
    }
}

impl Error for AdaptiveWarmupError {}

/// Mass-matrix adaptation strategy for warmup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MassMatrixAdaptation {
    None,
    Diagonal,
    Dense,
}

/// Controls warmup-time mass-matrix adaptation for NUTS.
#[derive(Clone, Debug)]
pub struct NUTSMassMatrixConfig {
    pub adaptation: MassMatrixAdaptation,
    pub start_buffer: usize,
    pub end_buffer: usize,
    pub initial_window: usize,
    pub regularize: f64,
    pub jitter: f64,
    pub dense_max_dim: usize,
}

impl NUTSMassMatrixConfig {
    pub fn disabled() -> Self {
        Self {
            adaptation: MassMatrixAdaptation::None,
            start_buffer: 0,
            end_buffer: 0,
            initial_window: 0,
            regularize: 0.0,
            jitter: 0.0,
            dense_max_dim: 0,
        }
    }
}

impl Default for NUTSMassMatrixConfig {
    fn default() -> Self {
        Self {
            adaptation: MassMatrixAdaptation::Diagonal,
            start_buffer: 75,
            end_buffer: 50,
            initial_window: 25,
            regularize: 0.05,
            jitter: 1e-6,
            dense_max_dim: 75,
        }
    }
}

struct RunningCov<S: Float> {
    dim: usize,
    n: usize,
    mean: Vec<S>,
    m2_diag: Vec<S>,
    m2_dense: Option<Vec<S>>,
}

impl<S: Float + FromPrimitive> RunningCov<S> {
    fn new(dim: usize, dense: bool) -> Self {
        Self {
            dim,
            n: 0,
            mean: vec![S::zero(); dim],
            m2_diag: vec![S::zero(); dim],
            m2_dense: dense.then(|| vec![S::zero(); dim * dim]),
        }
    }

    fn reset(&mut self) {
        self.n = 0;
        self.mean.fill(S::zero());
        self.m2_diag.fill(S::zero());
        if let Some(m2) = self.m2_dense.as_mut() {
            m2.fill(S::zero());
        }
    }

    fn update(&mut self, x: &[S]) {
        self.n += 1;
        let n_s = S::from_usize(self.n).unwrap();
        let mut delta = vec![S::zero(); self.dim];
        for i in 0..self.dim {
            delta[i] = x[i] - self.mean[i];
            self.mean[i] = self.mean[i] + delta[i] / n_s;
            let delta2 = x[i] - self.mean[i];
            self.m2_diag[i] = self.m2_diag[i] + delta[i] * delta2;
        }
        if let Some(m2) = self.m2_dense.as_mut() {
            let mut delta2 = vec![S::zero(); self.dim];
            for i in 0..self.dim {
                delta2[i] = x[i] - self.mean[i];
            }
            for i in 0..self.dim {
                for j in i..self.dim {
                    let idx = i * self.dim + j;
                    m2[idx] = m2[idx] + delta[i] * delta2[j];
                }
            }
        }
    }
}

struct MassMatrixWarmup<S: Float> {
    config: NUTSMassMatrixConfig,
    next_window_end: usize,
    window_len: usize,
    running: RunningCov<S>,
}

impl<S: Float + FromPrimitive> MassMatrixWarmup<S> {
    fn new(dim: usize, config: NUTSMassMatrixConfig, dense: bool) -> Self {
        let start_buffer = config.start_buffer.max(1);
        let window_len = config.initial_window.max(10);
        Self {
            config,
            next_window_end: start_buffer + window_len,
            window_len,
            running: RunningCov::new(dim, dense),
        }
    }

    fn should_collect(&self, m: usize, n_warmup: usize) -> bool {
        if m == 0 || m > n_warmup {
            return false;
        }
        if m <= self.config.start_buffer {
            return false;
        }
        m < n_warmup.saturating_sub(self.config.end_buffer)
    }

    fn note_if_window_end(&mut self, m: usize, n_warmup: usize) -> bool {
        if !self.should_collect(m, n_warmup) {
            return false;
        }
        if m >= self.next_window_end || m + 1 >= n_warmup.saturating_sub(self.config.end_buffer) {
            self.next_window_end = self.next_window_end.saturating_add(self.window_len);
            self.window_len = (self.window_len.saturating_mul(2)).min(400);
            return true;
        }
        false
    }
}

#[derive(Clone)]
enum MassMatrix<S: Float> {
    Identity {
        dim: usize,
    },
    Diagonal {
        inv: Vec<S>,
        sqrt: Vec<S>,
    },
    Dense {
        dim: usize,
        inv: Vec<S>,
        chol: Vec<S>,
    },
}

impl<S: Float + FromPrimitive> MassMatrix<S> {
    fn identity(dim: usize) -> Self {
        Self::Identity { dim }
    }

    fn diagonal_from_var(mut var: Vec<S>, jitter: S) -> Self {
        let mut inv = vec![S::zero(); var.len()];
        let mut sqrt = vec![S::zero(); var.len()];
        for i in 0..var.len() {
            let v = var[i].max(jitter);
            var[i] = v;
            inv[i] = S::one() / v;
            sqrt[i] = v.sqrt();
        }
        Self::Diagonal { inv, sqrt }
    }

    fn dense_from_cov(cov: Vec<S>, dim: usize, jitter: S) -> Option<Self> {
        let max_tries = 8usize;
        let mut j = jitter.max(S::from_f64(1e-10).unwrap());
        for _ in 0..max_tries {
            let mut cov_try = cov.clone();
            for d in 0..dim {
                cov_try[d * dim + d] = cov_try[d * dim + d] + j;
            }
            if let Some(chol) = cholesky_spd(&cov_try, dim)
                && let Some(inv) = invert_spd_from_cholesky(&chol, dim)
            {
                return Some(Self::Dense { dim, inv, chol });
            }
            j = j * S::from_f64(10.0).unwrap();
        }
        None
    }

    fn kinetic(&self, momentum: &[S]) -> S {
        let half = S::from_f64(0.5).unwrap();
        match self {
            Self::Identity { .. } => {
                let mut q = S::zero();
                for v in momentum {
                    q = q + *v * *v;
                }
                half * q
            }
            Self::Diagonal { inv, .. } => {
                let mut q = S::zero();
                for i in 0..momentum.len() {
                    q = q + momentum[i] * momentum[i] * inv[i];
                }
                half * q
            }
            Self::Dense { inv, dim, .. } => {
                let mut q = S::zero();
                for i in 0..*dim {
                    let mut row_dot = S::zero();
                    for j in 0..*dim {
                        row_dot = row_dot + inv[i * *dim + j] * momentum[j];
                    }
                    q = q + momentum[i] * row_dot;
                }
                half * q
            }
        }
    }

    fn inv_mul(&self, input: &[S], out: &mut [S]) {
        match self {
            Self::Identity { .. } => out.copy_from_slice(input),
            Self::Diagonal { inv, .. } => {
                for i in 0..input.len() {
                    out[i] = inv[i] * input[i];
                }
            }
            Self::Dense { inv, dim, .. } => {
                for i in 0..*dim {
                    let mut acc = S::zero();
                    for j in 0..*dim {
                        acc = acc + inv[i * *dim + j] * input[j];
                    }
                    out[i] = acc;
                }
            }
        }
    }

    fn sample_momentum(&self, rng: &mut SmallRng, out: &mut [S])
    where
        StandardNormal: RandDistribution<S>,
    {
        for v in out.iter_mut() {
            *v = rng.sample(StandardNormal);
        }
        match self {
            Self::Identity { .. } => {}
            Self::Diagonal { sqrt, .. } => {
                for i in 0..out.len() {
                    out[i] = out[i] * sqrt[i];
                }
            }
            Self::Dense { chol, dim, .. } => {
                let z = out.to_vec();
                for i in 0..*dim {
                    let mut acc = S::zero();
                    for j in 0..=i {
                        acc = acc + chol[i * *dim + j] * z[j];
                    }
                    out[i] = acc;
                }
            }
        }
    }
}

fn cholesky_spd<S: Float + FromPrimitive>(a: &[S], dim: usize) -> Option<Vec<S>> {
    let mut l = vec![S::zero(); dim * dim];
    for i in 0..dim {
        for j in 0..=i {
            let mut sum = a[i * dim + j];
            for k in 0..j {
                sum = sum - l[i * dim + k] * l[j * dim + k];
            }
            if i == j {
                if sum <= S::zero() || !sum.is_finite() {
                    return None;
                }
                l[i * dim + j] = sum.sqrt();
            } else {
                let d = l[j * dim + j];
                if d <= S::zero() || !d.is_finite() {
                    return None;
                }
                l[i * dim + j] = sum / d;
            }
        }
    }
    Some(l)
}

fn invert_spd_from_cholesky<S: Float + FromPrimitive>(l: &[S], dim: usize) -> Option<Vec<S>> {
    let mut inv_l = vec![S::zero(); dim * dim];
    for i in 0..dim {
        let d = l[i * dim + i];
        if d <= S::zero() || !d.is_finite() {
            return None;
        }
        inv_l[i * dim + i] = S::one() / d;
        for j in (i + 1)..dim {
            let mut sum = S::zero();
            for k in i..j {
                sum = sum + l[j * dim + k] * inv_l[k * dim + i];
            }
            inv_l[j * dim + i] = -sum / l[j * dim + j];
        }
    }
    let mut inv = vec![S::zero(); dim * dim];
    for i in 0..dim {
        for j in 0..=i {
            let mut sum = S::zero();
            for k in i.max(j)..dim {
                sum = sum + inv_l[k * dim + i] * inv_l[k * dim + j];
            }
            inv[i * dim + j] = sum;
            inv[j * dim + i] = sum;
        }
    }
    Some(inv)
}

impl<V, Target> GenericNUTS<V, Target>
where
    V: EuclideanVector + Send,
    V::Scalar: Float + FromPrimitive + ToPrimitive + Send,
    Target: HamiltonianTarget<V> + Sync + Send,
    StandardNormal: RandDistribution<V::Scalar>,
    StandardUniform: RandDistribution<V::Scalar>,
    Exp1: RandDistribution<V::Scalar>,
{
    pub fn new(target: Target, initial_positions: Vec<V>, target_accept_p: V::Scalar) -> Self {
        Self::new_with_mass_matrix(
            target,
            initial_positions,
            target_accept_p,
            NUTSMassMatrixConfig::disabled(),
        )
    }

    pub fn new_with_mass_matrix(
        target: Target,
        initial_positions: Vec<V>,
        target_accept_p: V::Scalar,
        mass_config: NUTSMassMatrixConfig,
    ) -> Self {
        let target = Arc::new(target);
        let chains = initial_positions
            .into_iter()
            .map(|pos| {
                GenericNUTSChain::new_shared(
                    Arc::clone(&target),
                    pos,
                    target_accept_p,
                    mass_config.clone(),
                )
            })
            .collect();
        Self { chains }
    }

    pub(crate) fn chains_mut(&mut self) -> &mut [GenericNUTSChain<V, Target>] {
        &mut self.chains
    }

    pub fn run(&mut self, n_collect: usize, n_discard: usize) -> Array3<V::Scalar> {
        let chain_samples: Vec<Array2<V::Scalar>> = self
            .chains
            .par_iter_mut()
            .map(|chain| chain.run(n_collect, n_discard))
            .collect();
        let views: Vec<ArrayView2<V::Scalar>> = chain_samples.iter().map(|s| s.view()).collect();
        ndarray::stack(Axis(0), &views).expect("expected stacking chain samples to succeed")
    }

    pub fn run_progress(&mut self, n_collect: usize, n_discard: usize) -> RunResult<V::Scalar> {
        let chains = &mut self.chains;

        let mut rxs: Vec<Receiver<ChainStats>> = vec![];
        let mut txs: Vec<Sender<ChainStats>> = vec![];
        (0..chains.len()).for_each(|_| {
            let (tx, rx) = mpsc::channel();
            rxs.push(rx);
            txs.push(tx);
        });

        let progress_handle = thread::spawn(move || {
            let sleep_ms = Duration::from_millis(250);
            let timeout_ms = Duration::from_millis(0);
            let multi = MultiProgress::new();

            let pb_style = ProgressStyle::default_bar()
                .template("{prefix:8} {bar:40.cyan/blue} {pos}/{len} ({eta}) | {msg}")
                .unwrap()
                .progress_chars("=>-");
            let total: u64 = (n_collect + n_discard).try_into().unwrap();

            let global_pb = multi.add(ProgressBar::new((rxs.len() as u64) * total));
            global_pb.set_style(pb_style.clone());
            global_pb.set_prefix("Global");

            let mut active: Vec<(usize, ProgressBar)> = (0..rxs.len().min(5))
                .map(|chain_idx| {
                    let pb = multi.add(ProgressBar::new(total));
                    pb.set_style(pb_style.clone());
                    pb.set_prefix(format!("Chain {chain_idx}"));
                    (chain_idx, pb)
                })
                .collect();
            let mut next_active = active.len();
            let mut n_finished = 0;
            let mut most_recent = vec![None; rxs.len()];

            loop {
                for (i, rx) in rxs.iter().enumerate() {
                    while let Ok(stats) = rx.recv_timeout(timeout_ms) {
                        most_recent[i] = Some(stats)
                    }
                }

                let mut to_replace = vec![false; active.len()];
                let mut avg_p_accept = 0.0;
                let mut n_available_stats = 0.0;
                for (vec_idx, (i, pb)) in active.iter().enumerate() {
                    if let Some(stats) = &most_recent[*i] {
                        pb.set_position(stats.n);
                        pb.set_message(format!("p(accept)≈{:.2}", stats.p_accept));
                        avg_p_accept += stats.p_accept;
                        n_available_stats += 1.0;

                        if stats.n == total {
                            to_replace[vec_idx] = true;
                            n_finished += 1;
                        }
                    }
                }
                if n_available_stats > 0.0 {
                    avg_p_accept /= n_available_stats;
                }

                let mut total_progress = 0;
                for stats in most_recent.iter().flatten() {
                    total_progress += stats.n;
                }
                global_pb.set_position(total_progress);
                let valid: Vec<&ChainStats> = most_recent.iter().flatten().collect();
                if valid.len() >= 2 {
                    let rhats = collect_rhat(valid.as_slice());
                    let max = max_skipnan(&rhats);
                    global_pb.set_message(format!(
                        "p(accept)≈{:.2} max(rhat)≈{:.2}",
                        avg_p_accept, max
                    ))
                }

                let mut to_remove = vec![];
                for (i, replace) in to_replace.iter().enumerate() {
                    if *replace && next_active < most_recent.len() {
                        let pb = multi.add(ProgressBar::new(total));
                        pb.set_style(pb_style.clone());
                        pb.set_prefix(format!("Chain {next_active}"));
                        active[i] = (next_active, pb);
                        next_active += 1;
                    } else if *replace {
                        to_remove.push(i);
                    }
                }

                to_remove.sort();
                for i in to_remove.iter().rev() {
                    active.remove(*i);
                }

                if n_finished >= most_recent.len() {
                    break;
                }
                std::thread::sleep(sleep_ms);
            }
        });

        let chain_sample: Vec<Array2<V::Scalar>> = thread::scope(|s| {
            let handles: Vec<thread::ScopedJoinHandle<Array2<V::Scalar>>> = chains
                .iter_mut()
                .zip(txs)
                .map(|(chain, tx)| {
                    s.spawn(|| {
                        chain
                            .run_progress(n_collect, n_discard, tx)
                            .expect("expected running chain to succeed.")
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .expect("expected thread to succeed in generating observation.")
                })
                .collect()
        });
        let views: Vec<ArrayView2<V::Scalar>> = chain_sample.iter().map(|s| s.view()).collect();
        let sample = ndarray::stack(Axis(0), &views).expect("expected stacking sample to succeed");

        if let Err(e) = progress_handle.join() {
            eprintln!("Progress bar thread emitted error message: {:?}", e);
        }

        let run_stats = RunStats::from(sample.view());
        Ok((sample, run_stats))
    }

    /// Warms up every chain until its adaptation has stabilized and the chains
    /// agree, then collects `n_collect` draws per chain with the adapted step size
    /// and metric.
    ///
    /// Warmup runs in windows that double from four transitions. Within a window
    /// the metric is fixed and dual averaging adapts the step size. The warmup
    /// ends at the first window where all of these hold:
    /// - the step size has stabilized: the window's mean acceptance statistic is
    ///   within the stationarity boundary of the target, in standard errors of
    ///   that mean;
    /// - the metric has stabilized: every coordinate's log posterior variance
    ///   differs from the previous window's by at most the boundary for `dim`
    ///   simultaneous statistics, in units of its sampling error
    ///   `sqrt(2/ESS)` in each window;
    /// - the chains mix: split R-hat and ESS meet `targets` in every coordinate.
    ///
    /// The boundary for `k` statistics after `n` transitions is
    /// `sqrt(2 ln k + 2 ln ln n)`. Under stationarity the chance that a window
    /// crosses it falls like `1/ln n`, so the expected number of windows is
    /// finite, while a departure that grows with the window still crosses it.
    /// A window whose metric had not stabilized replaces the metric with the
    /// pooled within-chain covariance of its draws, regularized as configured,
    /// and restarts step-size adaptation.
    ///
    /// Instead of doubling forever, the warmup refuses when two consecutive
    /// windows have stabilized but their chains disagree, or have agreeing
    /// chains but one coordinate's variance keeps moving the same way.
    pub fn run_adaptive(
        &mut self,
        n_collect: usize,
        targets: MixingTargets,
    ) -> Result<(Array3<V::Scalar>, RunStats, AdaptiveWarmup), AdaptiveWarmupError> {
        let n_chains = self.chains.len();
        if n_chains == 0 || n_collect == 0 {
            return Err(AdaptiveWarmupError::InvalidRequest(format!(
                "open warmup needs at least one chain and one requested draw, got {n_chains} chains and {n_collect} draws"
            )));
        }
        if !(targets.max_rhat.is_finite()
            && targets.max_rhat > 1.0
            && targets.min_ess.is_finite()
            && targets.min_ess >= 0.0)
        {
            return Err(AdaptiveWarmupError::InvalidRequest(format!(
                "mixing targets need a finite max_rhat above 1 and a finite non-negative min_ess, got {targets:?}"
            )));
        }
        self.chains
            .par_iter_mut()
            .for_each(|chain| chain.begin_open_warmup());
        let target_accept = self.chains[0]
            .target_accept_p
            .to_f64()
            .expect("target acceptance is representable as f64");
        let dim = self.chains[0].position.len();
        let mass_config = self.chains[0].mass_config.clone();
        let mut window_len = MIN_WARMUP_WINDOW;
        let mut transitions = 0usize;
        let mut windows = 0usize;
        let mut previous: Option<WindowMoments> = None;
        let mut disagreeing_windows = 0usize;
        let mut drifting: Option<(usize, bool, usize)> = None;
        loop {
            let outputs: Vec<(Array2<V::Scalar>, Vec<V::Scalar>)> = self
                .chains
                .par_iter_mut()
                .map(|chain| chain.warmup_window(window_len))
                .collect();
            transitions += window_len;
            windows += 1;
            let views: Vec<ArrayView2<V::Scalar>> =
                outputs.iter().map(|(draws, _)| draws.view()).collect();
            let sample = ndarray::stack(Axis(0), &views)
                .expect("expected stacking warmup windows to succeed");
            let moments = WindowMoments::from_sample(sample.view()).ok_or(
                AdaptiveWarmupError::NonFiniteDiagnostics {
                    windows,
                    transitions,
                },
            )?;
            let total_transitions = n_chains * transitions;
            let step_stable = acceptance_z(
                outputs.iter().map(|(_, accept)| accept.as_slice()),
                target_accept,
            ) <= stationarity_boundary(1, total_transitions);
            let change = previous
                .as_ref()
                .map(|earlier| moments.largest_variance_change(earlier));
            let variance_bound = stationarity_boundary(dim, total_transitions);
            let metric_stable = change.is_some_and(|change| change.z <= variance_bound);
            let ess_met = moments.min_ess > targets.min_ess;
            let chains_agree = moments.max_rhat < targets.max_rhat;
            if step_stable && metric_stable && ess_met && chains_agree {
                break;
            }
            if step_stable && metric_stable && moments.min_within_ess > targets.min_ess {
                disagreeing_windows += 1;
                if disagreeing_windows == 2 {
                    return Err(AdaptiveWarmupError::ChainsDisagree {
                        windows,
                        transitions,
                        max_rhat: moments.max_rhat,
                    });
                }
            } else {
                disagreeing_windows = 0;
            }
            drifting = match change {
                Some(change) if ess_met && chains_agree && change.z > variance_bound => {
                    let in_a_row = match drifting {
                        Some((coordinate, grew, count))
                            if coordinate == change.coordinate && grew == change.grew =>
                        {
                            count + 1
                        }
                        _ => 1,
                    };
                    if in_a_row == 2 {
                        return Err(AdaptiveWarmupError::VarianceDoesNotSettle {
                            windows,
                            transitions,
                            coordinate: change.coordinate,
                        });
                    }
                    Some((change.coordinate, change.grew, in_a_row))
                }
                _ => None,
            };
            if !metric_stable {
                let draws: Vec<&Array2<V::Scalar>> = outputs.iter().map(|(draws, _)| draws).collect();
                if let Some(metric) = pooled_metric(&draws, &mass_config) {
                    for chain in self.chains.iter_mut() {
                        chain.mass_matrix = metric.clone();
                        chain.restart_step_size_adaptation();
                    }
                }
            }
            previous = Some(moments);
            window_len = window_len
                .checked_mul(2)
                .ok_or(AdaptiveWarmupError::WindowOverflow {
                    windows,
                    transitions,
                })?;
        }
        self.chains
            .par_iter_mut()
            .for_each(|chain| chain.end_warmup());
        let collected: Vec<Array2<V::Scalar>> = self
            .chains
            .par_iter_mut()
            .map(|chain| chain.collect(n_collect))
            .collect();
        let views: Vec<ArrayView2<V::Scalar>> = collected.iter().map(|draws| draws.view()).collect();
        let sample =
            ndarray::stack(Axis(0), &views).expect("expected stacking chain samples to succeed");
        let run_stats = RunStats::from(sample.view());
        Ok((
            sample,
            run_stats,
            AdaptiveWarmup {
                transitions,
                windows,
            },
        ))
    }

    pub fn set_seed(mut self, seed: u64) -> Self {
        for (i, chain) in self.chains.iter_mut().enumerate() {
            let chain_seed = seed + i as u64 + 1;
            chain.rng = SmallRng::seed_from_u64(chain_seed);
        }
        self
    }
}

/// Single-chain state and adaptation for NUTS.
pub struct GenericNUTSChain<V, Target>
where
    V: EuclideanVector,
    Target: HamiltonianTarget<V>,
{
    target: Arc<Target>,
    position: V,
    target_accept_p: V::Scalar,
    epsilon: V::Scalar,
    m: usize,
    n_collect: usize,
    n_discard: usize,
    gamma: V::Scalar,
    t_0: usize,
    kappa: V::Scalar,
    mu: V::Scalar,
    epsilon_bar: V::Scalar,
    h_bar: V::Scalar,
    mass_matrix: MassMatrix<V::Scalar>,
    mass_warmup: Option<MassMatrixWarmup<V::Scalar>>,
    /// The mass-matrix configuration with the dense-to-diagonal fallback already
    /// resolved, kept for open-ended warmup.
    mass_config: NUTSMassMatrixConfig,
    rng: SmallRng,
}

impl<V, Target> GenericNUTSChain<V, Target>
where
    V: EuclideanVector,
    V::Scalar: Float + FromPrimitive + ToPrimitive,
    Target: HamiltonianTarget<V> + Sync + Send,
    StandardNormal: RandDistribution<V::Scalar>,
    StandardUniform: RandDistribution<V::Scalar>,
    Exp1: RandDistribution<V::Scalar>,
{
    pub fn new(target: Target, initial_position: V, target_accept_p: V::Scalar) -> Self {
        let target = Arc::new(target);
        Self::new_shared(
            target,
            initial_position,
            target_accept_p,
            NUTSMassMatrixConfig::disabled(),
        )
    }

    pub(crate) fn new_shared(
        target: Arc<Target>,
        initial_position: V,
        target_accept_p: V::Scalar,
        mass_config: NUTSMassMatrixConfig,
    ) -> Self {
        let mut thread_rng = rand::rng();
        let rng = SmallRng::from_rng(&mut thread_rng);
        let epsilon = -V::Scalar::one();
        let dim = initial_position.len();
        let adaptation = if mass_config.adaptation == MassMatrixAdaptation::Dense
            && dim > mass_config.dense_max_dim
        {
            MassMatrixAdaptation::Diagonal
        } else {
            mass_config.adaptation
        };
        let mass_matrix = MassMatrix::identity(dim);
        let mass_warmup = match adaptation {
            MassMatrixAdaptation::None => None,
            MassMatrixAdaptation::Diagonal => {
                Some(MassMatrixWarmup::new(dim, mass_config.clone(), false))
            }
            MassMatrixAdaptation::Dense => {
                Some(MassMatrixWarmup::new(dim, mass_config.clone(), true))
            }
        };
        let mass_config = NUTSMassMatrixConfig {
            adaptation,
            ..mass_config
        };

        Self {
            target,
            position: initial_position,
            target_accept_p,
            epsilon,
            m: 0,
            n_collect: 0,
            n_discard: 0,
            gamma: V::Scalar::from_f64(0.05).unwrap(),
            t_0: 10,
            kappa: V::Scalar::from_f64(0.75).unwrap(),
            mu: (V::Scalar::from_f64(10.0).unwrap() * V::Scalar::one()).ln(),
            epsilon_bar: V::Scalar::one(),
            h_bar: V::Scalar::zero(),
            mass_matrix,
            mass_warmup,
            mass_config,
            rng,
        }
    }

    pub fn set_seed(mut self, seed: u64) -> Self {
        self.rng = SmallRng::seed_from_u64(seed);
        self
    }

    pub fn position(&self) -> &V {
        &self.position
    }

    pub fn run(&mut self, n_collect: usize, n_discard: usize) -> Array2<V::Scalar> {
        let (dim, mut sample) = self.init_chain(n_collect, n_discard);
        let mut scratch = vec![V::Scalar::zero(); dim];

        for m in 1..(n_collect + n_discard) {
            self.step();

            if m >= n_discard {
                self.position.write_to_slice(&mut scratch);
                let view = ArrayView1::from(&scratch);
                sample.slice_mut(s![m - n_discard, ..]).assign(&view);
            }
        }
        sample
    }

    fn run_progress(
        &mut self,
        n_collect: usize,
        n_discard: usize,
        tx: Sender<ChainStats>,
    ) -> Result<Array2<V::Scalar>, Box<dyn Error>> {
        let (dim, mut sample) = self.init_chain(n_collect, n_discard);
        let mut scratch = vec![V::Scalar::zero(); dim];
        self.position.write_to_slice(&mut scratch);

        let mut tracker = ChainTracker::new(dim, &scratch);
        let mut last = Instant::now();
        let freq = Duration::from_secs(1);
        let total = n_discard + n_collect;

        for i in 0..total {
            self.step();
            self.position.write_to_slice(&mut scratch);
            tracker.step(&scratch).map_err(|e| {
                let msg = format!(
                    "Chain statistics tracker caused error: {}.\nAborting generation of further observations.",
                    e
                );
                println!("{}", msg);
                msg
            })?;

            let now = Instant::now();
            if (now >= last + freq) | (i == total - 1) {
                if let Err(e) = tx.send(tracker.stats()) {
                    eprintln!("Sending chain statistics failed: {e}");
                }
                last = now;
            }

            if i >= n_discard {
                let view = ArrayView1::from(&scratch);
                sample.slice_mut(s![i - n_discard, ..]).assign(&view);
            }
        }

        Ok(sample)
    }

    fn init_chain(&mut self, n_collect: usize, n_discard: usize) -> (usize, Array2<V::Scalar>) {
        let dim = self.init_chain_state(n_collect, n_discard);

        let mut sample = Array2::<V::Scalar>::zeros((n_collect, dim));
        let mut scratch = vec![V::Scalar::zero(); dim];
        self.position.write_to_slice(&mut scratch);
        let view = ArrayView1::from(&scratch);
        sample.slice_mut(s![0, ..]).assign(&view);

        (dim, sample)
    }

    pub(crate) fn init_chain_state(&mut self, n_collect: usize, n_discard: usize) -> usize {
        let dim = self.position.len();
        self.n_collect = n_collect;
        self.n_discard = n_discard;
        self.m = 0;

        let mut mom_0 = self.position.zeros_like();
        let mut mom_buf = vec![V::Scalar::zero(); dim];
        self.mass_matrix
            .sample_momentum(&mut self.rng, &mut mom_buf);
        mom_0.read_from_slice(&mom_buf);
        if let Some(warmup) = self.mass_warmup.as_mut() {
            warmup.running.reset();
        }
        if V::Scalar::abs(self.epsilon + V::Scalar::one()) <= V::Scalar::epsilon() {
            self.epsilon = find_reasonable_epsilon(&self.position, &mom_0, self.target.as_ref());
        }
        self.mu = (V::Scalar::from_f64(10.0).unwrap() * self.epsilon).ln();
        dim
    }

    pub fn step(&mut self) {
        self.step_accept();
    }

    /// Starts an open-ended warmup: `run`'s windowed schedule is switched off, a
    /// step size is found for the current metric, and dual averaging runs until
    /// `end_warmup`.
    fn begin_open_warmup(&mut self) {
        self.mass_warmup = None;
        self.n_collect = 0;
        self.n_discard = usize::MAX;
        self.restart_step_size_adaptation();
    }

    /// Finds a step size for the current metric and restarts dual averaging from it.
    fn restart_step_size_adaptation(&mut self) {
        let dim = self.position.len();
        let mut probe = self.position.zeros_like();
        let mut probe_buf = vec![V::Scalar::zero(); dim];
        self.mass_matrix
            .sample_momentum(&mut self.rng, &mut probe_buf);
        probe.read_from_slice(&probe_buf);
        self.epsilon = find_reasonable_epsilon_with_mass(
            &self.position,
            &probe,
            self.target.as_ref(),
            &self.mass_matrix,
        );
        self.mu = (V::Scalar::from_f64(10.0).unwrap() * self.epsilon).ln();
        self.epsilon_bar = self.epsilon;
        self.h_bar = V::Scalar::zero();
        self.m = 0;
    }

    /// Runs `len` adapting transitions and returns the positions they visited, one
    /// row per transition, with their acceptance statistics.
    fn warmup_window(&mut self, len: usize) -> (Array2<V::Scalar>, Vec<V::Scalar>) {
        let dim = self.position.len();
        let mut draws = Array2::<V::Scalar>::zeros((len, dim));
        let mut accept = Vec::with_capacity(len);
        let mut scratch = vec![V::Scalar::zero(); dim];
        for t in 0..len {
            accept.push(self.step_accept());
            self.position.write_to_slice(&mut scratch);
            draws
                .slice_mut(s![t, ..])
                .assign(&ArrayView1::from(&scratch));
        }
        (draws, accept)
    }

    /// Freezes the dual-averaged step size for every later transition.
    fn end_warmup(&mut self) {
        self.n_discard = self.m;
        self.epsilon = self.epsilon_bar;
    }

    /// Runs `n` transitions with adaptation frozen and returns the positions they
    /// visited.
    fn collect(&mut self, n: usize) -> Array2<V::Scalar> {
        let dim = self.position.len();
        let mut draws = Array2::<V::Scalar>::zeros((n, dim));
        let mut scratch = vec![V::Scalar::zero(); dim];
        for t in 0..n {
            self.step_accept();
            self.position.write_to_slice(&mut scratch);
            draws
                .slice_mut(s![t, ..])
                .assign(&ArrayView1::from(&scratch));
        }
        draws
    }

    /// One NUTS transition. Returns its acceptance statistic `alpha / n_alpha`, the
    /// quantity dual averaging steers toward `target_accept_p`.
    fn step_accept(&mut self) -> V::Scalar {
        self.m += 1;

        let dim = self.position.len();
        let mut mom_0 = self.position.zeros_like();
        let mut mom_buf = vec![V::Scalar::zero(); dim];
        self.mass_matrix
            .sample_momentum(&mut self.rng, &mut mom_buf);
        mom_0.read_from_slice(&mom_buf);

        let mut grad = self.position.zeros_like();
        let logp = self.target.logp_and_grad(&self.position, &mut grad);
        let joint = logp - kinetic_energy(&self.mass_matrix, &mom_0);
        let exp1_obs: V::Scalar = self.rng.sample(Exp1);
        let logu = joint - exp1_obs;

        let mut position_minus = self.position.clone();
        let mut position_plus = self.position.clone();
        let mut mom_minus = mom_0.clone();
        let mut mom_plus = mom_0.clone();
        let mut grad_minus = grad.clone();
        let mut grad_plus = grad.clone();
        let mut j = 0;
        let mut n = 1;
        let mut s = true;
        let mut alpha: V::Scalar = V::Scalar::zero();
        let mut n_alpha: usize = 0;

        while s {
            let u_run_1: V::Scalar = self.rng.random();
            let v = (2 * (u_run_1 < V::Scalar::from_f64(0.5).unwrap()) as i8) - 1;

            let (position_prime, n_prime, s_prime) = if v == -1 {
                let (
                    position_minus_2,
                    mom_minus_2,
                    grad_minus_2,
                    _,
                    _,
                    _,
                    position_prime_2,
                    _,
                    _,
                    n_prime_2,
                    s_prime_2,
                    alpha_2,
                    n_alpha_2,
                ) = build_tree_with_mass(
                    position_minus.clone(),
                    mom_minus.clone(),
                    grad_minus.clone(),
                    logu,
                    v,
                    j,
                    self.epsilon,
                    self.target.as_ref(),
                    &self.mass_matrix,
                    joint,
                    &mut self.rng,
                );

                position_minus = position_minus_2;
                mom_minus = mom_minus_2;
                grad_minus = grad_minus_2;

                alpha = alpha_2;
                n_alpha = n_alpha_2;
                (position_prime_2, n_prime_2, s_prime_2)
            } else {
                let (
                    _,
                    _,
                    _,
                    position_plus_2,
                    mom_plus_2,
                    grad_plus_2,
                    position_prime_2,
                    _,
                    _,
                    n_prime_2,
                    s_prime_2,
                    alpha_2,
                    n_alpha_2,
                ) = build_tree_with_mass(
                    position_plus.clone(),
                    mom_plus.clone(),
                    grad_plus.clone(),
                    logu,
                    v,
                    j,
                    self.epsilon,
                    self.target.as_ref(),
                    &self.mass_matrix,
                    joint,
                    &mut self.rng,
                );

                position_plus = position_plus_2;
                mom_plus = mom_plus_2;
                grad_plus = grad_plus_2;

                alpha = alpha_2;
                n_alpha = n_alpha_2;
                (position_prime_2, n_prime_2, s_prime_2)
            };

            let tmp = V::Scalar::one().min(
                V::Scalar::from_usize(n_prime)
                    .expect("successful conversion of n_prime from usize")
                    / V::Scalar::from_usize(n).expect("successful conversion of n from usize"),
            );
            let u_run_2: V::Scalar = self.rng.random();
            if s_prime && (u_run_2 < tmp) {
                self.position = position_prime;
            }
            n += n_prime;

            s = s_prime
                && stop_criterion_with_mass(
                    position_minus.clone(),
                    position_plus.clone(),
                    mom_minus.clone(),
                    mom_plus.clone(),
                    &self.mass_matrix,
                );
            j += 1
        }

        let accept =
            alpha / V::Scalar::from_usize(n_alpha).expect("successful conversion of n_alpha");
        let mut eta = V::Scalar::one()
            / V::Scalar::from_usize(self.m + self.t_0).expect("successful conversion of m + t_0");
        self.h_bar =
            (V::Scalar::one() - eta) * self.h_bar + eta * (self.target_accept_p - accept);
        if self.m <= self.n_discard {
            let m = V::Scalar::from_usize(self.m).expect("successful conversion of m");
            self.epsilon = (self.mu - m.sqrt() / self.gamma * self.h_bar).exp();
            eta = m.powf(-self.kappa);
            self.epsilon_bar =
                ((V::Scalar::one() - eta) * self.epsilon_bar.ln() + eta * self.epsilon.ln()).exp();

            if let Some(warmup) = self.mass_warmup.as_mut()
                && warmup.should_collect(self.m, self.n_discard)
            {
                let mut q = vec![V::Scalar::zero(); dim];
                self.position.write_to_slice(&mut q);
                warmup.running.update(&q);
                if warmup.note_if_window_end(self.m, self.n_discard)
                    && let Some(updated) = maybe_update_mass_matrix(&self.mass_matrix, warmup)
                {
                    self.mass_matrix = updated;
                    let mut probe = self.position.zeros_like();
                    let mut probe_buf = vec![V::Scalar::zero(); dim];
                    self.mass_matrix
                        .sample_momentum(&mut self.rng, &mut probe_buf);
                    probe.read_from_slice(&probe_buf);
                    self.epsilon =
                        find_reasonable_epsilon(&self.position, &probe, self.target.as_ref());
                    self.mu = (V::Scalar::from_f64(10.0).unwrap() * self.epsilon).ln();
                    self.epsilon_bar = self.epsilon;
                    self.h_bar = V::Scalar::zero();
                    warmup.running.reset();
                }
            }
        } else {
            self.epsilon = self.epsilon_bar;
        }
        accept
    }
}

/// Pooled within-chain moments and split diagnostics of one warmup window.
struct WindowMoments {
    /// Natural log of each coordinate's pooled within-chain variance, `-inf` for
    /// a coordinate that no chain moved in.
    log_variance: Vec<f64>,
    /// Each coordinate's ESS summed over chains, each chain split on its own. It
    /// prices the pooled within-chain variance, and unlike the multi-chain ESS it
    /// stays large when chains sample separate regions well.
    within_ess: Vec<f64>,
    min_within_ess: f64,
    max_rhat: f64,
    /// Multi-chain split ESS, smallest over coordinates.
    min_ess: f64,
}

/// The coordinate whose log variance moved most between two windows, in units of
/// its sampling error.
#[derive(Clone, Copy)]
struct VarianceChange {
    z: f64,
    coordinate: usize,
    grew: bool,
}

impl WindowMoments {
    /// `None` when split R-hat or ESS is not a number in some coordinate.
    fn from_sample<S: ToPrimitive + Clone>(sample: ArrayView3<S>) -> Option<Self> {
        let (rhat, ess) = split_rhat_mean_ess(sample.view());
        if rhat.iter().chain(ess.iter()).any(|value| value.is_nan()) {
            return None;
        }
        let (n_chains, len, dim) = sample.dim();
        let mut within_ess = vec![0.0; dim];
        for c in 0..n_chains {
            let (_, chain_ess) = split_rhat_mean_ess(sample.slice(s![c..c + 1, .., ..]));
            if chain_ess.iter().any(|value| value.is_nan()) {
                return None;
            }
            for (total, value) in within_ess.iter_mut().zip(chain_ess.iter()) {
                *total += value;
            }
        }
        let min_within_ess = within_ess.iter().copied().fold(f64::INFINITY, f64::min);
        let dof = (n_chains * (len - 1)) as f64;
        let mut log_variance = Vec::with_capacity(dim);
        for d in 0..dim {
            let mut sum_sq = 0.0;
            for c in 0..n_chains {
                let values: Vec<f64> = sample
                    .slice(s![c, .., d])
                    .iter()
                    .map(|x| x.to_f64().expect("draw is representable as f64"))
                    .collect();
                let mean = values.iter().sum::<f64>() / len as f64;
                sum_sq += values.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>();
            }
            log_variance.push((sum_sq / dof).ln());
        }
        let max_rhat = rhat.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min_ess = ess.iter().copied().fold(f64::INFINITY, f64::min);
        Some(Self {
            log_variance,
            within_ess,
            min_within_ess,
            max_rhat,
            min_ess,
        })
    }

    fn largest_variance_change(&self, earlier: &Self) -> VarianceChange {
        let mut largest = VarianceChange {
            z: 0.0,
            coordinate: 0,
            grew: false,
        };
        for d in 0..self.log_variance.len() {
            let delta = self.log_variance[d] - earlier.log_variance[d];
            let error = (2.0 / self.within_ess[d] + 2.0 / earlier.within_ess[d]).sqrt();
            let z = if delta.is_nan() {
                f64::INFINITY
            } else {
                delta.abs() / error
            };
            if !(z <= largest.z) {
                largest = VarianceChange {
                    z,
                    coordinate: d,
                    grew: delta > 0.0,
                };
            }
        }
        largest
    }
}

/// Distance of the mean acceptance statistic from `target`, in standard errors of
/// that mean.
fn acceptance_z<'a, S: ToPrimitive + 'a>(windows: impl Iterator<Item = &'a [S]>, target: f64) -> f64 {
    let values: Vec<f64> = windows
        .flat_map(|window| {
            window
                .iter()
                .map(|accept| accept.to_f64().expect("acceptance is representable as f64"))
        })
        .collect();
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|a| (a - mean) * (a - mean)).sum::<f64>() / (n - 1.0);
    let error = (variance / n).sqrt();
    let gap = (mean - target).abs();
    if error > 0.0 {
        gap / error
    } else if gap == 0.0 {
        0.0
    } else {
        f64::INFINITY
    }
}

/// Boundary for the largest of `d = simultaneous` standardized statistics after
/// `n = transitions` warmup transitions: `b = sqrt(2 ln d + 2 ln ln n)`.
///
/// Derivation. Under stationarity each statistic is approximately `N(0, 1)`,
/// with Gaussian tail `P(|Z| > b) <= sqrt(2/pi) exp(-b^2/2) / b`. The union bound
/// over the `d` statistics, together with `exp(-b^2/2) = 1 / (d ln n)` at this
/// `b`, gives
///
/// `P(max_i |Z_i| > b) <= sqrt(2/pi) / (b ln n)`.
///
/// So a stationary window fails with probability of order `1 / (b ln n)`. That
/// probability shrinks as the windows double. The chance of still warming up at
/// window `k` is a product of shrinking factors, which falls faster than the
/// window length `2^k` grows, so the expected warmup is finite.
///
/// A real departure `delta` enters its statistic as `delta * sqrt(ESS)`. That
/// grows like `sqrt(n)` while `b` grows only like `sqrt(ln ln n)`, so a
/// persistent departure still crosses the boundary.
///
/// No term is tuned: `d` is how many statistics the window tests and `n` is the
/// warmup already spent. Every window has at least four transitions, so
/// `ln ln n > 0`.
fn stationarity_boundary(simultaneous: usize, transitions: usize) -> f64 {
    (2.0 * (simultaneous as f64).ln() + 2.0 * (transitions as f64).ln().ln()).sqrt()
}

/// The metric one warmup window's draws support: the pooled within-chain
/// covariance, regularized toward the identity and floored at the configured
/// jitter. `None` keeps the current metric, when adaptation is off or a dense
/// factorization fails.
fn pooled_metric<S: Float + FromPrimitive>(
    windows: &[&Array2<S>],
    config: &NUTSMassMatrixConfig,
) -> Option<MassMatrix<S>> {
    let dim = windows.first()?.ncols();
    let dense = match config.adaptation {
        MassMatrixAdaptation::None => return None,
        MassMatrixAdaptation::Diagonal => false,
        MassMatrixAdaptation::Dense => true,
    };
    let dof: usize = windows
        .iter()
        .map(|draws| draws.nrows().saturating_sub(1))
        .sum();
    if dof == 0 {
        return None;
    }
    let n_denom = S::from_usize(dof)?;
    let reg = S::from_f64(config.regularize)?;
    let one_minus_reg = S::one() - reg;
    let jitter = metric_jitter::<S>(config)?;
    let mut sums = vec![S::zero(); if dense { dim * dim } else { dim }];
    let mut centered = vec![S::zero(); dim];
    for draws in windows {
        let len = S::from_usize(draws.nrows())?;
        let means: Vec<S> = (0..dim)
            .map(|d| draws.column(d).iter().fold(S::zero(), |acc, &x| acc + x) / len)
            .collect();
        for row in draws.rows() {
            for d in 0..dim {
                centered[d] = row[d] - means[d];
            }
            if dense {
                for i in 0..dim {
                    for j in i..dim {
                        sums[i * dim + j] = sums[i * dim + j] + centered[i] * centered[j];
                    }
                }
            } else {
                for d in 0..dim {
                    sums[d] = sums[d] + centered[d] * centered[d];
                }
            }
        }
    }
    if dense {
        for i in 0..dim {
            for j in i..dim {
                let raw = sums[i * dim + j] / n_denom;
                let v = if i == j {
                    (one_minus_reg * raw + reg).max(jitter)
                } else {
                    one_minus_reg * raw
                };
                sums[i * dim + j] = v;
                sums[j * dim + i] = v;
            }
        }
        MassMatrix::dense_from_cov(sums, dim, jitter)
    } else {
        let var = sums
            .into_iter()
            .map(|raw| (one_minus_reg * raw / n_denom + reg).max(jitter))
            .collect();
        Some(MassMatrix::diagonal_from_var(var, jitter))
    }
}

/// The configured jitter, floored where a metric inverse stays representable.
fn metric_jitter<S: FromPrimitive>(config: &NUTSMassMatrixConfig) -> Option<S> {
    S::from_f64(config.jitter.max(1e-10))
}

fn kinetic_energy<V: EuclideanVector>(mass: &MassMatrix<V::Scalar>, mom: &V) -> V::Scalar
where
    V::Scalar: Float + FromPrimitive,
{
    let mut p = vec![V::Scalar::zero(); mom.len()];
    mom.write_to_slice(&mut p);
    mass.kinetic(&p)
}

fn apply_inv_mass<V: EuclideanVector>(mass: &MassMatrix<V::Scalar>, input: &V, out: &mut V)
where
    V::Scalar: Float + FromPrimitive,
{
    let mut p = vec![V::Scalar::zero(); input.len()];
    let mut v = vec![V::Scalar::zero(); input.len()];
    input.write_to_slice(&mut p);
    mass.inv_mul(&p, &mut v);
    out.read_from_slice(&v);
}

fn maybe_update_mass_matrix<S: Float + FromPrimitive>(
    current: &MassMatrix<S>,
    warmup: &MassMatrixWarmup<S>,
) -> Option<MassMatrix<S>> {
    let n = warmup.running.n;
    if n < 5 {
        return None;
    }
    let n_denom = S::from_usize(n - 1).unwrap();
    let reg = S::from_f64(warmup.config.regularize).unwrap();
    let one_minus_reg = S::one() - reg;
    let jitter = metric_jitter::<S>(&warmup.config).unwrap();
    match warmup.config.adaptation {
        MassMatrixAdaptation::None => None,
        MassMatrixAdaptation::Diagonal => {
            let mut var = vec![S::zero(); warmup.running.dim];
            for (i, vi) in var.iter_mut().enumerate().take(warmup.running.dim) {
                let raw = warmup.running.m2_diag[i] / n_denom;
                *vi = (one_minus_reg * raw + reg).max(jitter);
            }
            Some(MassMatrix::diagonal_from_var(var, jitter))
        }
        MassMatrixAdaptation::Dense => {
            let dim = warmup.running.dim;
            let Some(m2_dense) = warmup.running.m2_dense.as_ref() else {
                return None;
            };
            let mut cov = vec![S::zero(); dim * dim];
            for i in 0..dim {
                for j in i..dim {
                    let idx = i * dim + j;
                    let raw = m2_dense[idx] / n_denom;
                    let v = if i == j {
                        (one_minus_reg * raw + reg).max(jitter)
                    } else {
                        one_minus_reg * raw
                    };
                    cov[idx] = v;
                    cov[j * dim + i] = v;
                }
            }
            MassMatrix::dense_from_cov(cov, dim, jitter).or_else(|| match current {
                MassMatrix::Diagonal { .. } | MassMatrix::Dense { .. } => None,
                MassMatrix::Identity { dim } => {
                    Some(MassMatrix::diagonal_from_var(vec![S::one(); *dim], jitter))
                }
            })
        }
    }
}

fn all_real_vec<V: EuclideanVector>(v: &V) -> bool
where
    V::Scalar: Float,
{
    let mut scratch = vec![V::Scalar::zero(); v.len()];
    v.write_to_slice(&mut scratch);
    scratch.iter().all(|x: &V::Scalar| x.is_finite())
}

#[allow(dead_code)]
pub(crate) fn find_reasonable_epsilon<V, Target>(
    position: &V,
    mom: &V,
    gradient_target: &Target,
) -> V::Scalar
where
    V: EuclideanVector,
    V::Scalar: Float + FromPrimitive,
    Target: HamiltonianTarget<V> + Sync,
    StandardNormal: RandDistribution<V::Scalar>,
    StandardUniform: RandDistribution<V::Scalar>,
{
    let mass_matrix = MassMatrix::identity(position.len());
    find_reasonable_epsilon_with_mass(position, mom, gradient_target, &mass_matrix)
}

fn find_reasonable_epsilon_with_mass<V, Target>(
    position: &V,
    mom: &V,
    gradient_target: &Target,
    mass_matrix: &MassMatrix<V::Scalar>,
) -> V::Scalar
where
    V: EuclideanVector,
    V::Scalar: Float + FromPrimitive,
    Target: HamiltonianTarget<V> + Sync,
    StandardNormal: RandDistribution<V::Scalar>,
    StandardUniform: RandDistribution<V::Scalar>,
{
    let mut epsilon = V::Scalar::one();
    let half = V::Scalar::from_f64(0.5).unwrap();

    let mut grad = position.zeros_like();
    let ulogp = gradient_target.logp_and_grad(position, &mut grad);

    let mut position_prime = position.clone();
    let mut mom_prime = mom.clone();
    let mut grad_prime = grad.clone();
    let mut ulogp_prime = leapfrog_with_mass(
        &mut position_prime,
        &mut mom_prime,
        &mut grad_prime,
        epsilon,
        gradient_target,
        mass_matrix,
    );
    let mut k = V::Scalar::one();

    while !ulogp_prime.is_finite() || !all_real_vec(&grad_prime) {
        k = k * half;
        position_prime.assign(position);
        mom_prime.assign(mom);
        grad_prime.assign(&grad);
        ulogp_prime = leapfrog_with_mass(
            &mut position_prime,
            &mut mom_prime,
            &mut grad_prime,
            epsilon * k,
            gradient_target,
            mass_matrix,
        );
    }

    epsilon = half * k * epsilon;
    let log_accept_prob = ulogp_prime
        - ulogp
        - (kinetic_energy(mass_matrix, &mom_prime) - kinetic_energy(mass_matrix, mom));
    let mut log_accept_prob = log_accept_prob;

    let a = if log_accept_prob > half.ln() {
        V::Scalar::one()
    } else {
        -V::Scalar::one()
    };

    while a * log_accept_prob > -a * V::Scalar::from_f64(2.0).unwrap().ln() {
        epsilon = epsilon * V::Scalar::from_f64(2.0).unwrap().powf(a);
        position_prime.assign(position);
        mom_prime.assign(mom);
        grad_prime.assign(&grad);
        ulogp_prime = leapfrog_with_mass(
            &mut position_prime,
            &mut mom_prime,
            &mut grad_prime,
            epsilon,
            gradient_target,
            mass_matrix,
        );
        log_accept_prob = ulogp_prime
            - ulogp
            - (kinetic_energy(mass_matrix, &mom_prime) - kinetic_energy(mass_matrix, mom));
    }

    epsilon
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn build_tree_with_mass<V, Target>(
    position: V,
    mom: V,
    grad: V,
    logu: V::Scalar,
    v: i8,
    j: usize,
    epsilon: V::Scalar,
    gradient_target: &Target,
    mass_matrix: &MassMatrix<V::Scalar>,
    joint_0: V::Scalar,
    rng: &mut SmallRng,
) -> (
    V,
    V,
    V,
    V,
    V,
    V,
    V,
    V,
    V::Scalar,
    usize,
    bool,
    V::Scalar,
    usize,
)
where
    V: EuclideanVector,
    V::Scalar: Float + FromPrimitive,
    Target: HamiltonianTarget<V> + Sync,
{
    if j == 0 {
        let mut position_prime = position.clone();
        let mut mom_prime = mom.clone();
        let mut grad_prime = grad.clone();
        let logp_prime = leapfrog_with_mass(
            &mut position_prime,
            &mut mom_prime,
            &mut grad_prime,
            V::Scalar::from_i64(v as i64).unwrap() * epsilon,
            gradient_target,
            mass_matrix,
        );
        let joint = logp_prime - kinetic_energy(mass_matrix, &mom_prime);
        let n_prime = (logu < joint) as usize;
        let s_prime = (logu - V::Scalar::from_f64(1000.0).unwrap()) < joint;
        let position_minus = position_prime.clone();
        let position_plus = position_prime.clone();
        let mom_minus = mom_prime.clone();
        let mom_plus = mom_prime.clone();
        let grad_minus = grad_prime.clone();
        let grad_plus = grad_prime.clone();
        let alpha_prime = V::Scalar::one().min((joint - joint_0).exp());
        let n_alpha_prime = 1_usize;
        (
            position_minus,
            mom_minus,
            grad_minus,
            position_plus,
            mom_plus,
            grad_plus,
            position_prime,
            grad_prime,
            logp_prime,
            n_prime,
            s_prime,
            alpha_prime,
            n_alpha_prime,
        )
    } else {
        let (
            mut position_minus,
            mut mom_minus,
            mut grad_minus,
            mut position_plus,
            mut mom_plus,
            mut grad_plus,
            mut position_prime,
            mut grad_prime,
            mut logp_prime,
            mut n_prime,
            mut s_prime,
            mut alpha_prime,
            mut n_alpha_prime,
        ) = build_tree_with_mass(
            position,
            mom,
            grad,
            logu,
            v,
            j - 1,
            epsilon,
            gradient_target,
            mass_matrix,
            joint_0,
            rng,
        );
        if s_prime {
            let (
                position_minus_2,
                mom_minus_2,
                grad_minus_2,
                position_plus_2,
                mom_plus_2,
                grad_plus_2,
                position_prime_2,
                grad_prime_2,
                logp_prime_2,
                n_prime_2,
                s_prime_2,
                alpha_prime_2,
                n_alpha_prime_2,
            ) = if v == -1 {
                build_tree_with_mass(
                    position_minus.clone(),
                    mom_minus.clone(),
                    grad_minus.clone(),
                    logu,
                    v,
                    j - 1,
                    epsilon,
                    gradient_target,
                    mass_matrix,
                    joint_0,
                    rng,
                )
            } else {
                build_tree_with_mass(
                    position_plus.clone(),
                    mom_plus.clone(),
                    grad_plus.clone(),
                    logu,
                    v,
                    j - 1,
                    epsilon,
                    gradient_target,
                    mass_matrix,
                    joint_0,
                    rng,
                )
            };
            if v == -1 {
                position_minus = position_minus_2;
                mom_minus = mom_minus_2;
                grad_minus = grad_minus_2;
            } else {
                position_plus = position_plus_2;
                mom_plus = mom_plus_2;
                grad_plus = grad_plus_2;
            }

            let u_build_tree: f64 = rng.random();
            if u_build_tree < (n_prime_2 as f64 / (n_prime + n_prime_2).max(1) as f64) {
                position_prime = position_prime_2;
                grad_prime = grad_prime_2;
                logp_prime = logp_prime_2;
            }

            n_prime += n_prime_2;

            s_prime = s_prime
                && s_prime_2
                && stop_criterion(
                    position_minus.clone(),
                    position_plus.clone(),
                    mom_minus.clone(),
                    mom_plus.clone(),
                );
            alpha_prime = alpha_prime + alpha_prime_2;
            n_alpha_prime += n_alpha_prime_2;
        }
        (
            position_minus,
            mom_minus,
            grad_minus,
            position_plus,
            mom_plus,
            grad_plus,
            position_prime,
            grad_prime,
            logp_prime,
            n_prime,
            s_prime,
            alpha_prime,
            n_alpha_prime,
        )
    }
}

pub(crate) fn stop_criterion<V>(
    position_minus: V,
    position_plus: V,
    mom_minus: V,
    mom_plus: V,
) -> bool
where
    V: EuclideanVector,
    V::Scalar: Float + FromPrimitive,
{
    let mass_matrix = MassMatrix::identity(position_minus.len());
    stop_criterion_with_mass(
        position_minus,
        position_plus,
        mom_minus,
        mom_plus,
        &mass_matrix,
    )
}

fn stop_criterion_with_mass<V>(
    position_minus: V,
    position_plus: V,
    mom_minus: V,
    mom_plus: V,
    mass_matrix: &MassMatrix<V::Scalar>,
) -> bool
where
    V: EuclideanVector,
    V::Scalar: Float + FromPrimitive,
{
    // Use proper subtraction to match original Tensor semantics
    let mut diff = position_plus.clone();
    diff.sub_assign(&position_minus);
    let mut vel_minus = mom_minus.zeros_like();
    let mut vel_plus = mom_plus.zeros_like();
    apply_inv_mass(mass_matrix, &mom_minus, &mut vel_minus);
    apply_inv_mass(mass_matrix, &mom_plus, &mut vel_plus);
    let dot_minus = diff.dot(&vel_minus);
    let dot_plus = diff.dot(&vel_plus);
    dot_minus >= V::Scalar::zero() && dot_plus >= V::Scalar::zero()
}

fn leapfrog_with_mass<V, Target>(
    position: &mut V,
    momentum: &mut V,
    grad: &mut V,
    epsilon: V::Scalar,
    gradient_target: &Target,
    mass_matrix: &MassMatrix<V::Scalar>,
) -> V::Scalar
where
    V: EuclideanVector,
    V::Scalar: Float + FromPrimitive,
    Target: HamiltonianTarget<V>,
{
    // Match original operation order: grad * epsilon * 0.5 (not grad * (0.5 * epsilon))
    let half = V::Scalar::from_f64(0.5).unwrap();
    momentum.add_scaled_assign(grad, epsilon * half);
    let mut velocity = momentum.zeros_like();
    apply_inv_mass(mass_matrix, momentum, &mut velocity);
    position.add_scaled_assign(&velocity, epsilon);
    let logp = gradient_target.logp_and_grad(position, grad);
    momentum.add_scaled_assign(grad, epsilon * half);
    logp
}

#[cfg(test)]
mod tests {
    use super::{
        AdaptiveWarmupError, GenericNUTS, MassMatrix, MassMatrixAdaptation, MassMatrixWarmup,
        MixingTargets, NUTSMassMatrixConfig, maybe_update_mass_matrix, pooled_metric,
        stationarity_boundary,
    };
    use crate::generic_hmc::HamiltonianTarget;
    use ndarray::{Array1, Array2};

    struct UnitGaussian;

    impl HamiltonianTarget<Array1<f64>> for UnitGaussian {
        fn logp_and_grad(&self, position: &Array1<f64>, grad: &mut Array1<f64>) -> f64 {
            grad.assign(&position.mapv(|x| -x));
            -0.5 * position.dot(position)
        }
    }

    /// Equal mixture of N(-center, 1) and N(center, 1) on the line.
    struct TwoModes {
        center: f64,
    }

    impl HamiltonianTarget<Array1<f64>> for TwoModes {
        fn logp_and_grad(&self, position: &Array1<f64>, grad: &mut Array1<f64>) -> f64 {
            let x = position[0];
            let y = self.center * x;
            let log_cosh = y.abs() + (-2.0 * y.abs()).exp().ln_1p() - std::f64::consts::LN_2;
            grad[0] = -x + self.center * y.tanh();
            -0.5 * (x * x + self.center * self.center) + log_cosh
        }
    }

    #[test]
    fn open_warmup_ends_on_a_unit_gaussian_and_samples_it() {
        let initial = vec![Array1::from_elem(4, 2.0), Array1::from_elem(4, -2.0)];
        let mut sampler = GenericNUTS::new_with_mass_matrix(
            UnitGaussian,
            initial,
            0.9,
            NUTSMassMatrixConfig::default(),
        )
        .set_seed(7);
        let (sample, _, warmup) = sampler
            .run_adaptive(
                400,
                MixingTargets {
                    max_rhat: 1.1,
                    min_ess: 100.0,
                },
            )
            .expect("a unit Gaussian warms up");
        assert_eq!(sample.dim(), (2, 400, 4));
        // The metric test compares a window with its predecessor.
        assert!(warmup.windows >= 2, "{warmup:?}");
        let draws = sample
            .into_shape_with_order((800, 4))
            .expect("chains flatten into one draw matrix");
        for d in 0..4 {
            let column = draws.column(d);
            let mean = column.mean().expect("non-empty column");
            let var = column.var(1.0);
            assert!(mean.abs() < 0.3, "coordinate {d} mean {mean}");
            assert!((var - 1.0).abs() < 0.35, "coordinate {d} variance {var}");
        }
    }

    #[test]
    fn open_warmup_refuses_chains_held_in_separate_modes() {
        let initial = vec![Array1::from_elem(1, -8.0), Array1::from_elem(1, 8.0)];
        let mut sampler = GenericNUTS::new_with_mass_matrix(
            TwoModes { center: 8.0 },
            initial,
            0.9,
            NUTSMassMatrixConfig::default(),
        )
        .set_seed(11);
        match sampler.run_adaptive(
            100,
            MixingTargets {
                max_rhat: 1.1,
                min_ess: 100.0,
            },
        ) {
            Err(AdaptiveWarmupError::ChainsDisagree { .. }) => {}
            other => panic!("expected the separated chains to be refused, got {other:?}"),
        }
    }

    #[test]
    fn pooled_metric_centres_each_chain_on_its_own_mean() {
        let low = Array2::from_shape_vec((4, 1), vec![-11.0, -9.0, -11.0, -9.0])
            .expect("4x1 draws");
        let high =
            Array2::from_shape_vec((4, 1), vec![9.0, 11.0, 9.0, 11.0]).expect("4x1 draws");
        let config = NUTSMassMatrixConfig {
            regularize: 0.0,
            ..NUTSMassMatrixConfig::default()
        };
        match pooled_metric(&[&low, &high], &config) {
            // Each chain deviates by 1 from its own mean: 8 squares over 6 degrees
            // of freedom, whatever the distance between the chains.
            Some(MassMatrix::Diagonal { sqrt, .. }) => {
                assert!((sqrt[0] * sqrt[0] - 8.0 / 6.0).abs() < 1e-12);
            }
            _ => panic!("expected a diagonal metric"),
        }
    }

    #[test]
    fn stationarity_boundary_grows_with_statistics_and_transitions() {
        assert!(stationarity_boundary(1, 4) > 0.0);
        assert!(stationarity_boundary(10, 1000) > stationarity_boundary(1, 1000));
        assert!(stationarity_boundary(1, 1_000_000) > stationarity_boundary(1, 1000));
    }

    #[test]
    fn diagonal_mass_matrix_kinetic_and_inv_mul_are_consistent() {
        let mass = MassMatrix::diagonal_from_var(vec![4.0_f64, 9.0_f64], 1e-12);
        let p = [2.0_f64, 3.0_f64];
        let ke = mass.kinetic(&p);
        // 0.5 * (2^2/4 + 3^2/9) = 1.0
        assert!((ke - 1.0).abs() < 1e-12);

        let mut out = [0.0_f64; 2];
        mass.inv_mul(&p, &mut out);
        assert!((out[0] - 0.5).abs() < 1e-12);
        assert!((out[1] - (1.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn dense_mass_matrix_inverse_matches_identity_action() {
        let cov = vec![
            2.0_f64, 0.3_f64, //
            0.3_f64, 1.0_f64,
        ];
        let mass = MassMatrix::dense_from_cov(cov, 2, 1e-12).expect("dense mass matrix");
        let p = [0.7_f64, -1.1_f64];
        let mut out = [0.0_f64; 2];
        mass.inv_mul(&p, &mut out);

        // For SPD matrix, p' M^{-1} p must be positive.
        let quad = p[0] * out[0] + p[1] * out[1];
        assert!(quad > 0.0);
    }

    #[test]
    fn warmup_diagonal_update_produces_positive_variances() {
        let cfg = NUTSMassMatrixConfig {
            adaptation: MassMatrixAdaptation::Diagonal,
            start_buffer: 1,
            end_buffer: 1,
            initial_window: 4,
            regularize: 0.05,
            jitter: 1e-6,
            dense_max_dim: 75,
        };
        let mut warmup = MassMatrixWarmup::new(2, cfg, false);
        let current = MassMatrix::identity(2);
        for x in [
            [-2.0_f64, 1.0_f64],
            [-1.0, 0.0],
            [0.0, 1.0],
            [2.0, -1.0],
            [1.0, 0.5],
        ] {
            warmup.running.update(&x);
        }
        let updated = maybe_update_mass_matrix(&current, &warmup).expect("updated mass");
        match updated {
            MassMatrix::Diagonal { inv, sqrt } => {
                for i in 0..2 {
                    assert!(inv[i].is_finite() && inv[i] > 0.0);
                    assert!(sqrt[i].is_finite() && sqrt[i] > 0.0);
                }
            }
            _ => panic!("expected diagonal mass matrix"),
        }
    }
}
