// Monte Carlo pricing engine (C++17), exposed to Python through a C interface (ctypes).
//
// Design choices
// - Paths are split into fixed blocks of BLOCK paths. Block b always uses its own
//   std::mt19937_64 stream seeded with (seed, b). Results are therefore reproducible
//   and identical whatever the number of threads.
// - Each kernel accumulates raw sums (sum, sum of squares, cross products) per block;
//   blocks are merged at the end. Means, standard errors and control-variate
//   coefficients are computed from these sums in one pass.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>

namespace {

constexpr long BLOCK = 1 << 14;   // 16,384 paths per random stream
constexpr int NACC = 16;          // number of running sums a kernel may use

struct Acc {
    double v[NACC] = {0.0};
    void add(const Acc& o) { for (int k = 0; k < NACC; ++k) v[k] += o.v[k]; }
};

double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

// Branchless max(d, 0). About half of the paths end in the money, so the branch in
// std::max is mispredicted ~50% of the time; this version halves the run time of the
// European kernel. Exact: d + |d| is 2d or 0, and halving is exact.
inline double pos(double d) { return 0.5 * (d + std::fabs(d)); }

// Standard normal generator: Ziggurat method (Marsaglia & Tsang, 2000), 128 layers.
// About 99% of draws cost one 64-bit random number, one comparison and one multiply,
// versus a log and a sqrt (plus rejections) for the polar method behind
// std::normal_distribution. The layer index and the value come from different bits of
// the 64-bit draw, which avoids the correlation flaw of the original 32-bit version.
class Ziggurat {
    uint32_t kn[128];
    double wn[128], fn[128];
    static constexpr double R = 3.442619855899;   // start of the tail

    static double uniform(std::mt19937_64& g) { return ((g() >> 11) + 0.5) * 0x1.0p-53; }

public:
    Ziggurat() {
        const double m1 = 2147483648.0, vn = 9.91256303526217e-3;
        double dn = R, tn = R;
        const double q = vn / std::exp(-0.5 * dn * dn);
        kn[0] = static_cast<uint32_t>((dn / q) * m1);
        kn[1] = 0;
        wn[0] = q / m1;
        wn[127] = dn / m1;
        fn[0] = 1.0;
        fn[127] = std::exp(-0.5 * dn * dn);
        for (int i = 126; i >= 1; --i) {
            dn = std::sqrt(-2.0 * std::log(vn / dn + std::exp(-0.5 * dn * dn)));
            kn[i + 1] = static_cast<uint32_t>((dn / tn) * m1);
            tn = dn;
            fn[i] = std::exp(-0.5 * dn * dn);
            wn[i] = dn / m1;
        }
    }

    double operator()(std::mt19937_64& g) const {
        for (;;) {
            const uint64_t u = g();
            const int32_t hz = static_cast<int32_t>(u >> 32);    // signed value: high bits
            const int iz = static_cast<int>(u & 127);             // layer: low bits
            const uint32_t ahz = hz < 0 ? static_cast<uint32_t>(-static_cast<int64_t>(hz))
                                        : static_cast<uint32_t>(hz);
            if (ahz < kn[iz]) return hz * wn[iz];                 // fast path: inside the box
            const double x = hz * wn[iz];
            if (iz == 0) {                                        // tail beyond R
                double xt, y;
                do {
                    xt = -std::log(uniform(g)) / R;
                    y = -std::log(uniform(g));
                } while (y + y < xt * xt);
                return hz > 0 ? R + xt : -R - xt;
            }
            // wedge: accept with probability proportional to the density
            if (fn[iz] + uniform(g) * (fn[iz - 1] - fn[iz]) < std::exp(-0.5 * x * x)) return x;
        }
    }
};

const Ziggurat N01;   // tables built once; read-only and therefore thread-safe

// Runs kernel(rng, n_paths_in_block, acc) over all blocks, spread across n_threads.
template <class Kernel>
Acc run_blocks(long n_paths, uint64_t seed, int n_threads, Kernel kernel) {
    const long n_blocks = (n_paths + BLOCK - 1) / BLOCK;
    std::vector<Acc> partial(n_blocks);
    n_threads = std::max(1, std::min<int>(n_threads, static_cast<int>(n_blocks)));

    auto worker = [&](int tid) {
        for (long b = tid; b < n_blocks; b += n_threads) {
            std::seed_seq seq{static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32),
                              static_cast<uint32_t>(b), static_cast<uint32_t>(b >> 32)};
            std::mt19937_64 rng(seq);
            const long count = std::min(BLOCK, n_paths - b * BLOCK);
            Acc local;                 // stack copy: lets the compiler keep the sums in registers
            kernel(rng, count, local);
            partial[b] = local;
        }
    };

    std::vector<std::thread> pool;
    for (int t = 1; t < n_threads; ++t) pool.emplace_back(worker, t);
    worker(0);
    for (auto& th : pool) th.join();

    Acc total;   // merged in block order => bit-for-bit identical across thread counts
    for (const auto& a : partial) total.add(a);
    return total;
}

// Sample mean and standard error from a sum and a sum of squares.
void mean_se(double s, double ss, double n, double& mean, double& se) {
    mean = s / n;
    const double var = std::max(0.0, (ss - n * mean * mean) / (n - 1.0));
    se = std::sqrt(var / n);
}

// Control-variate estimator: Y - beta (X - EX), beta = Cov(Y,X)/Var(X) estimated in-sample.
// Uses sums sY, sX, sYY, sXX, sXY over n samples.
void control_variate(double sY, double sX, double sYY, double sXX, double sXY, double n,
                     double EX, double& price, double& se, double& rho) {
    const double mY = sY / n, mX = sX / n;
    const double vY = (sYY - n * mY * mY) / (n - 1.0);
    const double vX = (sXX - n * mX * mX) / (n - 1.0);
    const double cYX = (sXY - n * mY * mX) / (n - 1.0);
    const double beta = vX > 0 ? cYX / vX : 0.0;
    price = mY - beta * (mX - EX);
    const double vRes = std::max(0.0, vY - beta * cYX);   // = Var(Y) (1 - rho^2)
    se = std::sqrt(vRes / n);
    rho = (vX > 0 && vY > 0) ? cYX / std::sqrt(vX * vY) : 0.0;
}

}  // namespace

extern "C" {

// ---------------------------------------------------------------------------
// Full GBM paths (exact scheme), used for plots. result has n_sims*(n_steps+1) doubles.
void simulate_gbm(double S0, double T, double r, double sigma, int n_steps, int n_sims,
                  uint64_t seed, double* result) {
    const double dt = T / n_steps;
    const double drift = (r - 0.5 * sigma * sigma) * dt, vol = sigma * std::sqrt(dt);
    std::mt19937_64 rng(seed);
    for (long i = 0; i < n_sims; ++i) {
        double* p = result + i * (n_steps + 1);
        p[0] = S0;
        for (int t = 1; t <= n_steps; ++t) p[t] = p[t - 1] * std::exp(drift + vol * N01(rng));
    }
}

// ---------------------------------------------------------------------------
// European call. n_draws = number of normal draws (the computational budget).
// method bit 1: antithetic variates (Z, -Z) -> n_draws/2 independent pair averages.
// method bit 2: control variate X = e^{-rT} S_T, whose mean S0 is known exactly.
// out[0] = price, out[1] = standard error, out[2] = correlation(Y, X) (0 if unused).
void european_call(double S0, double K, double T, double r, double sigma, long n_draws,
                   uint64_t seed, int method, int n_threads, double* out) {
    const bool anti = method & 1, cv = method & 2;
    const double drift = (r - 0.5 * sigma * sigma) * T, vol = sigma * std::sqrt(T);
    const double disc = std::exp(-r * T);
    const long n_samples = anti ? n_draws / 2 : n_draws;

    Acc a = run_blocks(n_samples, seed, n_threads, [&](std::mt19937_64& rng, long n, Acc& acc) {
        for (long i = 0; i < n; ++i) {
            const double z = N01(rng);
            double ST = S0 * std::exp(drift + vol * z);
            double Y = disc * pos(ST - K), X = disc * ST;
            if (anti) {
                const double ST2 = S0 * std::exp(drift - vol * z);
                Y = 0.5 * (Y + disc * pos(ST2 - K));
                X = 0.5 * (X + disc * ST2);
            }
            acc.v[0] += Y; acc.v[1] += Y * Y;
            acc.v[2] += X; acc.v[3] += X * X; acc.v[4] += X * Y;
        }
    });

    const double n = static_cast<double>(n_samples);
    if (cv) {
        control_variate(a.v[0], a.v[2], a.v[1], a.v[3], a.v[4], n, S0, out[0], out[1], out[2]);
    } else {
        mean_se(a.v[0], a.v[1], n, out[0], out[1]);
        out[2] = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Greeks of a European call, all from the SAME draws (common random numbers).
// out (value, standard error) pairs:
//  [0,1] delta  bump-and-reprice (central difference, h = hS * S0)
//  [2,3] gamma  bump-and-reprice (second difference)
//  [4,5] vega   bump-and-reprice (central difference, h = hV)
//  [6,7] delta  pathwise  e^{-rT} 1{S_T>K} S_T / S0
//  [8,9] vega   pathwise  e^{-rT} 1{S_T>K} S_T (sqrt(T) Z - sigma T)
void european_greeks(double S0, double K, double T, double r, double sigma, long n_draws,
                     double hS, double hV, uint64_t seed, int n_threads, double* out) {
    const double disc = std::exp(-r * T), sqT = std::sqrt(T);
    const double dS = hS * S0;
    auto payoff = [&](double s0, double sig, double z) {
        const double ST = s0 * std::exp((r - 0.5 * sig * sig) * T + sig * sqT * z);
        return disc * pos(ST - K);
    };

    Acc a = run_blocks(n_draws, seed, n_threads, [&](std::mt19937_64& rng, long n, Acc& acc) {
        for (long i = 0; i < n; ++i) {
            const double z = N01(rng);
            const double p0 = payoff(S0, sigma, z);
            const double pu = payoff(S0 + dS, sigma, z), pd = payoff(S0 - dS, sigma, z);
            const double vu = payoff(S0, sigma + hV, z), vd = payoff(S0, sigma - hV, z);
            const double ST = S0 * std::exp((r - 0.5 * sigma * sigma) * T + sigma * sqT * z);
            const double itm = ST > K ? 1.0 : 0.0;
            const double g[5] = {
                (pu - pd) / (2 * dS),
                (pu - 2 * p0 + pd) / (dS * dS),
                (vu - vd) / (2 * hV),
                disc * itm * ST / S0,
                disc * itm * ST * (sqT * z - sigma * T),
            };
            for (int k = 0; k < 5; ++k) { acc.v[2 * k] += g[k]; acc.v[2 * k + 1] += g[k] * g[k]; }
        }
    });
    for (int k = 0; k < 5; ++k)
        mean_se(a.v[2 * k], a.v[2 * k + 1], static_cast<double>(n_draws), out[2 * k], out[2 * k + 1]);
}

// ---------------------------------------------------------------------------
// Closed-form price of a discretely monitored GEOMETRIC Asian call (dates iT/m, i=1..m).
// log G is Gaussian, so the price has a Black-Scholes-type formula.
double geometric_asian_call(double S0, double K, double T, double r, double sigma, int m) {
    const double mu = std::log(S0) + (r - 0.5 * sigma * sigma) * T * (m + 1.0) / (2.0 * m);
    const double v = sigma * sigma * T * (m + 1.0) * (2.0 * m + 1.0) / (6.0 * m * m);
    const double d1 = (mu - std::log(K) + v) / std::sqrt(v), d2 = d1 - std::sqrt(v);
    return std::exp(-r * T) * (std::exp(mu + 0.5 * v) * norm_cdf(d1) - K * norm_cdf(d2));
}

// Arithmetic Asian call, plain MC and with the geometric Asian call as control variate.
// out[0,1] plain price, se   out[2,3] control-variate price, se   out[4] correlation
void asian_call(double S0, double K, double T, double r, double sigma, int m, long n_paths,
                uint64_t seed, int n_threads, double* out) {
    const double dt = T / m, drift = (r - 0.5 * sigma * sigma) * dt, vol = sigma * std::sqrt(dt);
    const double disc = std::exp(-r * T);

    Acc a = run_blocks(n_paths, seed, n_threads, [&](std::mt19937_64& rng, long n, Acc& acc) {
        for (long i = 0; i < n; ++i) {
            double logS = std::log(S0), sumS = 0.0, sumLog = 0.0;
            for (int t = 0; t < m; ++t) {
                logS += drift + vol * N01(rng);
                sumS += std::exp(logS);
                sumLog += logS;
            }
            const double Y = disc * pos(sumS / m - K);
            const double X = disc * pos(std::exp(sumLog / m) - K);
            acc.v[0] += Y; acc.v[1] += Y * Y;
            acc.v[2] += X; acc.v[3] += X * X; acc.v[4] += X * Y;
        }
    });
    const double n = static_cast<double>(n_paths);
    mean_se(a.v[0], a.v[1], n, out[0], out[1]);
    control_variate(a.v[0], a.v[2], a.v[1], a.v[3], a.v[4], n,
                    geometric_asian_call(S0, K, T, r, sigma, m), out[2], out[3], out[4]);
}

// ---------------------------------------------------------------------------
// Up-and-out call with a CONTINUOUSLY monitored barrier B, simulated on m steps.
// Naive: knock out only if S is above B at a grid date -> misses crossings between
//        dates, so it overprices.
// Brownian bridge: given two consecutive log-prices x0, x1 below b = log B, the
//        probability that the path crossed b in between is exp(-2 (b-x0)(b-x1) / (sigma^2 dt)).
//        Weighting the payoff by the survival probability removes the discretisation bias.
// out[0,1] naive price, se   out[2,3] Brownian-bridge price, se
void up_and_out_call(double S0, double K, double B, double T, double r, double sigma, int m,
                     long n_paths, uint64_t seed, int n_threads, double* out) {
    const double dt = T / m, drift = (r - 0.5 * sigma * sigma) * dt, vol = sigma * std::sqrt(dt);
    const double disc = std::exp(-r * T), b = std::log(B), s2dt = sigma * sigma * dt;

    Acc a = run_blocks(n_paths, seed, n_threads, [&](std::mt19937_64& rng, long n, Acc& acc) {
        for (long i = 0; i < n; ++i) {
            double x = std::log(S0), survive = 1.0;
            bool hit = x >= b;
            for (int t = 0; t < m && !hit; ++t) {
                const double xn = x + drift + vol * N01(rng);
                if (xn >= b) { hit = true; break; }
                survive *= 1.0 - std::exp(-2.0 * (b - x) * (b - xn) / s2dt);
                x = xn;
            }
            const double payoff = hit ? 0.0 : disc * pos(std::exp(x) - K);
            const double bb = payoff * survive;
            acc.v[0] += payoff; acc.v[1] += payoff * payoff;
            acc.v[2] += bb;     acc.v[3] += bb * bb;
        }
    });
    const double n = static_cast<double>(n_paths);
    mean_se(a.v[0], a.v[1], n, out[0], out[1]);
    mean_se(a.v[2], a.v[3], n, out[2], out[3]);
}

}  // extern "C"
