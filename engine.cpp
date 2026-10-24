#include <cmath>
#include <cstdint>
#include <random>

// extern "C" : garde des noms de fonctions lisibles par Python (ctypes).
extern "C" {

// Trajectoires completes (utile pour les graphes et, plus tard, les options path-dependent).
// Schema exact du GBM : S_{t+dt} = S_t * exp((r - sigma^2/2) dt + sigma sqrt(dt) Z)
void simulate_gbm(double S0, double T, double r, double sigma,
                  int n_steps, int n_sims, uint64_t seed, double* result_array) {
    const double dt = T / n_steps;
    const double drift = (r - 0.5 * sigma * sigma) * dt;
    const double vol = sigma * std::sqrt(dt);

    std::mt19937_64 gen(seed);              // graine fixe => resultats reproductibles
    std::normal_distribution<double> d(0.0, 1.0);

    for (int i = 0; i < n_sims; ++i) {
        const long offset = static_cast<long>(i) * (n_steps + 1);
        result_array[offset] = S0;
        for (int t = 1; t <= n_steps; ++t) {
            result_array[offset + t] = result_array[offset + t - 1] * std::exp(drift + vol * d(gen));
        }
    }
}

// Prix d'un call europeen par Monte Carlo.
// Le payoff ne depend que de S_T, et le GBM a une solution exacte : un seul pas suffit,
// sans biais de discretisation et sans stocker les trajectoires (memoire O(1)).
// Sorties : out[0] = prix estime, out[1] = erreur standard de l'estimateur.
void price_european_call(double S0, double K, double T, double r, double sigma,
                         long n_sims, uint64_t seed, double* out) {
    const double drift = (r - 0.5 * sigma * sigma) * T;
    const double vol = sigma * std::sqrt(T);
    const double disc = std::exp(-r * T);

    std::mt19937_64 gen(seed);
    std::normal_distribution<double> d(0.0, 1.0);

    // Somme et somme des carres en une passe pour la moyenne et la variance
    double sum = 0.0, sum_sq = 0.0;
    for (long i = 0; i < n_sims; ++i) {
        const double ST = S0 * std::exp(drift + vol * d(gen));
        const double payoff = disc * std::fmax(ST - K, 0.0);
        sum += payoff;
        sum_sq += payoff * payoff;
    }
    const double mean = sum / n_sims;
    const double var = (sum_sq - n_sims * mean * mean) / (n_sims - 1);  // variance non biaisee
    out[0] = mean;
    out[1] = std::sqrt(var / n_sims);
}

}
