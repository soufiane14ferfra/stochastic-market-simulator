#include <cmath>
#include <random>

// La commande extern "C" est cruciale : elle indique au compilateur C++ 
// de garder la fonction "lisible" pour d'autres langages comme Python.
extern "C" {
    
    // Le moteur prend les paramètres financiers et un "pointeur" (result_array) 
    // vers un espace mémoire que Python aura préparé.
    void simulate_gbm(double S0, double T, double mu, double sigma, int n_steps, int n_sims, double* result_array) {
        
        // Pré-calcul des constantes pour éviter de les recalculer dans la boucle
        double dt = T / n_steps;
        double drift = (mu - 0.5 * sigma * sigma) * dt;
        double vol = sigma * std::sqrt(dt);

        // Initialisation du générateur de nombres aléatoires (Standard de l'industrie)
        // Le "Mersenne Twister" (mt19937) est exigé en finance pour sa robustesse
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0.0, 1.0);

        // Double boucle : on simule chaque trajectoire, et pour chaque trajectoire, chaque jour
        for (int i = 0; i < n_sims; ++i) {
            int offset = i * (n_steps + 1); // Pour naviguer dans la mémoire à plat
            result_array[offset] = S0;      // Prix initial
            
            for (int t = 1; t <= n_steps; ++t) {
                double Z = d(gen); // Tirage d'un choc aléatoire
                result_array[offset + t] = result_array[offset + t - 1] * std::exp(drift + vol * Z);
            }
        }
    }
}