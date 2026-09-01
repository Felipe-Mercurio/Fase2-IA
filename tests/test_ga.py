# tests/test_ga.py
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ga_optimizer import RandomForestGA

RANDOM_SEED = 42

# ─── Fixture compartilhada ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dataset():
    """Dataset Wisconsin via sklearn (evita dependência de arquivo local nos testes)"""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target  # já 0/1

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def small_ga(dataset):
    """GA mínimo para testes rápidos (3 pop, 3 gen)"""
    X_train, X_test, y_train, y_test = dataset
    return RandomForestGA(
        X_train, X_test, y_train, y_test,
        pop_size=4,
        generations=3,
        mutation_rate=0.15,
        crossover_rate=0.7,
        experiment_name="GA_Test"
    )


# ─── 1. Inicialização ─────────────────────────────────────────────────────────

class TestGAInitialization:

    def test_ga_creates_with_correct_params(self, dataset):
        X_train, X_test, y_train, y_test = dataset
        ga = RandomForestGA(X_train, X_test, y_train, y_test,
                            pop_size=10, generations=5,
                            mutation_rate=0.1, crossover_rate=0.8,
                            experiment_name="TestInit")
        assert ga.pop_size == 10
        assert ga.generations == 5
        assert ga.mutation_rate == 0.1
        assert ga.crossover_rate == 0.8
        assert ga.experiment_name == "TestInit"

    def test_history_starts_empty(self, small_ga):
        for key in ['generation', 'best_fitness', 'avg_fitness', 'diversity']:
            assert small_ga.history[key] == []

    def test_best_fitness_starts_at_negative_infinity(self, small_ga):
        assert small_ga.best_fitness == -np.inf


# ─── 2. População inicial ─────────────────────────────────────────────────────

class TestPopulationInit:

    def test_population_size_correct(self, small_ga):
        pop = small_ga.init_population()
        assert len(pop) == small_ga.pop_size

    def test_individual_has_five_genes(self, small_ga):
        pop = small_ga.init_population()
        for individual in pop:
            assert len(individual) == 5, "Cada indivíduo deve ter 5 genes (hiperparâmetros)"

    def test_genes_within_bounds(self, small_ga):
        pop = small_ga.init_population()
        bounds = list(small_ga.param_bounds.values())
        for individual in pop:
            for gene, (low, high) in zip(individual, bounds):
                assert low <= gene <= high, f"Gene {gene} fora dos limites [{low}, {high}]"

    def test_population_has_diversity(self, small_ga):
        """Populações diferentes não devem ser idênticas"""
        pop1 = small_ga.init_population()
        pop2 = small_ga.init_population()
        # Pelo menos um gene deve diferir entre as duas populações
        all_same = all(
            np.allclose(ind1, ind2)
            for ind1, ind2 in zip(pop1, pop2)
        )
        assert not all_same, "Duas populações iniciais não devem ser idênticas"


# ─── 3. Conversão de indivíduo para parâmetros ───────────────────────────────

class TestParamConversion:

    def test_param_dict_has_all_keys(self, small_ga):
        individual = [100, 10, 5, 2, 0.5]
        params = small_ga.param_dict_from_individual(individual)
        expected_keys = {'n_estimators', 'max_depth', 'min_samples_split',
                         'min_samples_leaf', 'max_features'}
        assert set(params.keys()) == expected_keys

    def test_int_params_are_integers(self, small_ga):
        individual = [123.7, 14.9, 7.3, 3.8, 0.6]
        params = small_ga.param_dict_from_individual(individual)
        assert isinstance(params['n_estimators'], int)
        assert isinstance(params['max_depth'], int)
        assert isinstance(params['min_samples_split'], int)
        assert isinstance(params['min_samples_leaf'], int)

    def test_max_features_is_float(self, small_ga):
        individual = [100, 10, 5, 2, 0.75]
        params = small_ga.param_dict_from_individual(individual)
        assert isinstance(params['max_features'], float)

    def test_param_values_match_individual(self, small_ga):
        individual = [200, 15, 8, 3, 0.6]
        params = small_ga.param_dict_from_individual(individual)
        assert params['n_estimators'] == 200
        assert params['max_depth'] == 15
        assert params['max_features'] == pytest.approx(0.6)


# ─── 4. Função de fitness ─────────────────────────────────────────────────────

class TestFitnessFunction:

    def test_fitness_returns_tuple(self, small_ga):
        individual = [100, 10, 5, 2, 0.5]
        result = small_ga.evaluate_individual(individual)
        assert isinstance(result, tuple), "DEAP espera tupla"

    def test_fitness_between_zero_and_one(self, small_ga):
        individual = [100, 10, 5, 2, 0.5]
        fitness, = small_ga.evaluate_individual(individual)
        assert 0.0 <= fitness <= 1.0

    def test_invalid_individual_returns_zero(self, small_ga):
        """Indivíduo com parâmetros extremos não deve quebrar — retorna 0"""
        bad_individual = [1, 1, 2, 1, 0.01]  # n_estimators=1, depth=1
        fitness, = small_ga.evaluate_individual(bad_individual)
        assert fitness >= 0.0

    def test_good_params_beat_bad_params(self, small_ga):
        good = [100, 10, 5, 2, 0.5]
        bad  = [1,   1,  2, 1, 0.2]
        good_f1, = small_ga.evaluate_individual(good)
        bad_f1,  = small_ga.evaluate_individual(bad)
        assert good_f1 >= bad_f1, "Parâmetros razoáveis devem superar configuração degenerada"


# ─── 5. Operadores genéticos ──────────────────────────────────────────────────

class TestGeneticOperators:

    def test_crossover_produces_two_children(self, small_ga):
        p1 = [100, 10, 5, 2, 0.5]
        p2 = [200, 20, 8, 4, 0.7]
        c1, c2 = small_ga.crossover(p1, p2)
        assert len(c1) == 5
        assert len(c2) == 5

    def test_crossover_genes_come_from_parents(self, small_ga):
        p1 = [100, 10, 5, 2, 0.5]
        p2 = [200, 20, 8, 4, 0.7]
        c1, c2 = small_ga.crossover(p1, p2)
        for gene in c1:
            assert gene in p1 or gene in p2
        for gene in c2:
            assert gene in p1 or gene in p2

    def test_crossover_preserves_all_genes(self, small_ga):
        """Genes não devem se perder — cada gene de pai deve aparecer em algum filho"""
        p1 = [100, 10, 5, 2, 0.5]
        p2 = [200, 20, 8, 4, 0.7]
        c1, c2 = small_ga.crossover(p1, p2)
        combined = c1 + c2
        for gene in p1 + p2:
            assert gene in combined

    def test_mutation_returns_same_length(self, small_ga):
        individual = [100, 10, 5, 2, 0.5]
        mutated = small_ga.mutate_individual(individual)
        assert len(mutated) == len(individual)

    def test_mutation_stays_within_bounds(self, small_ga):
        individual = [100, 10, 5, 2, 0.5]
        bounds = list(small_ga.param_bounds.values())
        for _ in range(20):  # testar várias mutações
            mutated = small_ga.mutate_individual(individual)
            for gene, (low, high) in zip(mutated, bounds):
                assert low <= gene <= high, f"Mutação saiu dos limites: {gene} não está em [{low},{high}]"

    def test_mutation_does_not_modify_original(self, small_ga):
        individual = [100, 10, 5, 2, 0.5]
        original   = individual.copy()
        small_ga.mutate_individual(individual)
        assert individual == original, "mutate_individual não deve modificar o original"

    def test_zero_mutation_rate_keeps_individual(self, dataset):
        X_train, X_test, y_train, y_test = dataset
        ga_no_mut = RandomForestGA(X_train, X_test, y_train, y_test,
                                   pop_size=3, generations=2,
                                   mutation_rate=0.0, crossover_rate=0.7,
                                   experiment_name="NoMutation")
        individual = [100, 10, 5, 2, 0.5]
        mutated = ga_no_mut.mutate_individual(individual)
        assert mutated == pytest.approx(individual), "Taxa de mutação 0 não deve alterar o indivíduo"


# ─── 6. Evolução completa ─────────────────────────────────────────────────────

class TestEvolution:

    def test_evolve_returns_history(self, small_ga):
        history = small_ga.evolve(verbose=False)
        assert isinstance(history, dict)

    def test_history_has_correct_length(self, small_ga):
        history = small_ga.evolve(verbose=False)
        assert len(history['generation'])    == small_ga.generations
        assert len(history['best_fitness'])  == small_ga.generations
        assert len(history['avg_fitness'])   == small_ga.generations

    def test_best_fitness_is_set_after_evolve(self, small_ga):
        small_ga.evolve(verbose=False)
        assert small_ga.best_fitness > -np.inf

    def test_best_individual_is_set_after_evolve(self, small_ga):
        small_ga.evolve(verbose=False)
        assert small_ga.best_individual is not None
        assert len(small_ga.best_individual) == 5

    def test_fitness_never_decreases_globally(self, small_ga):
        """O melhor fitness global nunca deve piorar entre gerações"""
        history = small_ga.evolve(verbose=False)
        best = history['best_fitness']
        for i in range(1, len(best)):
            assert best[i] >= best[i-1] - 1e-9, \
                f"Melhor fitness caiu na geração {i}: {best[i-1]:.4f} → {best[i]:.4f}"

    def test_get_best_model_returns_fitted_model(self, small_ga):
        small_ga.evolve(verbose=False)
        model, params = small_ga.get_best_model()
        assert isinstance(model, RandomForestClassifier)
        # Modelo deve conseguir predizer sem erro (já está fitado)
        X_train, X_test, y_train, y_test = small_ga.X_train, small_ga.X_test, \
                                            small_ga.y_train, small_ga.y_test
        preds = model.predict(X_test)
        assert len(preds) == len(y_test)

    def test_get_best_model_params_within_bounds(self, small_ga):
        small_ga.evolve(verbose=False)
        _, params = small_ga.get_best_model()
        bounds = small_ga.param_bounds
        assert bounds['n_estimators'][0]      <= params['n_estimators']      <= bounds['n_estimators'][1]
        assert bounds['max_depth'][0]          <= params['max_depth']          <= bounds['max_depth'][1]
        assert bounds['min_samples_split'][0]  <= params['min_samples_split']  <= bounds['min_samples_split'][1]
        assert bounds['max_features'][0]       <= params['max_features']       <= bounds['max_features'][1]


# ─── 7. Métricas do modelo otimizado ─────────────────────────────────────────

class TestOptimizedModelMetrics:

    def test_optimized_model_achieves_minimum_f1(self, dataset):
        """Modelo otimizado deve atingir pelo menos F1=0.90 no Wisconsin"""
        X_train, X_test, y_train, y_test = dataset
        ga = RandomForestGA(X_train, X_test, y_train, y_test,
                            pop_size=5, generations=5,
                            mutation_rate=0.15, crossover_rate=0.7,
                            experiment_name="QualityCheck")
        ga.evolve(verbose=False)
        model, _ = ga.get_best_model()
        f1 = f1_score(y_test, model.predict(X_test))
        assert f1 >= 0.90, f"F1 esperado >= 0.90, obtido {f1:.4f}"

    def test_different_experiments_explore_different_params(self, dataset):
        """Dois experimentos com seeds diferentes devem encontrar parâmetros distintos"""
        X_train, X_test, y_train, y_test = dataset

        np.random.seed(0)
        ga1 = RandomForestGA(X_train, X_test, y_train, y_test,
                             pop_size=5, generations=5, mutation_rate=0.15,
                             crossover_rate=0.7, experiment_name="Exp_A")
        ga1.evolve(verbose=False)

        np.random.seed(99)
        ga2 = RandomForestGA(X_train, X_test, y_train, y_test,
                             pop_size=5, generations=5, mutation_rate=0.15,
                             crossover_rate=0.7, experiment_name="Exp_B")
        ga2.evolve(verbose=False)

        params1 = ga1.param_dict_from_individual(ga1.best_individual)
        params2 = ga2.param_dict_from_individual(ga2.best_individual)

        # Pelo menos um parâmetro deve ser diferente entre os dois
        any_different = any(
            abs(params1[k] - params2[k]) > 1e-6
            for k in params1
        )
        assert any_different, "Experimentos com seeds diferentes devem explorar espaços distintos"