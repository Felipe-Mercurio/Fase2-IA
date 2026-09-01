# src/ga_optimizer.py
# Cole aqui a classe RandomForestGA do notebook
# (exatamente o mesmo código da Célula 3.1)

import numpy as np
import random
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import logging

logger = logging.getLogger(__name__)
RANDOM_SEED = 42

class RandomForestGA:
   from deap import base, creator, tools, algorithms
from copy import deepcopy

class RandomForestGA:
    """
    Otimizador de hiperparâmetros Random Forest usando Algoritmo Genético
    """
    
    def __init__(self, X_train, X_test, y_train, y_test, 
                 pop_size=20, generations=50, mutation_rate=0.15, 
                 crossover_rate=0.7, experiment_name="GA_Exp"):
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.experiment_name = experiment_name
        
        # Espaço de hiperparâmetros para otimizar
        self.param_bounds = {
            'n_estimators': (10, 500),      # int
            'max_depth': (3, 30),           # int
            'min_samples_split': (2, 20),   # int
            'min_samples_leaf': (1, 10),    # int
            'max_features': (0.2, 1.0),     # float
        }
        
        # Histórico
        self.history = {
            'generation': [],
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': [],
            'best_individual': [],
            'best_params': []
        }
        
        self.best_individual = None
        self.best_fitness = -np.inf
        
        logger.info(f"GA Initializado: {experiment_name}")
        logger.info(f"  Population: {pop_size}, Generations: {generations}")
        logger.info(f"  Mutation Rate: {mutation_rate}, Crossover Rate: {crossover_rate}")
    
    def param_dict_from_individual(self, individual):
        """Converter array individual -> dict parâmetros"""
        return {
            'n_estimators': int(individual[0]),
            'max_depth': int(individual[1]),
            'min_samples_split': int(individual[2]),
            'min_samples_leaf': int(individual[3]),
            'max_features': individual[4],
        }
    
    def evaluate_individual(self, individual):
        """
        Fitness function: treinar RF com estes hiperparâmetros
        Retorna F1-score (métrica crítica para recall)
        """
        try:
            params = self.param_dict_from_individual(individual)
            
            # Treinar modelo com estes parâmetros
            rf = RandomForestClassifier(
                **params,
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
            rf.fit(self.X_train, self.y_train)
            
            # Predições
            y_pred = rf.predict(self.X_test)
            
            # Fitness = F1-score (importante para recall médico)
            fitness = f1_score(self.y_test, y_pred)
            
            return fitness,  # DEAP espera tupla
        
        except Exception as e:
            logger.warning(f"Erro ao avaliar indivíduo: {e}")
            return 0.0,  # Penalidade se erro
    
    def init_population(self):
        """Inicializar população aleatória dentro dos limites"""
        population = []
        for _ in range(self.pop_size):
            individual = [
                np.random.uniform(self.param_bounds['n_estimators'][0], 
                                 self.param_bounds['n_estimators'][1]),
                np.random.uniform(self.param_bounds['max_depth'][0], 
                                 self.param_bounds['max_depth'][1]),
                np.random.uniform(self.param_bounds['min_samples_split'][0], 
                                 self.param_bounds['min_samples_split'][1]),
                np.random.uniform(self.param_bounds['min_samples_leaf'][0], 
                                 self.param_bounds['min_samples_leaf'][1]),
                np.random.uniform(self.param_bounds['max_features'][0], 
                                 self.param_bounds['max_features'][1]),
            ]
            population.append(individual)
        return population
    
    def mutate_individual(self, individual):
        """Mutação: adicionar noise Gaussiano"""
        mutated = deepcopy(individual)
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                # Adicionar ruído Gaussiano
                noise = np.random.normal(0, 0.1)
                mutated[i] += noise * (self.param_bounds[list(self.param_bounds.keys())[i]][1] - 
                                       self.param_bounds[list(self.param_bounds.keys())[i]][0])
                
                # Clip aos limites
                bounds = list(self.param_bounds.values())[i]
                mutated[i] = np.clip(mutated[i], bounds[0], bounds[1])
        
        return mutated
    
    def crossover(self, parent1, parent2):
        """Crossover: single-point crossover"""
        point = np.random.randint(1, len(parent1))
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    
    def evolve(self, verbose=True):
        """Executar o algoritmo genético"""
        logger.info(f"Iniciando evolução para {self.experiment_name}")
        
        # Inicializar população
        population = self.init_population()
        
        # Avaliar população inicial
        fitnesses = [self.evaluate_individual(ind)[0] for ind in population]
        
        for generation in range(self.generations):
            # Seleção (Tournament selection - top 50%)
            sorted_pop = sorted(zip(population, fitnesses), 
                               key=lambda x: x[1], 
                               reverse=True)
            elite_size = max(2, self.pop_size // 2)   # garante mínimo 2 para crossover
            population = [ind for ind, _ in sorted_pop[:elite_size]]
            fitnesses  = [fit for _, fit in sorted_pop[:elite_size]]
            
            # Crossover + Mutação (preencher população)
            while len(population) < self.pop_size:
                parent1, parent2 = random.sample(population, 2)
                
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = deepcopy(parent1), deepcopy(parent2)
                
                child1 = self.mutate_individual(child1)
                child2 = self.mutate_individual(child2)
                
                population.extend([child1, child2])
            
            # Truncar população ao tamanho correto
            population = population[:self.pop_size]
            
            # Re-avaliar
            fitnesses = [self.evaluate_individual(ind)[0] for ind in population]
            
            # Armazenar histórico
            best_fit = max(fitnesses)
            avg_fit = np.mean(fitnesses)
            diversity = np.std(fitnesses)
            best_ind = population[np.argmax(fitnesses)]
            
            self.history['generation'].append(generation)
            self.history['best_fitness'].append(best_fit)
            self.history['avg_fitness'].append(avg_fit)
            self.history['diversity'].append(diversity)
            self.history['best_individual'].append(best_ind)
            self.history['best_params'].append(self.param_dict_from_individual(best_ind))
            
            # Atualizar melhor global
            if best_fit > self.best_fitness:
                self.best_fitness = best_fit
                self.best_individual = deepcopy(best_ind)
            
            if verbose and (generation + 1) % 10 == 0:
                print(f"Gen {generation+1:3d} | Best: {best_fit:.4f} | Avg: {avg_fit:.4f} | Div: {diversity:.4f}")
                logger.info(f"Gen {generation+1:3d} | Best: {best_fit:.4f} | Avg: {avg_fit:.4f}")
        
        logger.info(f"Evolução concluída para {self.experiment_name}")
        logger.info(f"Best Fitness: {self.best_fitness:.4f}")
        return self.history
    
    def get_best_model(self):
        """Retornar modelo treinado com melhores hiperparâmetros"""
        best_params = self.param_dict_from_individual(self.best_individual)
        model = RandomForestClassifier(
            **best_params,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)
        return model, best_params
    
    def plot_evolution(self):
        """Plotar convergência do GA"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Best Fitness
        axes[0, 0].plot(self.history['generation'], self.history['best_fitness'], 'b-', linewidth=2)
        axes[0, 0].set_title(f'{self.experiment_name} - Best Fitness Over Generations')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('F1-Score')
        axes[0, 0].grid(True)
        
        # Avg vs Best
        axes[0, 1].plot(self.history['generation'], self.history['avg_fitness'], 'r--', label='Average')
        axes[0, 1].plot(self.history['generation'], self.history['best_fitness'], 'b-', label='Best')
        axes[0, 1].set_title(f'{self.experiment_name} - Average vs Best Fitness')
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Diversity
        axes[1, 0].plot(self.history['generation'], self.history['diversity'], 'g-', linewidth=2)
        axes[1, 0].set_title(f'{self.experiment_name} - Population Diversity')
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Std Dev of Fitness')
        axes[1, 0].grid(True)
        
        # Hyperparameters ao longo gerações
        n_est = [p['n_estimators'] for p in self.history['best_params']]
        max_d = [p['max_depth'] for p in self.history['best_params']]
        
        ax = axes[1, 1]
        ax2 = ax.twinx()
        
        ax.plot(self.history['generation'], n_est, 'purple', marker='o', label='n_estimators')
        ax2.plot(self.history['generation'], max_d, 'orange', marker='s', label='max_depth')
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('n_estimators', color='purple')
        ax2.set_ylabel('max_depth', color='orange')
        ax.set_title(f'{self.experiment_name} - Hyperparameter Evolution')
        ax.grid(True)
        
        plt.tight_layout()
        return fig

print("✅ Classe RandomForestGA definida")