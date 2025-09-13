# MO824-A3: Implementação Tabu Search para MAX-SC-QBF

## Visão Geral

Este repositório contém uma implementação de algoritmo de **Tabu Search** para resolver o problema de otimização **Maximum Set Cover Quadratic Binary Formulation (MAX-SC-QBF)**. O projeto foi desenvolvido como parte da disciplina MO824 (Programação Inteira).

## O Problema MAX-SC-QBF

O MAX-SC-QBF é um problema de otimização combinatória que combina dois problemas clássicos:

- **Set Cover (SC)**: Encontrar o menor conjunto de subconjuntos que cubra todos os elementos de um universo
- **Quadratic Binary Formulation (QBF)**: Maximizar uma função objetivo quadrática sobre variáveis binárias

### Formulação Matemática

Dado:
- Um conjunto de **n** subconjuntos
- Uma matriz **A** (n×n) de coeficientes quadráticos
- Cada subconjunto cobre determinados elementos de um universo

**Objetivo**: Maximizar a função quadrática dos subconjuntos selecionados, sujeito à restrição de que todos os elementos do universo sejam cobertos.

## Estrutura do Projeto

### Arquivos Principais

```
MO824-A3/
├── scqbf/                          # Módulo principal
│   ├── scqbf_instance.py           # Definição de instâncias do problema
│   ├── scqbf_solution.py           # Estrutura de soluções
│   ├── scqbf_evaluator.py          # Avaliação de soluções e deltas
│   └── scqbf_ts.py                 # Implementação do Tabu Search
├── instances/                      # Instâncias de teste
│   ├── gen1/                       # Instâncias geradas - Conjunto 1
│   ├── gen2/                       # Instâncias geradas - Conjunto 2  
│   ├── gen3/                       # Instâncias geradas - Conjunto 3
│   └── sample_instances/           # Instâncias de exemplo
├── testing.ipynb                   # Notebook para testes
└── .gitignore                      # Arquivos ignorados pelo Git
```

### Componentes do Sistema

#### 1. **scqbf_instance.py**
- **`ScQbfInstance`**: Classe que representa uma instância do problema
- **`read_max_sc_qbf_instance()`**: Função para ler instâncias de arquivos
- Formato das instâncias:
  - Primeira linha: número de subconjuntos (n)
  - Segunda linha: tamanhos de cada subconjunto
  - Linhas seguintes: elementos de cada subconjunto
  - Matriz triangular superior: coeficientes da função objetivo

#### 2. **scqbf_solution.py**
- **`ScQbfSolution`**: Representa uma solução (lista de subconjuntos selecionados)
- Armazena o valor da função objetivo para otimização

#### 3. **scqbf_evaluator.py**
- **`ScQbfEvaluator`**: Classe responsável por avaliar soluções
- **Métodos principais**:
  - `evaluate_objfun()`: Calcula valor da função objetivo
  - `evaluate_insertion_delta()`: Delta para inserção de elemento
  - `evaluate_removal_delta()`: Delta para remoção de elemento  
  - `evaluate_exchange_delta()`: Delta para troca de elementos
  - `evaluate_coverage()`: Verifica cobertura dos elementos
  - `is_solution_valid()`: Valida se solução cobre todos elementos

#### 4. **scqbf_ts.py**
- **`ScQbfTS`**: Implementação do algoritmo Tabu Search
- **Parâmetros configuráveis**:
  - `tenure`: Tempo de permanência na lista tabu
  - `max_iter`: Máximo de iterações
  - `time_limit_secs`: Limite de tempo em segundos
  - `patience`: Iterações sem melhoria para parada antecipada
- **Métodos em desenvolvimento**:
  - `_constructive_heuristic()`: Construção de solução inicial
  - `_neighborhood_move()`: Movimento na vizinhança

## Como Usar

### Pré-requisitos

- Python 3.7+
- NumPy (para operações matemáticas)
- Jupyter Notebook (para testes)

### Instalação

```bash
# Clone o repositório
git clone https://github.com/Everton-Colombo/MO824-A3.git
cd MO824-A3

# Instale dependências (se necessário)
pip install numpy jupyter
```

### Exemplo Básico de Uso

```python
from scqbf.scqbf_instance import read_max_sc_qbf_instance
from scqbf.scqbf_evaluator import ScQbfEvaluator
from scqbf.scqbf_solution import ScQbfSolution
from scqbf.scqbf_ts import ScQbfTS

# Carregar instância
instance = read_max_sc_qbf_instance('instances/sample_instances/1.txt')

# Criar avaliador
evaluator = ScQbfEvaluator(instance)

# Criar uma solução de exemplo
solution = ScQbfSolution(elements=[0, 1, 2])

# Avaliar solução
obj_value = evaluator.evaluate_objfun(solution)
coverage = evaluator.evaluate_coverage(solution)
is_valid = evaluator.is_solution_valid(solution)

print(f"Valor objetivo: {obj_value}")
print(f"Cobertura: {coverage:.2%}")
print(f"Solução válida: {is_valid}")

# Executar Tabu Search
tabu_search = ScQbfTS(
    instance=instance, 
    tenure=10, 
    max_iter=1000, 
    time_limit_secs=60
)

best_solution = tabu_search.solve()
```

### Testando com Jupyter Notebook

```bash
jupyter notebook testing.ipynb
```

## Formato das Instâncias

As instâncias seguem o formato:

```
n                          # Número de subconjuntos
s1 s2 ... sn              # Tamanhos dos subconjuntos
e11 e12 ... e1s1          # Elementos do subconjunto 1
e21 e22 ... e2s2          # Elementos do subconjunto 2
...
en1 en2 ... ensn          # Elementos do subconjunto n
a11 a12 ... a1n           # Primeira linha da matriz A
a22 ... a2n               # Segunda linha da matriz A (triangular superior)
...
ann                       # Última entrada da matriz A
```

## Algoritmo Tabu Search

O algoritmo implementa os conceitos clássicos do Tabu Search:

1. **Solução Inicial**: Construída por heurística construtiva
2. **Estrutura de Vizinhança**: Movimentos de inserção, remoção e troca
3. **Lista Tabu**: Previne ciclos com tenure configurável
4. **Critérios de Parada**: Por iterações, tempo ou falta de melhoria
5. **Avaliação Eficiente**: Cálculo incremental de deltas

### Características da Implementação

- **Avaliação Incremental**: Calcula mudanças na função objetivo sem recalcular completamente
- **Validação de Cobertura**: Garante que todas as soluções cubram o universo completo  
- **Critérios Múltiplos de Parada**: Flexibilidade na configuração de terminação
- **Estrutura Modular**: Separação clara entre instância, solução, avaliação e busca

## Status do Desenvolvimento

### ✅ Implementado
- [x] Leitura de instâncias
- [x] Estruturas de dados para soluções
- [x] Sistema de avaliação completo
- [x] Cálculos de delta eficientes
- [x] Validação de cobertura
- [x] Framework base do Tabu Search

### 🚧 Em Desenvolvimento
- [ ] Heurística construtiva
- [ ] Definição de vizinhança
- [ ] Estratégias de diversificação
- [ ] Otimizações de performance

## Instâncias Disponíveis

- **`sample_instances/`**: Instâncias pequenas para teste inicial
- **`gen1/`, `gen2/`, `gen3/`**: Conjuntos de instâncias com diferentes características
- **Tamanhos variados**: De problemas pequenos (n=6) até grandes (n=400)

## Contribuição

Para contribuir com o projeto:

1. Faça um fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## Licença

Este projeto é parte de trabalho acadêmico da disciplina MO824 (Programação Inteira).

## Contato

- **Autor**: Everton Colombo
- **Disciplina**: MO824 - Programação Inteira
- **Repositório**: https://github.com/Everton-Colombo/MO824-A3

---

*Última atualização: Setembro 2025*