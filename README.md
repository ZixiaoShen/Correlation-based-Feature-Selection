# Correlation-based-Feature-Selection

Introduce the principle and the theory below the Correlation based Feature Selection method.


CFS: Correlation-based Feature Selection is composed with three parts:

1. Feature Evaluation

The heart of CFS algorithm is a heuristic for evaluation the worth or merit of a subset of features;

2. Feature Correlations

Information gain is used to calculate the correlation between different features and class;

3. Searching the Feature Subset Space

CFS starts from the empty set of features and uses a forward best first search with a stopping criterion of
five consecutive fully expanded non-improving subsets.