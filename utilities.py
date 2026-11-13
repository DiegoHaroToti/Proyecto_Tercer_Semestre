import numpy as np


def gradient_descent(grad_fn, x0, learning_rate=0.1, n_iterations=100):
    x = np.array(x0, dtype=float)
    for _ in range(n_iterations):
        x = x - learning_rate * grad_fn(x)
    return x


def train_test_split(X, y, test_size=0.25, seed=0):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    n_test = int(round(len(X) * test_size))
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std


def k_fold_indices(n_samples, k=5, seed=0):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    folds = np.array_split(indices, k)
    splits = []
    for i in range(k):
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, folds[i]))
    return splits


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred)/len(y_true)

def confusion_counts(y_true, y_pred):
    vp = int(np.sum((y_true == 1) & (y_pred == 1)))
    vn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return vp, fp,fn,vn

def precision_recall_f1(y_true, y_pred):
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    precision = tp / ((tp + fp) or 1)
    recall = tp / ((tp + fn) or 1)

    f1 = (2 * precision * recall) / ((precision + recall) or 1)

    return (precision, recall, f1)


class LinearRegression:
    import numpy as np
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.bias = 0
        self.weights = None


    def predict(self, X):
        y_pred = X @ self.weights + self.bias
        return y_pred

    def compute_cost(self, X, y):
        predictions = self.predict(X)
        errors = predictions - y
        return np.mean(errors ** 2)

    def compute_gradients(self, X, y):
        n_samples = X.shape[0]
        predictions = self.predict(X)
        errors = predictions - y
        dw = (2 / n_samples) * X.T @(predictions - y)
        db = (2 / n_samples) * sum(predictions - y)
        return (dw, db)



    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for _ in range(self.n_iterations):
            dw, db = self.compute_gradients(X, y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        return self



class MultipleLinearRegression:
    def __init__(self):
        self.bias = 0
        self.weights = None

    def add_intercept(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])


    def predict(self, X):
        y_pred = X @ self.weights + self.bias
        return y_pred


    def fit(self, X, y):
        Xb = self.add_intercept(X)
        theta = np.linalg.pinv(Xb.T.dot(Xb)).dot(Xb.T).dot(y)
        self.bias = theta[0]
        self.weights = theta[1:]
        return self


    def score(self, X, y):
        y_pred = self.predict(X)
        SS_res = sum((y_pred - y)**2)
        SS_tot = sum((y-np.mean(y))**2)
        R2 = 1- SS_res / SS_tot
        return R2


    def adjusted_r2(self, X, y):
        r2 = self.score (X, y)
        n, p = X.shape
        R2_ajust = 1 - (1-r2) *(n-1)/(n - p - 1)
        return  R2_ajust

    def correlated_pairs(self, X, threshold=0.9):
        corr = np.corrcoef(X, rowvar=False)
        pares = []
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                if i < j and abs(corr[i, j]) > threshold:
                    pares.append((i, j))
        return pares


class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def _sigmoid(self, z):
        return  (1 / (1 + np.exp(-z)))

    def predict_proba(self, X):
        z = X.dot(self.weights) + self.bias
        p = self._sigmoid(z)
        return p


    def predict(self, X, threshold=0.5):
        y_pred = (self.predict_proba(X) >= threshold).astype(int)
        return y_pred

    def compute_cost(self, X, y):
        p = self.predict_proba(X)
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))



    def compute_gradients(self, X, y):
        n_samples = X.shape[0]
        errors = self.predict_proba(X) - y
        dw = (1 / n_samples) * X.T.dot(errors)
        db = (1 / n_samples) * np.sum(errors)
        return dw, db


    def fit(self, X, y):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        for _ in range(self.n_iterations):
            dw, db = self.compute_gradients(X, y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        return self


from collections import Counter


class KNNClassifier:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors


    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self


    def _euclidean_distance(self, a, b):
        distance = np.sqrt(np.sum((a - b) ** 2,  axis=1))
        return distance

    def _predict_one(self, x):
        distances = self._euclidean_distance(self.X_train, x )
        indices = np.argsort(distances)[:self.n_neighbors]
        nearest_labels = self.y_train[indices]
        return Counter(nearest_labels.tolist()).most_common(1)[0][0]



    def predict(self, X):
        A = []
        for fila in X:
            A.append(self._predict_one(fila))
        return np.array(A)




class DecisionTreeClassifier:
    """Arbol de decision de la leccion 4, ya implementado. Las etiquetas son enteros 0..K-1.
    Si max_features no es None, en cada division solo se consideran max_features
    caracteristicas elegidas al azar con rng (esto es lo que decorrela los arboles del bosque)."""

    def __init__(self, max_depth=5, min_samples_split=2, max_features=None , rng=None,):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.rng = rng
        self.max_features = max_features


    def _gini(self, y):
        counts = np.bincount(y)
        proportions = counts / len(y)
        return 1 - np.sum(proportions ** 2)

    def _split_dataset(self, X, y, feature_index, threshold=.5):
        mask = X[:, feature_index] <= threshold
        return X[mask], y[mask], X[~mask], y[~mask]

    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gini = np.inf
        n = len(y)
        n_features = X.shape[1]

        for feature_index in range(n_features):
            threshold = np.unique(X[:, feature_index])

            for b in (threshold):
                n_derecha, y_derecha, n_izquierda, y_izquierda = self._split_dataset(X, y, feature_index, b)

                if len(y_izquierda) == 0 or len(y_derecha) == 0:
                    continue

                weighted_gini = (len(n_izquierda) / n) * self._gini(y_izquierda) + (
                            len(n_derecha) / n) * self._gini(y_derecha)

                if (weighted_gini < best_gini):
                    best_feature = feature_index
                    best_gini = weighted_gini
                    best_threshold = b
        return best_feature, best_threshold


    def _build_tree(self, X, y, depth):
        leaf = {
            'leaf': True, 'class': np.bincount(y).argmax()
        }
        if (len(np.unique(y)) == 1) or depth >= self.max_depth or len(y) < self.min_samples_split:
            return leaf


        else:
            feature, threshold = self._best_split(X, y)
            if (feature is None):
                return leaf
            else:
                n_izquierda, y_izquierda, n_derecha, y_derecha = self._split_dataset(X, y, feature, threshold)
                depth += 1
                left = self._build_tree(n_izquierda, y_izquierda, depth)
                right = self._build_tree(n_derecha, y_derecha, depth)
                internal_node = {
                    'leaf': False,
                    'feature': feature,
                    'threshold': threshold,
                    'left': left,
                    'right': right
                }
        return internal_node

    def fit(self, X, y):
        self.tree_ = self._build_tree(X, y, 0)
        return self

    def _predict_one(self, x, node):
        if node['leaf'] == True:
            return node['class']
        else:
            if x[node['feature']] <= node['threshold']:
                return self._predict_one(x, node['left'])
            else:
                return self._predict_one(x, node['right'])

    def predict(self, X):
        result = []
        for i in range(X.shape[0]):
            Predic = self._predict_one(X[i], self.tree_)
            result.append(Predic)
        return result

class RandomForestClassifier:

    def __init__(self, n_trees=25, max_depth=5, seed=0):
        self.n_trees = n_trees
        self.seed = seed
        self.max_depth = max_depth


    def _bootstrap_sample(self, X, y, rng):
        n = len(X)
        indices = rng.integers(0, n, size=n)
        return X[indices], y[indices], indices

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.n_classes_ = int(y.max()) + 1
        max_features = max(1, int(np.sqrt(X.shape[1])))
        self.trees_ = []
        self.oob_indices_ = []
        for i in range(self.n_trees):
            X_b, y_b, idx = self._bootstrap_sample(X, y, rng)
            model = DecisionTreeClassifier(max_depth=self.max_depth, max_features=max_features, rng=rng)
            model.fit(X_b, y_b)
            self.trees_.append(model)
            self.oob_indices_.append(np.setdiff1d(np.arange(len(X)), idx))

        return self


    def predict(self, X):
        clases = []
        Matriz = np.array([tree.predict(X) for tree in self.trees_])
        for col in range(Matriz.shape[1]):
            votos = np.bincount(Matriz[: , col], minlength=self.n_classes_)
            clases.append(np.argmax(votos))
        return clases

    def oob_score(self, X, y):
        votes = np.zeros((len(X), self.n_classes_), dtype=int)

        for (tree, oob) in zip(self.trees_, self.oob_indices_):
            if (len(oob) > 0):
                preds = tree.predict(X[oob])
                votes[oob, preds] += 1
        covered = votes.sum(axis=1) > 0
        oob_score =np.mean(votes[covered].argmax(axis=1) == y[covered])
        return oob_score



class RegressionTree:
    """Arbol de regresion pequeno, ya implementado. Cada division minimiza la suma de errores
    cuadraticos y cada hoja predice la media de sus muestras."""

    def __init__(self, max_depth=5, min_samples_split=2, max_features=None , rng=None,):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.rng = rng
        self.max_features = max_features


    def _var (self, y):
        mse = np.mean((y - np.mean(y))**2)
        return mse

    def _split_dataset(self, X, y, feature_index, threshold=.5):
        mask = X[:, feature_index] <= threshold
        return X[mask], y[mask], X[~mask], y[~mask]

    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_var = np.inf
        n = len(y)
        n_features = X.shape[1]

        for feature_index in range(n_features):
            threshold = np.unique(X[:, feature_index])

            for b in (threshold):
                n_derecha, y_derecha, n_izquierda, y_izquierda = self._split_dataset(X, y, feature_index, b)

                if len(y_izquierda) == 0 or len(y_derecha) == 0:
                    continue

                weighted_var = (len(n_izquierda) / n) * self._var(y_izquierda) + (
                            len(n_derecha) / n) * self._var(y_derecha)

                if (weighted_var < best_var):
                    best_feature = feature_index
                    best_var = weighted_var
                    best_threshold = b
        return best_feature, best_threshold


    def _build_tree(self, X, y, depth):
        leaf = {
            'leaf': True, 'class': np.mean(y)
        }
        if (len(np.unique(y)) == 1) or depth >= self.max_depth or len(y) < self.min_samples_split:
            return leaf


        else:
            feature, threshold = self._best_split(X, y)
            if (feature is None):
                return leaf
            else:
                n_izquierda, y_izquierda, n_derecha, y_derecha = self._split_dataset(X, y, feature, threshold)
                depth += 1
                left = self._build_tree(n_izquierda, y_izquierda, depth)
                right = self._build_tree(n_derecha, y_derecha, depth)
                internal_node = {
                    'leaf': False,
                    'feature': feature,
                    'threshold': threshold,
                    'left': left,
                    'right': right
                }
        return internal_node

    def fit(self, X, y):
        self.tree_ = self._build_tree(X, y, 0)
        return self

    def _predict_one(self, x, node):
        if node['leaf'] == True:
            return node['class']
        else:
            if x[node['feature']] <= node['threshold']:
                return self._predict_one(x, node['left'])
            else:
                return self._predict_one(x, node['right'])

    def predict(self, X):
        result = []
        for i in range(X.shape[0]):
            Predic = self._predict_one(X[i], self.tree_)
            result.append(Predic)
        return np.array(result)



class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth


    def _initial_prediction(self, y):
        F0 = np.mean(y)
        return F0

    def _residuals(self, y, F):
        return y - F


    def fit(self, X, y):
        self.init_ = self._initial_prediction(y)
        F = np.full(len(y), self.init_)
        self.trees_ = []
        for _ in range(self.n_estimators):
            residuals = self._residuals(y, F)
            tree = RegressionTree(max_depth=self.max_depth).fit(X, residuals)
            F = F + self.learning_rate * np.array(tree.predict(X))
            self.trees_.append(tree)
        return self


    def predict(self, X):
        F = []
        F = np.full(X.shape[0], self.init_)
        for tree in (self.trees_):
            F = F + self.learning_rate * tree.predict(X)
        return F


    def staged_predict(self, X):
        F = np.full(X.shape[0], self.init_)
        stages = []
        for tree in (self.trees_):
            F = F + self.learning_rate * tree.predict(X)
            stages.append(F.copy())
        return stages


class GaussianNaiveBayes:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.priors_ =[]
        self.means_ =[]
        self.vars_ = []
        for c in self.classes_:
            self.priors_.append(np.mean(y == c))
            self.means_.append( X[y == c].mean(axis=0))
            self.vars_.append((X[y == c].var(axis=0))+ self.var_smoothing )
        self.priors_ = np.array(self.priors_)
        self.means_ = np.array(self.means_)
        self.vars_ = np.array(self.vars_)

        return self

    def _log_gaussian(self, X, mean, var):
        return (-.5 * np.log( 2 * np.pi * var) - (X - mean) **2 / (2 * var))

    def joint_log_likelihood(self, X):
        valor = []
        for k in range(len(self.classes_)):
            log_likelihood = self._log_gaussian(X, self.means_[k], self.vars_[k]).sum(axis=1)
            valor.append(np.log(self.priors_[k]) + log_likelihood)
        lista = np.column_stack(valor)
        return lista

    def predict(self, X):
        return self.classes_[np.argmax(self.joint_log_likelihood(X), axis=1)]

    def predict_proba(self, X):
        jll = self.joint_log_likelihood(X)
        jll = jll - jll.max(axis=1, keepdims=True)
        probs = np.exp(jll)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs


class LinearSVM:
    def __init__(self, learning_rate=0.01, n_iterations=1000, C=1.0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.C = C


    def decision_function(self, X):
        score = X.dot(self.weights) + self.bias
        return score


    def predict(self, X):
        scores  = self.decision_function(X)
        return np.where(scores >= 0, 1, 0)


    def compute_cost(self, X, y):
        margins = y * self.decision_function(X)
        hinge = np.maximum(0, 1 - margins)
        return 0.5 * np.sum(self.weights ** 2) + self.C * np.mean(hinge)


    def compute_gradients(self, X, y):
        n_samples = X.shape[0]
        violators = (y * self.decision_function(X)) < 1
        slack_signal = y * violators
        dw = self.weights - (self.C / n_samples) * X.T.dot(slack_signal)
        db = -(self.C / n_samples) * np.sum(slack_signal)
        return dw, db



    def fit(self, X, y):
        y_signed = np.where(y <= 0, -1, 1)
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        for _ in range(self.n_iterations) :
            dw, db = self.compute_gradients(X, y_signed)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        return self





class KMeans:
    def __init__(self, n_clusters=2, n_iterations=100):
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.labels = None


    def _assign_clusters(self, X, centroids):
        distances = np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)


    def _update_centroids(self, X, labels, n_clusters):
        cluster = np.array([X[labels == k].mean(axis=0) for k in range(n_clusters) ])
        return cluster

    def fit(self, X, initial_centroids):
        centroids = np.array(initial_centroids, dtype=float)
        for i in range(self.n_iterations):
            labels = self._assign_clusters(X, centroids)
            centroids = self._update_centroids(X, labels, len(centroids))
            self.centroids_ = centroids
            self.labels_ = labels
        return self



class PCA:
    def __init__(self, n_components=1):
        self.n_components = n_components

    def _center(self, X):
        return X - self.mean_

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        X_centered = self._center(X)
        cov = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        return self


    def transform(self, X):
        X_centered = self._center(X)
        X = X_centered.dot(self.components_.T)
        return X



    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)



class AgglomerativeClustering:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters

    def _pairwise_distances(self, X):
        distances = np.sqrt(((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2).sum(axis=2))
        return distances


    def _cluster_distance(self, cluster_a, cluster_b, distances):
        cluster = distances[np.ix_(cluster_a, cluster_b)]
        return cluster.min()

    def fit(self, X):
        n = X.shape[0]
        distances =  self._pairwise_distances(X)
        clusters = [[i] for i in range(n)]
        while len(clusters) > self.n_clusters :
            min = np.inf
            A = 0
            B = 0
            for a in range(len(clusters)) :
                for b in range(a + 1, len(clusters)):
                    distance = self._cluster_distance(clusters[a], clusters[b], distances)
                    if distance < min:
                        B = b
                        A = a
                        min = distance
            clusters[A] = clusters[A] + clusters[B]
            del clusters[B]
        zeros = np.zeros(n, dtype=int)
        for id in range(len(clusters)):
            zeros[clusters[id]] = id
        self.labels_ = zeros
        return self


class DBSCAN:
    def __init__(self, eps=1.0, min_samples=3):
        self.eps  = eps
        self.min_samples = min_samples


    def _region_query(self, X, point_idx):
        distances = []
        for _ in range(X.shape[0]):
            distances = np.sqrt(((X - X[point_idx]) ** 2).sum(axis=1))
        return np.where(distances <= self.eps)[0]


    def _expand_cluster(self, X, labels, point_idx, neighbors, cluster_id):
        labels[point_idx] = cluster_id
        queue = list(neighbors)
        enqueued = set(neighbors)
        while len(queue) > 0:
            current = queue.pop(0)

            if labels[current] == -1:
                labels[current] = cluster_id

                current_neighbors = self._region_query(X, current)
                if len(current_neighbors) >= self.min_samples:
                    for neighbor in current_neighbors:
                        if neighbor not in enqueued:
                            queue.append(neighbor)
                            enqueued.add(neighbor)


    def fit(self, X):
        n = X.shape[0]
        labels = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        for point_idx in range(n):
            if labels[point_idx] != -1:
                continue
            vecindario = self._region_query(X,point_idx)
            if len(vecindario) < self.min_samples :
                continue
            result = self._expand_cluster(X, labels, point_idx, vecindario, cluster_id)
            cluster_id += 1
        self.labels_ = labels
        return self

def softmax(Z):
    Z = Z - Z.max(axis=1)[:, None]
    E = np.exp(Z)
    return E / E.sum(axis=1)[:, None]

import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return - np.mean(y * np.log(p) + (1-y) * np.log(1-p))

class NeuralNetwork:
    """Red neuronal de una capa oculta (tanh) y salida sigmoide, para clasificacion binaria 0/1."""

    def __init__(self, hidden_size=8, learning_rate=0.5, n_iterations=3000, seed=0):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.seed = seed
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.loss_history_ = []

    def _init_params(self, n_features):
        rng = np.random.default_rng(self.seed)
        self.W1 = rng.normal(0, 1, size=(n_features, self.hidden_size)) * np.sqrt(1 / n_features)
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = rng.normal(0, 1, size=self.hidden_size) * np.sqrt(1 / self.hidden_size)
        self.b2 = 0.0

    def _forward(self, X):
        Z1 = X.dot(self.W1) + self.b1
        A1 = np.tanh(Z1)
        Z2 = A1.dot(self.W2) + self.b2
        p = sigmoid(Z2)
        return A1 , p



    def predict_proba(self, X):
        a , p =self._forward(X)
        return p

    def predict(self, X):
        p = self.predict_proba(X)
        return (p >= 0.5).astype(int)

    def _backward(self, X, y, A1, p):
        n = X.shape[0]
        dZ2 = (p - y) / n
        dW2 = A1.T.dot(dZ2)
        dA1 = np.outer(dZ2, self.W2) #producto externo
        dZ1 = dA1 * (1 - A1 ** 2)
        dW1 = X.T.dot(dZ1)
        db1 = dZ1.sum(axis=0)
        db2 = dZ2.sum()
        return dW1, db1, dW2, db2


    def fit(self, X, y):
        self._init_params(X.shape[1])
        self.loss_history_ = []
        for i in range(self.n_iterations):
            A1, p = self._forward(X)
            self.loss_history_.append(binary_cross_entropy(y, p))
            dW1, db1, dW2, db2 = self._backward(X, y, A1, p)
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
        return self


class NeuralNetworkMulti(NeuralNetwork):

    def _init_params(self, n_features):
        super()._init_params(n_features)
        rng = np.random.default_rng(self.seed)
        self.W2 = rng.normal(0, 1, size=(self.hidden_size, self.n_classes)) * np.sqrt(1 / self.hidden_size)
        self.b2 = np.zeros(self.n_classes)

    def _forward(self, X):
        A1 = np.tanh(X.dot(self.W1) + self.b1)
        P = softmax(A1.dot(self.W2) + self.b2)
        return A1, P

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def _backward(self, X, idx, A1, P):
        n = X.shape[0]
        dZ2 = P.copy()
        dZ2[np.arange(n), idx] -= 1
        dZ2 /= n
        dW2 = A1.T.dot(dZ2)
        dA1 = dZ2.dot(self.W2.T)
        dZ1 = dA1 * (1 - A1 ** 2)
        dW1 = X.T.dot(dZ1)
        return dW1, dZ1.sum(axis=0), dW2, dZ2.sum(axis=0)

    def fit(self, X, y):
        self.classes_, idx = np.unique(y, return_inverse=True)
        self.n_classes = len(self.classes_)
        filas = np.arange(len(idx))
        self._init_params(X.shape[1])
        self.loss_history_ = []
        for _ in range(self.n_iterations):
            A1, P = self._forward(X)
            self.loss_history_.append(-np.mean(np.log(P[filas, idx] + 1e-12)))
            dW1, db1, dW2, db2 = self._backward(X, idx, A1, P)
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
        return self


class NeuralNetworkReg(NeuralNetwork):
    def _forward(self, X):
        A1 = np.tanh(X.dot(self.W1) + self.b1)
        return A1, A1.dot(self.W2) + self.b2

    def predict(self, X):
        return self.predict_proba(X) * self.y_std_ + self.y_mean_

    def _backward(self, X, y, A1, y_hat):
        n = X.shape[0]
        dZ2 = 2 * (y_hat - y) / n
        dW2 = A1.T.dot(dZ2)
        dA1 = np.outer(dZ2, self.W2)
        dZ1 = dA1 * (1 - A1 ** 2)
        return X.T.dot(dZ1), dZ1.sum(axis=0), dW2, dZ2.sum()

    def fit(self, X, y):
        self.y_mean_, self.y_std_ = y.mean(), y.std()
        y = (y - self.y_mean_) / self.y_std_
        self._init_params(X.shape[1])
        self.loss_history_ = []
        for _ in range(self.n_iterations):
            A1, y_hat = self._forward(X)
            self.loss_history_.append(np.mean((y_hat - y) ** 2))
            dW1, db1, dW2, db2 = self._backward(X, y, A1, y_hat)
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
        return self