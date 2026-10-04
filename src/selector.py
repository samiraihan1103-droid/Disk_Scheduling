"""Learned scheduler selector: a shallow decision tree over window features.
Confidence = fraction of training windows in the leaf whose best scheduler
matches the prediction (DecisionTreeClassifier.predict_proba)."""
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from schedulers import run_all, NAMES
from features import extract, FEATURES
from workload import GENERATORS, W
from schedulers import MAX_CYL


def best_scheduler(head, reqs):
    res = run_all(head, reqs)
    # ties resolve in NAMES order (FCFS first: cheapest to run)
    return min(NAMES, key=lambda n: res[n][1]), res


def make_dataset(n_per_type, seed, types=None):
    rng = np.random.default_rng(seed)
    X, y, wl, heads, windows = [], [], [], [], []
    for t in (types or list(GENERATORS)):
        for _ in range(n_per_type):
            h = int(rng.integers(0, MAX_CYL + 1))
            r = GENERATORS[t](rng, h)
            lab, _ = best_scheduler(h, r)
            X.append(extract(h, r)); y.append(lab); wl.append(t)
            heads.append(h); windows.append(r)
    return np.array(X), np.array(y), np.array(wl), heads, windows


def train(X, y, seed=0):
    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=seed)
    return clf.fit(X, y)


def predict(clf, head, reqs):
    p = clf.predict_proba([extract(head, reqs)])[0]
    i = int(np.argmax(p))
    return clf.classes_[i], float(p[i])


def describe(clf):
    return export_text(clf, feature_names=FEATURES)
