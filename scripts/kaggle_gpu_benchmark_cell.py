"""Paste this entire file into one Kaggle cell after loading the raw tables."""

from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter

import cupy as cp
import nltk
import numpy as np
import pandas as pd
from cuml.linear_model import LogisticRegression as GpuLogisticRegression
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression as CpuLogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


MODEL_C = 1.0  # Replace with study.best_params['C'] to reuse your tuned value.
MAX_ITER = 1500
VALID_FRACTION = 0.20
CHAT_COLUMNS = ['radiant_chat', 'dire_chat']
ADV_COLUMNS = {'xp': 'radiant_exp_adv', 'gold': 'radiant_gold_adv'}
WINDOWS = ((1, 4), (4, 7), (7, 10), (10, 13), (13, 16))


def check_gpu_sparse_support():
    # This also warms up CUDA before the benchmark.
    cp.cuda.Device().use()
    warmup_X = csr_matrix(np.eye(4, dtype=np.float32))
    warmup_y = np.array([0, 1, 0, 1], dtype=np.float32)
    gpu = GpuLogisticRegression(max_iter=10, output_type='numpy')
    try:
        gpu.fit(warmup_X, warmup_y)
        gpu.predict_proba(warmup_X)
    except (TypeError, ValueError, NotImplementedError) as exc:
        raise RuntimeError(
            'The full benchmark needs cuML LogisticRegression with CSR support. '
            'Use a RAPIDS build that supports sparse fit and predict_proba.'
        ) from exc
    CpuLogisticRegression().fit(warmup_X, warmup_y).predict_proba(warmup_X)
    cp.cuda.Device().synchronize()


def clean_rosters(raw_players, match_ids, hero_ids):
    # Drop exact duplicate records before reducing to the four relevant columns.
    roster = raw_players.loc[raw_players['match_id'].isin(match_ids)].drop_duplicates()
    roster = roster[['match_id', 'hero_id', 'player_slot', 'account_id']].copy()

    valid_slots = [*range(5), *range(128, 133)]
    invalid = ~roster['hero_id'].isin(hero_ids) | ~roster['player_slot'].isin(valid_slots)
    bad_ids = pd.Index(roster.loc[invalid, 'match_id'].unique())
    roster = roster.loc[~roster['match_id'].isin(bad_ids)].copy()
    roster['radiant'] = roster['player_slot'].isin(range(5))

    known = roster['account_id'].notna() & ~roster['account_id'].isin([-1, 4294967295])
    accounts = roster.loc[known, ['match_id', 'account_id', 'radiant']]
    repeated = accounts.duplicated(['match_id', 'account_id'], keep=False)
    sides = accounts.loc[repeated].groupby(['match_id', 'account_id'])['radiant'].nunique()
    both_sides = sides.loc[sides == 2].index.get_level_values('match_id')

    counts = roster.groupby('match_id').agg(
        total=('player_slot', 'size'),
        slots=('player_slot', 'nunique'),
        heroes=('hero_id', 'nunique'),
    )
    complete = counts.index[(counts == 10).all(axis=1)].difference(both_sides)
    return roster.loc[roster['match_id'].isin(complete), [
        'match_id', 'hero_id', 'radiant',
    ]], complete


class HeroLineups(TransformerMixin, BaseEstimator):
    def __init__(self, roster, hero_ids):
        self.roster = roster
        self.hero_ids = hero_ids

    def fit(self, X, y=None):
        self.hero_index_ = pd.Index(self.hero_ids)
        return self

    def transform(self, X):
        match_index = pd.Index(X['match_id'])
        roster = self.roster.loc[self.roster['match_id'].isin(match_index)]
        rows = match_index.get_indexer(roster['match_id'])
        columns = self.hero_index_.get_indexer(roster['hero_id'])
        signs = np.where(roster['radiant'], 1, -1).astype(np.float32)
        return csr_matrix(
            (signs, (rows, columns)),
            shape=(len(X), len(self.hero_index_)),
        )


def summarize_advantage(raw):
    values = np.array([]) if pd.isna(raw) else np.array(raw.strip().strip('[]').split(), dtype=float)
    if not values.size:
        return (np.nan,) * (6 + len(WINDOWS))

    radiant = values[values > 0]
    dire = -values[values < 0]
    last = values[-1]
    radiant_median = np.median(radiant) if radiant.size else 0
    dire_median = np.median(dire) if dire.size else 0

    slope = intercept = r2 = np.nan
    if values.size >= 2:
        t = np.arange(values.size)
        centered_t = t - t.mean()
        centered_values = values - values.mean()
        slope = np.dot(centered_t, centered_values) / np.dot(centered_t, centered_t)
        intercept = values.mean() - slope * t.mean()
        residuals = centered_values - slope * centered_t
        total = np.dot(centered_values, centered_values)
        r2 = 1 - np.dot(residuals, residuals) / total if total else 1.0

    means = [values[start:stop].mean() if start < values.size else np.nan
             for start, stop in WINDOWS]
    return (last, radiant_median, dire_median, slope, intercept, r2, *means)


def add_features(frame):
    frame = frame.copy()
    frame['day_of_week'] = frame['date'].dt.dayofweek
    frame['mmr_missing'] = frame['avg_mmr'].isna().astype('float32')
    for prefix, name in ADV_COLUMNS.items():
        new_columns = [f'{prefix}_{suffix}' for suffix in ADV_SUFFIXES]
        frame[new_columns] = pd.DataFrame(
            frame[name].map(summarize_advantage).tolist(),
            index=frame.index,
            columns=new_columns,
        )
    return frame


def prepare_tokenizer():
    packages = {'pymorphy3': 'pymorphy3', 'category_encoders': 'category-encoders'}
    missing = [package for module, package in packages.items() if find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])
    try:
        wordnet.ensure_loaded()
    except LookupError:
        nltk.download('wordnet', quiet=True, raise_on_error=True)
        wordnet.ensure_loaded()


def preprocess_chat(text):
    return [
        lemmatize_token(token)
        for token in tokenizer.tokenize(text)
        if re.fullmatch(r'[.,|]+', token) is None
    ]


@lru_cache(maxsize=100_000)
def lemmatize_token(token):
    return lemmatizer.lemmatize(morph.parse(token)[0].normal_form)


def benchmark(model, use_gpu):
    if use_gpu:
        cp.cuda.Device().synchronize()
    start = perf_counter()
    model.fit(train_matrix, target_train)
    if use_gpu:
        cp.cuda.Device().synchronize()
    fitted = perf_counter()

    positive = np.flatnonzero(np.asarray(model.classes_) == 1).item()
    probabilities = model.predict_proba(valid_matrix)[:, positive]
    if use_gpu:
        cp.cuda.Device().synchronize()
    predicted = perf_counter()
    return {
        'fit_s': fitted - start,
        'predict_s': predicted - fitted,
        'fit_and_predict_s': predicted - start,
        'gini': 2 * roc_auc_score(target_valid, probabilities) - 1,
    }


check_gpu_sparse_support()
prepare_tokenizer()
from category_encoders import OneHotEncoder
from pymorphy3 import MorphAnalyzer

if 'chat' not in globals():
    chat_paths = list(Path('/kaggle/input').rglob('game_chat.csv'))
    if len(chat_paths) != 1:
        raise ValueError('Load game_chat.csv into chat or attach exactly one copy via Add Input.')
    chat = pd.read_csv(chat_paths[0])

tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
lemmatizer = WordNetLemmatizer()
morph = MorphAnalyzer()
ADV_SUFFIXES = [
    'last', 'radiant_median', 'dire_median', 'slope', 'intercept', 'r2',
    *[f'mean_{start}_{stop - 1}' for start, stop in WINDOWS],
]

start_preprocessing = perf_counter()
hero_ids = heroes['id'].dropna().unique()
roster, clean_match_ids = clean_rosters(players, matches_train['match_id'], hero_ids)
matches = matches_train.loc[
    matches_train['match_id'].isin(clean_match_ids),
    ['match_id', 'date', 'region', 'avg_mmr', 'radiant_win'],
].copy()
matches['date'] = pd.to_datetime(matches['date'], errors='raise')
matches = (
    matches
    .merge(chat[['match_id', *CHAT_COLUMNS]], on='match_id', how='left', validate='one_to_one')
    .merge(dota_adv[['match_id', *ADV_COLUMNS.values()]], on='match_id', how='left', validate='one_to_one')
    .sort_values('date')
)
matches[CHAT_COLUMNS] = matches[CHAT_COLUMNS].fillna('')
cutoff = matches['date'].iloc[int((1 - VALID_FRACTION) * len(matches))]
train = matches.loc[matches['date'] < cutoff]
valid = matches.loc[matches['date'] >= cutoff]
target_train = train['radiant_win'].to_numpy(dtype=np.float32)
target_valid = valid['radiant_win'].to_numpy(dtype=np.float32)

mmr = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('sqrt', FunctionTransformer(np.sqrt)),
    ('scale', StandardScaler()),
])
adv = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler()),
])
text_branches = [
    (column, TfidfVectorizer(
        tokenizer=preprocess_chat, token_pattern=None, lowercase=False, dtype=np.float32,
    ), column)
    for column in CHAT_COLUMNS
]
transformer = ColumnTransformer([
    ('categories', OneHotEncoder(cols=['region', 'day_of_week']), ['region', 'day_of_week']),
    ('mmr', mmr, ['avg_mmr']),
    ('flags', 'passthrough', ['mmr_missing']),
    ('heroes', HeroLineups(roster, hero_ids), ['match_id']),
    *text_branches,
    ('advantage', adv, [f'{prefix}_{suffix}' for prefix in ADV_COLUMNS for suffix in ADV_SUFFIXES]),
], sparse_threshold=1.0)
preprocessor = Pipeline([
    ('add_features', FunctionTransformer(add_features)),
    ('columns', transformer),
])

train_matrix = csr_matrix(preprocessor.fit_transform(train), dtype=np.float32)
valid_matrix = csr_matrix(preprocessor.transform(valid), dtype=np.float32)
preprocessing_s = perf_counter() - start_preprocessing
print(f'OOT cutoff: {cutoff.date()} | train {train_matrix.shape} | valid {valid_matrix.shape}')
print(f'C={MODEL_C} | shared preprocessing: {preprocessing_s:.2f} s', flush=True)

results = pd.DataFrame.from_dict({
    'sklearn CPU': benchmark(
        CpuLogisticRegression(C=MODEL_C, solver='lbfgs', max_iter=MAX_ITER), use_gpu=False,
    ),
    'cuML GPU': benchmark(
        GpuLogisticRegression(C=MODEL_C, max_iter=MAX_ITER, output_type='numpy'), use_gpu=True,
    ),
}, orient='index')
results['total_with_preprocessing_s'] = results['fit_and_predict_s'] + preprocessing_s
print(results.round(4).to_string())
