import numpy as np
import pandas as pd
from scipy.stats import entropy, rankdata
from catboost import CatBoostClassifier

from metric import precision_at_recall

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Загружаем данные. Даты парсим сразу, чтобы сравнивать окна и считать возраст.
train = pd.read_csv(
    'data/train.csv',
    parse_dates=['cookie_created_at', 'window_start_ts', 'window_end_ts']
)
test = pd.read_csv(
    'data/test.csv',
    parse_dates=['cookie_created_at', 'window_start_ts', 'window_end_ts']
)
events = pd.read_csv('data/events.csv.gz', parse_dates=['event_ts'])
# Бейзлайн: простая модель на двух базовых признаках из quickstart.
# Нужен, чтобы видеть нижнюю планку и обосновать улучшение.
from sklearn.ensemble import RandomForestClassifier

events_baseline = events.copy()
events_baseline = events_baseline.merge(
    train[['cookie_id', 'window_start_ts', 'window_end_ts']],
    on='cookie_id', how='inner'
)
events_baseline = events_baseline[
    (events_baseline.event_ts >= events_baseline.window_start_ts)
    & (events_baseline.event_ts < events_baseline.window_end_ts)
]

baseline_features = events_baseline.groupby('cookie_id').agg(
    n_events=('event_ts', 'count'),
    item_nunique=('item_id', 'nunique'),
).reset_index()

baseline_train = train[['cookie_id', 'window_start_ts', 'target']].merge(
    baseline_features, on='cookie_id', how='left'
).fillna(0)

baseline_is_valid = baseline_train['window_start_ts'].ge('2026-04-17').values
baseline_X = baseline_train[['n_events', 'item_nunique']]
baseline_y = baseline_train['target'].values

baseline_model = RandomForestClassifier(
    n_estimators=300, min_samples_leaf=3, random_state=42
)
baseline_model.fit(
    baseline_X.loc[~baseline_is_valid],
    baseline_y[~baseline_is_valid],
)
baseline_predictions = baseline_model.predict_proba(
    baseline_X.loc[baseline_is_valid]
)[:, 1]
baseline_score = precision_at_recall(
    baseline_y[baseline_is_valid], baseline_predictions
)
# События, характерные для человека: логин, контакт с продавцом, избранное.
# Бот-парсер их не делает, у него цель собрать данные, а не связаться.
human_events = [
    'login', 'contact_phone_show', 'contact_chat_open',
    'contact_message_sent', 'favorite_add',
]

# Показ капчи означает, что антибот уже заподозрил автоматизацию.
captcha_event = 'captcha_shown'

# Круглые паузы, характерные для sleep() в парсерах.
round_deltas = [0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]


def events_in_window(events_df, meta_df):
    """Оставляет только события внутри окна наблюдения каждой куки.

    Условие из задачи: window_start_ts <= event_ts < window_end_ts.
    Всё, что было до или после окна, в признаки не попадает.
    """
    ev = events_df.merge(
        meta_df[['cookie_id', 'window_start_ts', 'window_end_ts']],
        on='cookie_id',
        how='inner',
    )
    in_window = (ev.event_ts >= ev.window_start_ts) & (ev.event_ts < ev.window_end_ts)
    return ev[in_window].copy()


def shannon_entropy(values):
    """Энтропия Шеннона по дискретному распределению. Ноль, если пусто."""
    if len(values) == 0:
        return 0.0
    counts = pd.Series(values).value_counts()
    if len(counts) == 0:
        return 0.0
    return float(entropy(counts))


def delta_entropy_log(series):
    """Энтропия интервалов между событиями.

    Интервалы округляем до логарифмических бинов. У бота интервалы повторяются
    (циклы запрос-пауза), энтропия низкая. У человека интервалы разнообразные,
    энтропия выше.
    """
    clean = series.dropna()
    clean = clean[clean >= 0]
    if len(clean) < 2:
        return 0.0
    binned = np.round(np.log1p(clean) * 4).astype('int64')
    return shannon_entropy(binned.values)


def delta_mode_share_log(series):
    """Доля самого частого интервала среди всех интервалов куки.

    У бота эта доля высокая: паузы фиксированной длины повторяются.
    У человека распределение гладкое, доля одного бина низкая.
    """
    clean = series.dropna()
    clean = clean[clean >= 0]
    if len(clean) < 3:
        return 0.0
    binned = np.round(np.log1p(clean) * 4).astype('int64')
    counts = pd.Series(binned).value_counts()
    return float(counts.iloc[0] / len(clean))


def is_round_delta(value):
    """Интервал близок к одной из типовых пауз парсера."""
    if pd.isna(value):
        return 0
    close_to_round = [abs(value - r) < 0.02 for r in round_deltas]
    if any(close_to_round):
        return 1
    return 0


def extract_features(ev_df, meta_df):
    """Считает признаки по событиям внутри окна и присоединяет их к метаданным куки."""
    ev = ev_df.sort_values(['cookie_id', 'event_ts']).copy()

    # Чистка категориальных полей: пропуски в unknown, убираем лишние пробелы.
    for col in ['event_name', 'platform', 'seller_type', 'item_category', 'item_location']:
        if col in ev.columns:
            ev[col] = ev[col].fillna('unknown').astype(str).str.strip()

    # Платформы приходят в разном регистре (WEB, Web, web). Без нормализации
    # одна и та же платформа распадается на несколько значений и сигнал размывается.
    if 'platform' in ev.columns:
        ev['platform'] = ev['platform'].str.lower()

    # --- Интервалы между последовательными событиями одной куки ---
    ev['prev_ts'] = ev.groupby('cookie_id')['event_ts'].shift(1)
    ev['delta_ts'] = (ev['event_ts'] - ev['prev_ts']).dt.total_seconds()
    ev['is_exact_zero'] = (ev['delta_ts'] == 0).astype(int)
    ev['is_fast_click'] = ((ev['delta_ts'] > 0) & (ev['delta_ts'] < 0.08)).astype(int)
    ev['is_round_delta'] = ev['delta_ts'].apply(is_round_delta).astype(int)
    ev['log_delta'] = np.log1p(ev['delta_ts'].clip(lower=0))
    ev['is_very_long_pause'] = (ev['delta_ts'] > 300).astype(int)

    # --- Поведение курсора ---
    # Считаем только по событиям с координатами. Между двумя такими событиями
    # может быть десятки других без координат, иначе получится мусор.
    pointer_events = ev[ev['pointer_x'].notnull() & ev['pointer_y'].notnull()].copy()

    if len(pointer_events) > 0:
        pointer_events = pointer_events.sort_values(['cookie_id', 'event_ts'])
        pointer_grouped = pointer_events.groupby('cookie_id')

        pointer_events['dx'] = pointer_grouped['pointer_x'].diff()
        pointer_events['dy'] = pointer_grouped['pointer_y'].diff()
        pointer_events['distance'] = np.hypot(pointer_events['dx'], pointer_events['dy'])
        pointer_events['dt'] = pointer_grouped['event_ts'].diff().dt.total_seconds()
        pointer_events['speed'] = pointer_events['distance'] / (pointer_events['dt'] + 1e-5)

        pointer_events['angle'] = np.arctan2(pointer_events['dy'], pointer_events['dx'])
        pointer_events['angle_diff'] = pointer_grouped['angle'].diff().abs()
        pointer_events['direction_change'] = (pointer_events['angle_diff'] > np.pi / 2).astype(int)

        # Телепорт мыши: скачок на сотни пикселей, скорость выше 5000 px/s.
        # Признак грубой эмуляции.
        pointer_events['is_teleport'] = (pointer_events['speed'] > 5000).astype(int)

        # Жёсткий обрез: выше 200 пикселей за событие нетипично для человека.
        pointer_events['dist_hard'] = pointer_events['distance'].clip(upper=200)

        # Огромные и средние перемещения как отдельные сигналы.
        pointer_events['is_huge_move'] = (pointer_events['distance'] > 1000).astype(int)
        pointer_events['is_mid_move'] = (
            (pointer_events['distance'] > 200) & (pointer_events['distance'] <= 1000)
        ).astype(int)

        pointer_events['log_dist'] = np.log1p(pointer_events['distance'])
        pointer_events['log_speed'] = np.log1p(pointer_events['speed'].clip(0, 1e6))

        pointer_features = pointer_events.groupby('cookie_id').agg(
            ptr_count=('distance', 'count'),
            dist_mean=('distance', 'mean'),
            dist_std=('distance', 'std'),
            dist_clip_mean=('dist_hard', 'mean'),
            dist_clip_std=('dist_hard', 'std'),
            dist_hard_q95=('dist_hard', lambda s: s.quantile(0.95)),
            huge_move_ratio=('is_huge_move', 'mean'),
            mid_move_ratio=('is_mid_move', 'mean'),
            teleport_ratio=('is_teleport', 'mean'),
            log_dist_mean=('log_dist', 'mean'),
            log_dist_std=('log_dist', 'std'),
            log_speed_mean=('log_speed', 'mean'),
            log_speed_std=('log_speed', 'std'),
            speed_max=('speed', 'max'),
            angle_std=('angle', 'std'),
            direction_change_ratio=('direction_change', 'mean'),
        ).reset_index()
    else:
        pointer_features = pd.DataFrame({'cookie_id': meta_df.cookie_id})

    # --- Признаки User-Agent ---
    # Разбираем на структурные части и ищем явные маркеры автоматизации.
    # Реальные сервисы маскируются, поэтому маркер срабатывает редко.
    if 'user_agent' in ev.columns:
        user_agent = ev['user_agent'].fillna('').astype(str).str.lower()

        ev['ua_len'] = user_agent.str.len()
        ev['ua_is_mozilla'] = user_agent.str.startswith('mozilla').astype(int)
        ev['ua_has_chrome'] = user_agent.str.contains('chrome').astype(int)
        ev['ua_has_firefox'] = user_agent.str.contains('firefox').astype(int)
        ev['ua_has_safari'] = user_agent.str.contains('safari').astype(int)
        ev['ua_has_mobile'] = user_agent.str.contains('mobile|android|iphone').astype(int)
        ev['ua_is_obvious_bot'] = user_agent.str.contains(
            r'headless|phantom|selenium|puppeteer|playwright|'
            r'python|requests|curl|wget|scrapy|httpx|aiohttp|go-http|java/'
        ).astype(int)

        user_agent_features = ev.groupby('cookie_id').agg(
            ua_nunique=('user_agent', 'nunique'),
            ua_len_mean=('ua_len', 'mean'),
            ua_is_mozilla_ratio=('ua_is_mozilla', 'mean'),
            ua_chrome_ratio=('ua_has_chrome', 'mean'),
            ua_firefox_ratio=('ua_has_firefox', 'mean'),
            ua_safari_ratio=('ua_has_safari', 'mean'),
            ua_mobile_ratio=('ua_has_mobile', 'mean'),
            ua_obvious_bot=('ua_is_obvious_bot', 'max'),
        ).reset_index()
    else:
        user_agent_features = pd.DataFrame({'cookie_id': meta_df.cookie_id})

    # --- Базовые агрегаты и статистики интервалов ---
    base_features = ev.groupby('cookie_id').agg(
        n_events=('event_ts', 'count'),
        item_nunique=('item_id', 'nunique'),
        query_nunique=('search_query', 'nunique'),
        category_nunique=('item_category', 'nunique'),
        location_nunique=('item_location', 'nunique'),
        seller_nunique=('seller_type', 'nunique'),
        platform_nunique=('platform', 'nunique'),
        search_page_max=('search_page', 'max'),
        search_page_mean=('search_page', 'mean'),
        has_pointer_ratio=('pointer_x', lambda s: float(s.notnull().mean())),

        zero_delta_ratio=('is_exact_zero', 'mean'),
        fast_click_ratio=('is_fast_click', 'mean'),
        round_delta_ratio=('is_round_delta', 'mean'),
        very_long_pause_ratio=('is_very_long_pause', 'mean'),

        delta_ts_mean=('delta_ts', 'mean'),
        delta_ts_std=('delta_ts', 'std'),
        delta_ts_median=('delta_ts', 'median'),
        delta_ts_min=('delta_ts', 'min'),
        delta_ts_max=('delta_ts', 'max'),
        delta_ts_q25=('delta_ts', lambda s: s.quantile(0.25)),
        delta_ts_q75=('delta_ts', lambda s: s.quantile(0.75)),

        log_delta_mean=('log_delta', 'mean'),
        log_delta_std=('log_delta', 'std'),
        log_delta_median=('log_delta', 'median'),

        active_duration=('event_ts', lambda x: (x.max() - x.min()).total_seconds()),
    ).reset_index()

    base_features['events_per_sec'] = (
        base_features['n_events'] / (base_features['active_duration'] + 1.0)
    )
    base_features['item_per_event'] = (
        base_features['item_nunique'] / (base_features['n_events'] + 1e-5)
    )
    base_features['query_per_event'] = (
        base_features['query_nunique'] / (base_features['n_events'] + 1e-5)
    )
    base_features['category_per_item'] = (
        base_features['category_nunique'] / (base_features['item_nunique'] + 1e-5)
    )
    base_features['delta_ts_cv'] = (
        base_features['delta_ts_std'] / (base_features['delta_ts_mean'] + 1e-5)
    )
    base_features['delta_ts_iqr'] = (
        base_features['delta_ts_q75'] - base_features['delta_ts_q25']
    )

    # --- Регулярность интервалов ---
    delta_features = ev.groupby('cookie_id')['delta_ts'].agg(
        delta_entropy_log=delta_entropy_log,
        delta_mode_share_log=delta_mode_share_log,
    ).reset_index()

    # --- Энтропии по категориальным полям ---
    entropy_features = ev.groupby('cookie_id').agg(
        event_entropy=('event_name', shannon_entropy),
        category_entropy=('item_category', shannon_entropy),
        platform_entropy=('platform', shannon_entropy),
        location_entropy=('item_location', shannon_entropy),
    ).reset_index()

    # --- Бинарные флаги "было ли событие у куки" ---
    # Отсутствие человеческих событий сильный сигнал бота.
    ev['one'] = 1
    has_event = ev.pivot_table(
        index='cookie_id',
        columns='event_name',
        values='one',
        aggfunc='max',
        fill_value=0,
    )
    has_event.columns = ['has_' + str(c) for c in has_event.columns]
    has_event = has_event.reset_index()

    present_human = [c for c in human_events if c in ev['event_name'].unique()]

    if present_human:
        human_columns = ['has_' + c for c in present_human]
        has_event['human_total'] = has_event[human_columns].sum(axis=1)
        all_has_columns = [c for c in has_event.columns if c.startswith('has_')]
        has_event['human_ratio'] = (
            has_event['human_total'] / (has_event[all_has_columns].sum(axis=1) + 1e-5)
        )
        has_event['human_zero'] = (has_event['human_total'] == 0).astype(int)
    else:
        has_event['human_total'] = 0
        has_event['human_ratio'] = 0.0
        has_event['human_zero'] = 1

    # --- Капча ---
    if captcha_event in ev['event_name'].unique():
        ev['is_captcha'] = (ev['event_name'] == captcha_event).astype(int)
    else:
        ev['is_captcha'] = 0

    captcha_features = ev.groupby('cookie_id').agg(
        captcha_count=('is_captcha', 'sum'),
        captcha_ratio=('is_captcha', 'mean'),
    ).reset_index()

    # --- Доли типов событий и платформ ---
    event_counts = pd.crosstab(
        ev['cookie_id'], ev['event_name'], normalize='index'
    ).reset_index()
    event_counts.columns = ['cookie_id'] + [
        'ev_' + str(c) for c in event_counts.columns[1:]
    ]

    platform_counts = pd.crosstab(
        ev['cookie_id'], ev['platform'], normalize='index'
    ).reset_index()
    platform_counts.columns = ['cookie_id'] + [
        'pl_' + str(c) for c in platform_counts.columns[1:]
    ]

    # --- Контактные композиты ---
    # Бот просматривает много объявлений, но никогда не показывает телефон,
    # не открывает чат и не пишет продавцу.
    contact_columns = [
        'ev_contact_phone_show',
        'ev_contact_chat_open',
        'ev_contact_message_sent',
    ]
    contact_columns = [c for c in contact_columns if c in event_counts.columns]

    if contact_columns:
        event_counts['contact_score'] = event_counts[contact_columns].sum(axis=1)
        event_counts['contact_zero'] = (event_counts['contact_score'] == 0).astype(int)
        event_counts['contact_per_view'] = (
            event_counts['contact_score'] / (event_counts.get('ev_item_view', 0) + 1e-5)
        )
    else:
        event_counts['contact_score'] = 0
        event_counts['contact_zero'] = 1
        event_counts['contact_per_view'] = 0

    engagement_columns = ['ev_favorite_add', 'ev_seller_page_view']
    engagement_columns = [c for c in engagement_columns if c in event_counts.columns]
    if engagement_columns:
        event_counts['engagement_score'] = event_counts[engagement_columns].sum(axis=1)
    else:
        event_counts['engagement_score'] = 0

    # --- Смена категории между последовательными событиями ---
    ev_sorted = ev.sort_values(['cookie_id', 'event_ts'])
    ev_sorted['prev_cat'] = ev_sorted.groupby('cookie_id')['item_category'].shift(1)
    ev_sorted['cat_switch'] = (
        ev_sorted['item_category'] != ev_sorted['prev_cat']
    ).astype(int)

    cat_switch_features = ev_sorted.groupby('cookie_id').agg(
        cat_switch_ratio=('cat_switch', 'mean'),
    ).reset_index()

    # --- Повторные просмотры одного объявления и одной категории ---
    item_repeat = ev.groupby(['cookie_id', 'item_id']).size().reset_index(name='cnt')
    item_repeat_features = item_repeat.groupby('cookie_id')['cnt'].agg(
        item_max_repeat='max',
        item_repeat_ratio=lambda s: (s > 1).mean(),
    ).reset_index()

    cat_repeat = ev.groupby(['cookie_id', 'item_category']).size().reset_index(name='cnt')
    cat_repeat_features = cat_repeat.groupby('cookie_id')['cnt'].agg(
        cat_max_repeat='max',
        cat_dominant_share=lambda s: s.max() / s.sum(),
    ).reset_index()

    # --- Интервалы отдельно по типам событий ---
    # У бота item_view идёт строго за item_view с одинаковой задержкой.
    delta_by_event = None
    for event_type in ['item_view', 'search_results_view', 'photo_swipe']:
        subset = ev[ev['event_name'] == event_type]
        if len(subset) == 0:
            continue
        agg = subset.groupby('cookie_id')['delta_ts'].agg(**{
            f'delta_{event_type}_median': 'median',
            f'delta_{event_type}_std': 'std',
            f'delta_{event_type}_q25': lambda s: s.quantile(0.25),
        }).reset_index()
        if delta_by_event is None:
            delta_by_event = agg
        else:
            delta_by_event = delta_by_event.merge(agg, on='cookie_id', how='outer')

    if delta_by_event is None:
        delta_by_event = pd.DataFrame({'cookie_id': meta_df.cookie_id})

    # --- Переходы между типами событий ---
    # Бот ходит одним маршрутом, доля самого частого перехода высокая.
    ev_sorted['prev_event'] = ev_sorted.groupby('cookie_id')['event_name'].shift(1)
    ev_sorted['transition'] = (
        ev_sorted['prev_event'].astype(str) + '→' + ev_sorted['event_name'].astype(str)
    )
    ev_sorted['event_switch'] = (
        ev_sorted['event_name'] != ev_sorted['prev_event']
    ).astype(int)

    switch_features = ev_sorted.groupby('cookie_id').agg(
        event_switch_ratio=('event_switch', 'mean'),
    ).reset_index()

    transition_mode_share = ev_sorted.groupby('cookie_id')['transition'].apply(
        lambda s: s.value_counts().iloc[0] / len(s) if len(s) > 0 else 0
    ).rename('transition_mode_share').reset_index()

    # --- Время просмотра карточки ---
    # Считаем интервал от item_view до следующего события куки.
    ev_sorted['next_ts'] = ev_sorted.groupby('cookie_id')['event_ts'].shift(-1)
    ev_sorted['next_delta'] = (
        ev_sorted['next_ts'] - ev_sorted['event_ts']
    ).dt.total_seconds()

    item_view_events = ev_sorted[ev_sorted['event_name'] == 'item_view']
    if len(item_view_events) > 0:
        item_view_time = item_view_events.groupby('cookie_id')['next_delta'].agg(
            item_view_time_median='median',
            item_view_time_std='std',
        ).reset_index()
    else:
        item_view_time = pd.DataFrame({'cookie_id': meta_df.cookie_id})

    # --- Первое и последнее событие куки ---
    first_last_features = ev.groupby('cookie_id').agg(
        first_event=('event_name', 'first'),
        last_event=('event_name', 'last'),
        first_platform=('platform', 'first'),
    ).reset_index()

    # --- Сборка всех признаков ---
    features = base_features
    additional = [
        pointer_features,
        user_agent_features,
        delta_features,
        entropy_features,
        has_event,
        captcha_features,
        event_counts,
        platform_counts,
        cat_switch_features,
        item_repeat_features,
        cat_repeat_features,
        delta_by_event,
        item_view_time,
        switch_features,
        transition_mode_share,
        first_last_features,
    ]

    for extra in additional:
        features = features.merge(extra, on='cookie_id', how='left')

    # --- Метаданные куки ---
    result = meta_df.merge(features, on='cookie_id', how='left')

    result['age_at_window_start'] = (
        result['window_start_ts'] - result['cookie_created_at']
    ).dt.total_seconds()
    result['log_age'] = np.log1p(result['age_at_window_start'].clip(lower=0))

    result['window_duration'] = (
        result['window_end_ts'] - result['window_start_ts']
    ).dt.total_seconds()
    result['has_zero_events'] = (result['n_events'].fillna(0) == 0).astype(int)

    # Числовые пропуски заполняем нулём, чтобы модель не видела NaN.
    for column in result.columns:
        is_numeric = result[column].dtype.kind in 'fi'
        if is_numeric and column != 'target':
            result[column] = result[column].fillna(0)

    return result


events_train = events_in_window(events, train)
events_test = events_in_window(events, test)

Xtr = extract_features(events_train, train)
Xte = extract_features(events_test, test)

drop_columns = [
    'cookie_id', 'cookie_created_at', 'window_start_ts', 'window_end_ts', 'target'
]
feature_columns = [c for c in Xtr.columns if c not in drop_columns]

categorical_features = [
    c for c in ['first_event', 'last_event', 'first_platform'] if c in feature_columns
]
for column in categorical_features:
    Xtr[column] = Xtr[column].astype(str)
    Xte[column] = Xte[column].astype(str)

X = Xtr[feature_columns]
y = train['target'].values
X_test_final = Xte[feature_columns]


def fit_catboost(X_train, y_train, X_valid, y_valid, spw, depth, l2=8):
    """Обучение CatBoost с ранним остановом по PRAUC.

    PRAUC ставим метрикой, потому что итоговая метрика задачи оценивает
    качество ранжирования в верхней части распределения.
    """
    model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=depth,
        l2_leaf_reg=l2,
        random_seed=RANDOM_SEED,
        scale_pos_weight=spw,
        eval_metric='PRAUC',
        cat_features=categorical_features if categorical_features else None,
        verbose=0,
        od_type='Iter',
        od_wait=100,
    )
    model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
    )
    return model


# Валидация по времени на трёх точках отсечения внутри трейна.
# Случайный сплит использовать нельзя: тест лежит позже трейна,
# и модель увидела бы будущее.
cutoffs = ['2026-04-15', '2026-04-17', '2026-04-19']

best_score = -1.0
best_spw = 1.0
best_depth = 5

for spw in [1.0, 2.0]:
    for depth in [4, 5]:
        fold_scores = []
        for cutoff in cutoffs:
            is_validation = train['window_start_ts'].ge(cutoff).values
            if is_validation.sum() < 100:
                continue
            if (~is_validation).sum() < 100:
                continue

            model = fit_catboost(
                X.loc[~is_validation], y[~is_validation],
                X.loc[is_validation], y[is_validation],
                spw, depth, l2=8,
            )
            predictions = model.predict_proba(X.loc[is_validation])[:, 1]
            fold_scores.append(precision_at_recall(y[is_validation], predictions))

        if len(fold_scores) == 0:
            continue

        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_spw = spw
            best_depth = depth

# Финальное обучение на всём трейне.
best_iterations = model.get_best_iteration()
if not best_iterations:
    best_iterations = 800
n_iterations = max(best_iterations, 500)

# Ансамбль из нескольких CatBoost с разными seed. Усредняем по рангам,
# а не по вероятностям: разные seed могут быть по-разному откалиброваны,
# а ранги всегда сопоставимы.
all_predictions = []
for seed in [42, 7, 101, 2024, 777]:
    ensemble_model = CatBoostClassifier(
        iterations=n_iterations,
        learning_rate=0.03,
        depth=best_depth,
        l2_leaf_reg=8,
        random_seed=seed,
        scale_pos_weight=best_spw,
        cat_features=categorical_features if categorical_features else None,
        verbose=0,
    )
    ensemble_model.fit(X, y)
    all_predictions.append(ensemble_model.predict_proba(X_test_final)[:, 1])

ranked_predictions = []
for predictions in all_predictions:
    ranked_predictions.append(rankdata(predictions) / len(predictions))

final_scores = np.mean(ranked_predictions, axis=0)

submission = pd.DataFrame({
    'cookie_id': Xte['cookie_id'],
    'score': final_scores,
})

assert len(submission) == len(test)
assert submission['cookie_id'].is_unique
assert submission['score'].between(0, 1).all()

submission.to_csv('submission.csv', index=False)