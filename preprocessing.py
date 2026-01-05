import numpy as np
import polars as pl
from datetime import timedelta


def create_time_bins(
    df,
    bin_size_days,
    skill_names,
    origin_date=None,
    end_date=None,
    trim_before_first=True,
    max_inactive_years=1
):
    """
    Bin observations by time and aggregate.
    
    Parameters
    ----------
    df : polars.DataFrame
        Columns: [player_id, timestamp, skill_name, observed_value]
    bin_size_days : int
        Width of each time bin in days
    skill_names : list
        All skill names in the model
    origin_date : datetime, optional
        Fixed origin for bin indexing. If None, uses min timestamp in df.
    end_date : datetime, optional
        End date for bins. If None, uses max timestamp in df.
    trim_before_first : bool, optional
        If True, drop bins before each player's first observation.
    max_inactive_years : float or None, optional
        If set, include at most this many years after a player's last observation.
    
    Returns
    -------
    binned_data : dict
        player_id -> list of observation dicts per bin
    time_bins : list
        (bin_start, bin_end) tuples
    origin_date : datetime
        The origin used for bin indexing (for storage)
    player_start_bins : dict
        Mapping player_id -> first bin index in that player's sequence
    """
    df = df.with_columns(
        pl.col('timestamp').cast(pl.Datetime).alias('timestamp')
    )
    
    if origin_date is None:
        origin_date = df.select(pl.col('timestamp').min()).item()
    
    if end_date is None:
        end_date = df.select(pl.col('timestamp').max()).item()
    
    bin_delta = timedelta(days=bin_size_days)
    bin_starts = []
    current = origin_date
    while current <= end_date:
        bin_starts.append(current)
        current = current + bin_delta
    
    n_bins = len(bin_starts)
    if n_bins == 0:
        n_bins = 1
        bin_starts = [origin_date]
    
    time_bins = [(bin_starts[i], bin_starts[i] + bin_delta) for i in range(n_bins)]
    
    df = df.with_columns(
        ((pl.col('timestamp') - origin_date).dt.total_days() // bin_size_days)
        .cast(pl.Int32)
        .alias('bin_idx')
    )
    
    df = df.filter((pl.col('bin_idx') >= 0) & (pl.col('bin_idx') < n_bins))
    
    agg_df = df.group_by(['player_id', 'bin_idx', 'skill_name']).agg(
        pl.col('observed_value').mean().alias('mean_value'),
        pl.col('observed_value').len().alias('n_obs')
    )
    
    n_skills = len(skill_names)
    
    player_bin_stats = agg_df.group_by('player_id').agg(
        pl.col('bin_idx').min().alias('first_bin'),
        pl.col('bin_idx').max().alias('last_bin')
    )

    if max_inactive_years is not None:
        max_gap_days = int(round(max_inactive_years * 365))
        max_gap_bins = max(0, max_gap_days // bin_size_days)
    else:
        max_gap_bins = None

    if trim_before_first:
        start_expr = pl.col('first_bin')
    else:
        start_expr = pl.lit(0)

    if max_gap_bins is None:
        end_expr = pl.col('last_bin')
    else:
        end_expr = pl.col('last_bin') + pl.lit(max_gap_bins)

    player_bin_stats = player_bin_stats.with_columns(
        start_expr.alias('start_bin'),
        end_expr.clip(upper_bound=n_bins - 1).alias('end_bin')
    )

    # Build full index of (player_id, bin_idx) combinations
    full_index = (
        player_bin_stats
        .with_columns(
            pl.int_ranges(
                pl.col('start_bin'),
                pl.col('end_bin') + 1,
                dtype=pl.Int32
            ).alias('bin_idx')
        )
        .explode('bin_idx')
        .select(['player_id', 'bin_idx'])
    )

    # Pivot to wide format
    mean_wide = agg_df.pivot(
        index=['player_id', 'bin_idx'],
        on='skill_name',
        values='mean_value'
    )
    count_wide = agg_df.pivot(
        index=['player_id', 'bin_idx'],
        on='skill_name',
        values='n_obs'
    )

    # Join with full index to fill gaps
    mean_wide = full_index.join(mean_wide, on=['player_id', 'bin_idx'], how='left')
    count_wide = full_index.join(count_wide, on=['player_id', 'bin_idx'], how='left')

    # Ensure all skill columns exist
    for skill_name in skill_names:
        if skill_name not in mean_wide.columns:
            mean_wide = mean_wide.with_columns(pl.lit(None).cast(pl.Float64).alias(skill_name))
        if skill_name not in count_wide.columns:
            count_wide = count_wide.with_columns(pl.lit(None).cast(pl.Int64).alias(skill_name))

    # Standardize types and fill nulls
    mean_wide = (
        mean_wide
        .with_columns([pl.col(s).cast(pl.Float64) for s in skill_names])
        .with_columns([pl.col(s).fill_null(float('nan')).alias(s) for s in skill_names])
        .sort(['player_id', 'bin_idx'])
    )
    count_wide = (
        count_wide
        .with_columns([pl.col(s).cast(pl.Int64) for s in skill_names])
        .with_columns([pl.col(s).fill_null(0).alias(s) for s in skill_names])
        .sort(['player_id', 'bin_idx'])
    )

    # Build player_start_bins lookup
    player_start_bins = dict(
        player_bin_stats
        .select(['player_id', 'start_bin'])
        .iter_rows()
    )

    # Join mean and count into single frame with suffixed count columns
    count_cols_renamed = ['player_id', 'bin_idx'] + [f'{s}_count' for s in skill_names]
    count_wide_renamed = count_wide.select(
        ['player_id', 'bin_idx'] + [
            pl.col(s).alias(f'{s}_count') for s in skill_names
        ]
    )
    
    combined = mean_wide.join(
        count_wide_renamed,
        on=['player_id', 'bin_idx'],
        how='inner'
    )

    # Build observation sequences using group_by iteration
    binned_data = {}
    
    for group_key, group_df in combined.group_by('player_id', maintain_order=True):
        # group_key is a tuple, extract the player_id
        pid = group_key[0] if isinstance(group_key, tuple) else group_key
        
        group_df = group_df.sort('bin_idx')
        
        mean_arr = group_df.select(skill_names).to_numpy()
        count_arr = group_df.select([f'{s}_count' for s in skill_names]).to_numpy()
        mask_mat = ~np.isnan(mean_arr)
        
        obs_sequence = []
        for i in range(mean_arr.shape[0]):
            mask = mask_mat[i]
            if mask.any():
                y = mean_arr[i, mask]
                n_obs = count_arr[i, mask]
            else:
                y = np.array([])
                n_obs = np.array([])
            
            obs_sequence.append({
                'mask': mask.copy(),
                'y': y,
                'n_obs': n_obs
            })
        
        binned_data[pid] = obs_sequence
    
    return binned_data, time_bins, origin_date, player_start_bins

def compute_population_prior(df, skill_names, subject_col='player_id', min_obs_per_subject=3):
    """
    Compute population mean and standard deviation for each skill, using
    between-subject variation (weighted variance of subject means).

    Parameters
    ----------
    df : polars.DataFrame
        Must include columns: subject_col, 'skill_name', 'observed_value'
    skill_names : list[str]
        Ordered list of skills that defines the output vector order
    subject_col : str
        Column identifying the subject (player) id
    min_obs_per_subject : int
        Minimum observations required for a subject to contribute to that skill's prior

    Returns
    -------
    mu_pop : np.ndarray
        (n_skills,) population means (across-subject mean of subject means)
    sigma_pop : np.ndarray
        (n_skills,) population std dev (across-subject std of subject means)
    """
    skill_to_idx = {name: i for i, name in enumerate(skill_names)}
    n_skills = len(skill_names)

    mu_pop = np.zeros(n_skills)
    sigma_pop = np.ones(n_skills)

    # Per-(skill, subject) mean and count
    subj_stats = (
        df.group_by(['skill_name', subject_col])
        .agg(
            pl.col('observed_value').mean().alias('subj_mean'),
            pl.col('observed_value').len().alias('n_obs')
        )
        .filter(pl.col('n_obs') >= min_obs_per_subject)
    )

    # Weighted moments across subjects, per skill
    # Weighted mean: sum(w*m)/sum(w)
    # Weighted variance: sum(w*(m-mu)^2)/sum(w)
    skill_stats = (
        subj_stats
        .with_columns(
            (pl.col('n_obs') * pl.col('subj_mean')).alias('w_times_m'),
            (pl.col('n_obs') * (pl.col('subj_mean') ** 2)).alias('w_times_m2'),
        )
        .group_by('skill_name')
        .agg(
            pl.col('n_obs').sum().alias('w_sum'),
            pl.col('w_times_m').sum().alias('w_m_sum'),
            pl.col('w_times_m2').sum().alias('w_m2_sum'),
        )
        .with_columns(
            (pl.col('w_m_sum') / pl.col('w_sum')).alias('mu_w'),
            ((pl.col('w_m2_sum') / pl.col('w_sum')) - (pl.col('w_m_sum') / pl.col('w_sum')) ** 2).alias('var_w'),
        )
    )

    for row in skill_stats.iter_rows(named=True):
        skill_name = row['skill_name']
        if skill_name in skill_to_idx:
            idx = skill_to_idx[skill_name]
            mu = row['mu_w']
            var = row['var_w']

            mu_pop[idx] = float(mu) if mu is not None else 0.0
            if var is not None and var > 0:
                sigma_pop[idx] = float(np.sqrt(var))
            else:
                sigma_pop[idx] = 1.0

    return mu_pop, sigma_pop


def compute_observation_counts(df, skill_names):
    """
    Compute total observation counts per skill.
    """
    skill_to_idx = {name: i for i, name in enumerate(skill_names)}
    n_skills = len(skill_names)
    counts = np.zeros(n_skills)
    
    count_df = df.group_by('skill_name').agg(
        pl.col('observed_value').len().alias('count')
    )
    
    for row in count_df.iter_rows(named=True):
        skill_name = row['skill_name']
        if skill_name in skill_to_idx:
            counts[skill_to_idx[skill_name]] = row['count']
    
    return counts
