import numpy as np
import polars as pl
from joblib import Parallel, delayed


def _estimate_child_weights(child_idx, parent_indices, binned_data, skill_estimates):
    X_list = []
    y_list = []
    weight_list = []

    for player_id, obs_seq in binned_data.items():
        s_player = skill_estimates.get(player_id)
        if s_player is None:
            continue

        T = len(obs_seq)
        for t in range(T):
            obs = obs_seq[t]
            mask = obs['mask']

            if not mask[child_idx]:
                continue

            obs_indices = np.where(mask)[0]
            child_pos_arr = np.where(obs_indices == child_idx)[0]
            if len(child_pos_arr) == 0:
                continue
            child_pos = child_pos_arr[0]

            y_child = obs['y'][child_pos]
            n_obs = obs['n_obs'][child_pos]

            x = np.empty(1 + len(parent_indices))
            x[0] = 1.0
            for k, p_idx in enumerate(parent_indices):
                x[1 + k] = s_player[t, p_idx]

            X_list.append(x)
            y_list.append(y_child)
            weight_list.append(n_obs)

    n_samples = len(X_list)
    if n_samples < len(parent_indices) + 1:
        return child_idx, None

    X = np.vstack(X_list)
    y = np.array(y_list)
    weights = np.array(weight_list)

    sqrt_w = np.sqrt(weights)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w

    XtX = Xw.T @ Xw
    Xty = Xw.T @ yw

    ridge = 1e-6 * np.eye(XtX.shape[0])
    try:
        beta = np.linalg.solve(XtX + ridge, Xty)
    except np.linalg.LinAlgError:
        return child_idx, None

    return child_idx, beta[1:]


class ParameterEstimator:
    """
    EM-based parameter estimation for the hierarchical Kalman filter.
    """
    
    def __init__(self, dag, n_skills):
        self.dag = dag
        self.n_skills = n_skills
        self.edge_mask = dag.get_edge_mask()
    
    def estimate_edge_weights(self, binned_data, z_smooth, B_current, n_jobs=1):
        W = np.zeros((self.n_skills, self.n_skills))
        
        skill_estimates = {}
        for player_id, z_player in z_smooth.items():
            skill_estimates[player_id] = (B_current @ z_player.T).T

        tasks = []
        for child_idx in range(self.n_skills):
            parent_indices = self.dag.get_parent_indices(child_idx)
            if len(parent_indices) == 0:
                continue
            tasks.append((child_idx, parent_indices))

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_estimate_child_weights)(
                child_idx,
                parent_indices,
                binned_data,
                skill_estimates
            )
            for child_idx, parent_indices in tasks
        )

        for child_idx, beta_parents in results:
            if beta_parents is None:
                continue
            parent_indices = self.dag.get_parent_indices(child_idx)
            for k, p_idx in enumerate(parent_indices):
                W[child_idx, p_idx] = beta_parents[k]
        
        return W
    
    def estimate_process_noise(self, z_smooth, P_smooth, P_cross):
        """
        Estimate process noise variances Q from smoothed estimates.
        
        E[(z_t - z_{t-1})^2] = E[z_t^2] - 2*E[z_t*z_{t-1}] + E[z_{t-1}^2]
        """
        Q_sum = np.zeros(self.n_skills)
        Q_count = np.zeros(self.n_skills)
        
        for player_id, z_player in z_smooth.items():
            P_player = P_smooth[player_id]
            Pc_player = P_cross[player_id]
            T = z_player.shape[0]

            if T < 2:
                continue

            diag_P = np.diagonal(P_player, axis1=1, axis2=2)
            diag_Pc = np.diagonal(Pc_player, axis1=1, axis2=2)

            Ez_t_sq = diag_P[1:] + z_player[1:] ** 2
            Ez_tm1_sq = diag_P[:-1] + z_player[:-1] ** 2
            Ez_t_tm1 = diag_Pc + z_player[1:] * z_player[:-1]

            diff_sq = Ez_t_sq - 2 * Ez_t_tm1 + Ez_tm1_sq

            count = diff_sq.shape[0]

            Q_sum += diff_sq.sum(axis=0)
            Q_count += count
        
        Q = np.where(Q_count > 0, Q_sum / Q_count, 0.01)
        Q = np.maximum(Q, 1e-8)
        
        return Q
    
    def estimate_observation_noise(
        self,
        z_smooth,
        B,
        raw_df,
        skill_names,
        bin_size_days,
        origin_date,
        player_start_bins
    ):
        n_skills = len(skill_names)

        player_ids = list(z_smooth.keys())
        if len(player_ids) == 0:
            return np.ones(n_skills)

        lengths = np.array([z_smooth[pid].shape[0] for pid in player_ids], dtype=np.int32)
        start_bins = np.array([player_start_bins.get(pid, 0) for pid in player_ids], dtype=np.int32)
        end_bins = start_bins + lengths - 1
        max_end_bin = int(end_bins.max()) if end_bins.size > 0 else -1

        raw_df = (
            raw_df
            .with_columns(pl.col('timestamp').cast(pl.Datetime).alias('timestamp'))
            .with_columns(
                ((pl.col('timestamp') - pl.lit(origin_date)).dt.total_days() // bin_size_days)
                .cast(pl.Int32)
                .alias('bin_idx')
            )
            .filter((pl.col('bin_idx') >= 0) & (pl.col('bin_idx') <= max_end_bin))
        )

        needed = raw_df.select(['player_id', 'bin_idx', 'skill_name']).unique()

        pid_map = pl.DataFrame({
            'player_id': player_ids,
            'p_row': np.arange(len(player_ids), dtype=np.int32),
            'start_bin': start_bins,
            'end_bin': end_bins
        })
        skill_map = pl.DataFrame({
            'skill_name': skill_names,
            'skill_idx': np.arange(n_skills, dtype=np.int32)
        })

        needed_idx = (
            needed
            .join(pid_map, on='player_id', how='inner')
            .join(skill_map, on='skill_name', how='inner')
            .with_columns((pl.col('bin_idx') - pl.col('start_bin')).alias('local_idx'))
            .filter((pl.col('local_idx') >= 0) & (pl.col('bin_idx') <= pl.col('end_bin')))
        )

        if needed_idx.height == 0:
            return np.ones(n_skills)

        offsets = np.cumsum(np.concatenate(([0], lengths[:-1]))).astype(np.int64)
        Z_flat = np.concatenate([z_smooth[pid] for pid in player_ids], axis=0)

        p_rows = needed_idx.get_column('p_row').to_numpy()
        b_idxs = needed_idx.get_column('local_idx').to_numpy()
        s_idxs = needed_idx.get_column('skill_idx').to_numpy()

        row_idx = offsets[p_rows] + b_idxs
        z_rows = Z_flat[row_idx, :]            # (n_needed, n_skills)
        s_rows = z_rows @ B.T                  # (n_needed, n_skills)
        s_hat = s_rows[np.arange(len(s_idxs)), s_idxs]

        estimates_df = needed_idx.select(['player_id', 'bin_idx', 'skill_name']).with_columns(
            pl.Series('s_hat', s_hat)
        )

        joined = raw_df.join(estimates_df, on=['player_id', 'bin_idx', 'skill_name'], how='inner')

        residuals = (
            joined
            .with_columns(((pl.col('observed_value') - pl.col('s_hat')) ** 2).alias('resid_sq'))
            .group_by('skill_name')
            .agg(
                pl.col('resid_sq').sum().alias('sum_resid_sq'),
                pl.col('resid_sq').len().alias('count')
            )
            .join(skill_map, on='skill_name', how='inner')
            .with_columns((pl.col('sum_resid_sq') / pl.col('count')).alias('r_hat'))
        )

        R = np.ones(n_skills)
        idxs = residuals.get_column('skill_idx').to_numpy()
        vals = residuals.get_column('r_hat').to_numpy()
        R[idxs] = vals

        return np.maximum(R, 1e-8)
    
    def compute_log_likelihood(self, innovations, S_list):
        """
        Compute log-likelihood from filter innovations.
        
        LL = sum_t [ -0.5 * log|S_t| - 0.5 * v_t' S_t^{-1} v_t - 0.5 * n_t * log(2*pi) ]
        """
        ll = 0.0
        n_obs_total = 0
        
        for player_id in innovations:
            player_innov = innovations[player_id]
            player_S = S_list[player_id]
            
            for v, S in zip(player_innov, player_S):
                if v is None or S is None:
                    continue
                
                n_t = len(v)
                
                try:
                    sign, logdet = np.linalg.slogdet(S)
                    if sign <= 0:
                        continue
                    
                    mahal = v @ np.linalg.solve(S, v)
                    
                    ll -= 0.5 * logdet
                    ll -= 0.5 * mahal
                    ll -= 0.5 * n_t * np.log(2 * np.pi)
                    n_obs_total += n_t
                    
                except np.linalg.LinAlgError:
                    continue
        
        return ll
