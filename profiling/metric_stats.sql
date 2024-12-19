-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.
--
-- Authored by Liyan Chen (liyanc@cs.utexas.edu) on 4/28/25
--

-- metric_stats.sql
-- Parameters:
-- :scope_name - The name of the NVTX scope (e.g., 'Backbone')
-- :metric_name - The name of the GPU metric (e.g., 'SMs Active [Throughput %]')
-- :k - The number of initial scopes to skip

-- Common Table Expression (CTE) to find relevant scopes and skip the first k
WITH RelevantScopes AS (
    SELECT
        start,
end
FROM NVTX_EVENTS
    WHERE text = :scope_name
    ORDER BY start -- Order scopes chronologically to ensure consistent skipping
    LIMIT -1 OFFSET :k -- Skip the first :k scopes, take all remaining (-1 means no upper limit)
),
-- CTE to calculate the average metric value within each relevant scope
ScopeAvgMetrics AS (
    SELECT
        AVG(gm.value) AS avg_value_in_scope
    FROM RelevantScopes rs
    -- Join GPU_METRICS based on the timestamp falling within a scope's start and end
    JOIN GPU_METRICS gm ON gm.timestamp BETWEEN rs.start AND rs.end
    -- Join TARGET_INFO to filter by the correct metric name using metricId
    JOIN TARGET_INFO_GPU_METRICS tigm ON gm.metricId = tigm.metricId
                                     AND tigm.metricName = :metric_name
    GROUP BY rs.start, rs.end -- Group by scope to get the average *per scope*
)
-- Final SELECT to calculate the overall average and standard deviation
-- of the per-scope averages calculated in the previous CTE.
SELECT
    AVG(avg_value_in_scope) AS overall_avg,
    -- Calculate Standard Deviation using SQRT(AVG(X^2) - AVG(X)^2)
    -- This calculates the population standard deviation of the scope averages.
    SQRT(AVG(avg_value_in_scope * avg_value_in_scope) - AVG(avg_value_in_scope) * AVG(avg_value_in_scope)) AS overall_stddev
FROM ScopeAvgMetrics;