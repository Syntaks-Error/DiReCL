"""
### TASK SUMMARY

The ego vehicle drives in naturalistic highway scenarios. It should reach the
CommonRoad planning goal region while avoiding collisions, remaining on the
road, respecting lane geometry, keeping safe distance to surrounding vehicles,
and driving smoothly. The original CommonRoad hand-designed reward is hidden;
the generated reward should be a surrogate reward computed only from the inputs
of `compute_reward`.

High-level priorities, in decreasing order:

1. Safety:
   - Avoid collision.
   - Avoid leaving the drivable road.
   - Avoid friction-limit violation and unstable motions.
   - Maintain safe longitudinal/lateral margins to surrounding vehicles.

2. Goal achievement and route progress:
   - Move toward the route-defined goal region.
   - Reach the goal before timeout.
   - Do not learn a reward that prefers stopping far away from the goal.

3. Lane discipline and route following:
   - Stay close to the lane/reference-path center when no lane change is needed.
   - Avoid wrong-way driving and large heading error.
   - Avoid unnecessary lane changes, but allow lane changes required by the
     route or by safety.

4. Comfort and physical plausibility:
   - Avoid large acceleration, jerk, yaw rate, slip/curvature, and action
     magnitude.
   - Prefer smooth highway-like behavior.

5. Efficiency:
   - Maintain a reasonable highway speed when safe.
   - Avoid reverse driving, excessive braking, or standing still unless safety
     requires it.

### INPUT MEANINGS

1. `obs`
With the default CommonRoad/highD configuration, observations are flattened into
one vector of length 53. For batched reward evaluation, `obs` has shape `[B, 53]`.
For single-step reward evaluation, `obs` may have shape `[53]`.

The flattened order is documented below. If `environment_configurations.yml` is
changed, the observation length/order may change. In that case, rebuild the
slices from CommonRoad's `observation_space_dict` instead of assuming length 53.

2. `action`
Action selected by the policy at `obs`.

Default action space:

```text
Box(low=-1, high=1, shape=(2,), dtype=float32)
```

The two action dimensions are normalized controls. CommonRoad internally rescales
them to physical vehicle-model inputs. With the default continuous action config
and `action_base='acceleration'`, the dimensions should be interpreted as:

- `action[..., 0]`: normalized longitudinal control, mainly acceleration/braking.
- `action[..., 1]`: normalized lateral/steering/curvature-related control,
  depending on the selected vehicle model.

For the default QP vehicle model, the exact physical rescaling is handled inside
CommonRoad. Therefore, in reward code, treat the action as normalized control and
use it mainly for comfort/regularization, e.g., penalize large action magnitude
or abrupt control if previous action is available in `info`.

3. `next_obs`
Observation after applying `action` and stepping the environment.

Use `next_obs` for most reward terms because it reflects the consequence of the
action. Important examples:

- `next_obs[5]`: whether the goal is reached after the action.
- `next_obs[6]`: whether the episode times out after the action.
- `next_obs[13]`: whether friction is violated after the action.
- `next_obs[27]`: whether collision occurs after the action.
- `next_obs[29]`: whether the ego vehicle is off-road after the action.
- `next_obs[1]`: longitudinal progress toward the goal.
- `next_obs[3]`: lateral progress toward the goal.

4. `info`
Auxiliary dictionary returned by `env.step`. Keys may vary across wrappers and
versions, so always access defensively with `info.get(key, default)`.

Common keys may include:

- `scenario_name`: CommonRoad benchmark/scenario id.
- `chosen_action`: clipped/rescaled action used by the environment.
- `current_episode_time_step`: current time step.
- `max_episode_time_steps`: episode horizon.
- `termination_reason`: reason for termination if terminated.
- `v_ego_mean`: running mean ego velocity.
- `is_collision`, `is_off_road`, `is_time_out`, `is_goal_reached`, etc.
- `ttc_follow`, `ttc_lead` when lane-based surrounding observations are active.

Do not assume that every key exists. Do not use any hidden/oracle reward key.

### DEFAULT FLATTENED OBSERVATION VECTOR

A. Goal-related observations

[0:1] distance_goal_long, shape=(1,), unit=m
    Meaning:
        Route-relative longitudinal distance from the ego vehicle to the goal
        region along the planned reference route. Smaller absolute value means
        closer to the goal. The sign is route/local-coordinate dependent, so
        avoid relying only on the sign.
    Reward use:
        Penalize large absolute distance or use progress features below.

[1:2] distance_goal_long_advance, shape=(1,), unit=m/step
    Meaning:
        Previous absolute longitudinal goal distance minus current absolute
        longitudinal goal distance. Positive means the ego vehicle got closer to
        the goal along the route.
    Reward use:
        This is usually the cleanest dense progress signal. Reward positive
        values and mildly penalize negative values.

[2:3] distance_goal_lat, shape=(1,), unit=m
    Meaning:
        Lateral distance from the ego vehicle to the goal region in the
        route/local Frenet frame. Near zero means laterally aligned with the goal
        lane/region.
    Reward use:
        Penalize absolute lateral goal error, especially near the goal.

[3:4] distance_goal_lat_advance, shape=(1,), unit=m/step
    Meaning:
        Previous absolute lateral goal distance minus current absolute lateral
        goal distance. Positive means the ego vehicle got laterally closer to
        the goal alignment.
    Reward use:
        Reward positive lateral alignment progress, but keep safety dominant.

[4:5] distance_goal_time, shape=(1,), unit=time steps
    Meaning:
        Distance from current time step to the goal time interval. Usually
        negative before the earliest goal time, zero inside the admissible time
        interval, and positive after the latest admissible goal time.
    Reward use:
        Penalize timeout or being too late. Do not strongly penalize being early
        while still far from the goal.

[5:6] is_goal_reached, shape=(1,), unit=boolean {0,1}
    Meaning:
        Whether the CommonRoad planning-problem goal region is reached.
    Reward use:
        Large positive terminal bonus. Prefer `next_obs[:, 5]`.

[6:7] is_time_out, shape=(1,), unit=boolean {0,1}
    Meaning:
        Whether the episode exceeded the allowed time before reaching the goal.
    Reward use:
        Terminal penalty, usually weaker than collision/off-road but strong
        enough to prevent waiting forever. Prefer `next_obs[:, 6]`.

B. Ego-vehicle observations

[7:8] v_ego, shape=(1,), unit=m/s
    Meaning:
        Ego longitudinal speed under the default QP vehicle model.
    Reward use:
        Encourage reasonable highway speed when safe. Do not reward speed so much
        that the agent ignores safety or route progress.

[8:9] a_ego, shape=(1,), unit=m/s^2
    Meaning:
        Ego longitudinal acceleration under the default QP vehicle model.
    Reward use:
        Penalize very large acceleration/deceleration for comfort and physical
        plausibility.

[9:10] jerk_ego, shape=(1,), unit=m/s^3
    Meaning:
        Ego jerk in the QP vehicle model, measuring abrupt acceleration changes.
    Reward use:
        Penalize squared or absolute jerk to encourage smooth driving.

[10:11] relative_heading, shape=(1,), unit=rad
    Meaning:
        Heading angle of ego vehicle relative to current lane/reference direction.
        Zero means aligned with the lane/reference path.
    Reward use:
        Penalize large absolute heading error to discourage unstable/wrong-way
        behavior.

[11:12] global_turn_rate, shape=(1,), unit=rad/s
    Meaning:
        Ego yaw rate / global turn rate.
    Reward use:
        Penalize excessive yaw rate. Small values are expected in highway tasks.

[12:13] slip_angle, shape=(1,), unit=model-dependent, approximately rad or curvature-like
    Meaning:
        QP vehicle-model slip/curvature-related state. Large magnitude indicates
        unstable or sharp motion.
    Reward use:
        Penalize large magnitude as a comfort/stability term.

[13:14] is_friction_violation, shape=(1,), unit=boolean {0,1}
    Meaning:
        Whether the vehicle violates friction/physical tire-force constraints.
    Reward use:
        Strong safety/stability penalty. Prefer `next_obs[:, 13]`.

[14:15] remaining_steps, shape=(1,), unit=time steps
    Meaning:
        Number of remaining time steps until the goal-region horizon / episode
        end.
    Reward use:
        Can increase urgency near the end. Avoid directly rewarding large
        remaining time.

C. Surrounding-vehicle observations

[15:21] lane_based_v_rel, shape=(6,), unit=m/s
    Meaning:
        Relative velocities of six lane-based surrounding slots:

        index 0: left_follow  - nearest following vehicle in the left lane
        index 1: same_follow  - nearest following vehicle in the ego lane
        index 2: right_follow - nearest following vehicle in the right lane
        index 3: left_lead    - nearest leading vehicle in the left lane
        index 4: same_lead    - nearest leading vehicle in the ego lane
        index 5: right_lead   - nearest leading vehicle in the right lane

        Values are relative to ego according to CommonRoad's lane-based
        observation implementation. Absent vehicles use configured dummy values.
    Reward use:
        Combine with lane_based_p_rel for safe-distance or TTC-like penalties.
        The most important highway slots are same_lead and same_follow.

[21:27] lane_based_p_rel, shape=(6,), unit=m
    Meaning:
        Relative lane-based distances to the same six surrounding slots:

        index 0: left_follow
        index 1: same_follow
        index 2: right_follow
        index 3: left_lead
        index 4: same_lead
        index 5: right_lead

        The default dummy distance is large, about 500 m, indicating no relevant
        vehicle detected in that slot.
    Reward use:
        Penalize too-small same-lane leading/following distance. Ignore dummy
        large distances. A useful smooth safety rule is based on time headway:
        `safe_distance = base_distance + time_headway * v_ego`.

[27:28] is_collision, shape=(1,), unit=boolean {0,1}
    Meaning:
        Whether the ego vehicle collides with another obstacle.
    Reward use:
        Largest terminal safety penalty. Prefer `next_obs[:, 27]` or
        `info.get('is_collision', ...)` if available.

[28:29] lane_change, shape=(1,), unit=boolean {0,1}
    Meaning:
        Whether ego currently occupies a lanelet outside the previously
        identified ego-lane set. Roughly indicates an ongoing lane change.
    Reward use:
        Penalize unnecessary lane changes mildly, but do not forbid lane changes
        needed for safety or route progress.

D. Lanelet / road-network observations

[29:30] is_off_road, shape=(1,), unit=boolean {0,1}
    Meaning:
        Whether the ego vehicle is outside the drivable road boundary.
    Reward use:
        Large terminal safety penalty comparable to collision. Prefer
        `next_obs[:, 29]`.

[30:31] left_marker_distance, shape=(1,), unit=m
    Meaning:
        Distance from ego vehicle to the left lane marker of the current lanelet.
    Reward use:
        Use with right_marker_distance and lat_offset for lane-centering and
        boundary-margin rewards.

[31:32] right_marker_distance, shape=(1,), unit=m
    Meaning:
        Distance from ego vehicle to the right lane marker of the current lanelet.
    Reward use:
        Use with left_marker_distance and lat_offset for lane-centering and
        boundary-margin rewards.

[32:33] left_road_edge_distance, shape=(1,), unit=m
    Meaning:
        Distance from ego vehicle to the left road edge/boundary.
    Reward use:
        Penalize very small edge distance before off-road occurs.

[33:34] right_road_edge_distance, shape=(1,), unit=m
    Meaning:
        Distance from ego vehicle to the right road edge/boundary.
    Reward use:
        Penalize very small edge distance before off-road occurs.

[34:35] lat_offset, shape=(1,), unit=m
    Meaning:
        Lateral offset from the local lane/reference centerline. Positive means
        left of the centerline; zero means centered.
    Reward use:
        Penalize absolute offset for lane keeping, with tolerance so lane changes
        remain possible.

[35:45] route_reference_path_positions, shape=(10,), unit=m
    Meaning:
        Flattened 2-D positions of reference-route waypoints sampled at route
        distances `[-1000, 0, 5, 15, 100]` meters around/ahead of the ego vehicle.
        The order is:

        [x_-1000, y_-1000, x_0, y_0, x_5, y_5, x_15, y_15, x_100, y_100]

        Coordinates are produced by CommonRoad's navigator using its automatic
        ego/local coordinate convention.
    Reward use:
        Useful for route-following and future curvature anticipation. Avoid
        overfitting to absolute map coordinates; prefer relative alignment,
        heading, and offset features when possible.

[45:50] route_reference_path_orientations, shape=(5,), unit=rad
    Meaning:
        Reference-route headings at the same sampled distances
        `[-1000, 0, 5, 15, 100]` meters. The order corresponds to
        route_reference_path_positions.
    Reward use:
        Align ego heading with future route and discourage wrong-way driving.

[50:53] distance_togoal_via_referencepath, shape=(3,), unit=[m, m, boolean-like]
    Meaning:
        Three values:

        index 0: longitudinal distance to goal on reference path
        index 1: lateral distance to reference path
        index 2: whether ego is inside the reference-path projection domain

    Reward use:
        Reward decreasing longitudinal distance, penalize lateral distance, and
        penalize leaving the projection domain.

E. Optional features if CommonRoad config changes

The default highD/CommonRoad config above produces 53 dimensions. If observation
settings are changed, additional features may appear. Their meanings are:

- euclidean_distance: Euclidean distance to goal center.
- distance_goal_long_lane: distance by which a required lane change must finish.
- distance_goal_orientation: deviation from goal orientation interval in radians.
- distance_goal_velocity: deviation from goal velocity interval in m/s.
- steering_angle: steering angle if exposed by the vehicle model.
- lane_curvature: curvature of the current local lane/reference path.
- extrapolation_static_off: future lateral offsets at fixed distance samples.
- extrapolation_dynamic_off: future lateral offsets under current velocity.
- lidar_circle_dist: lidar-like radial obstacle distances.
- lidar_circle_dist_rate: distance-rate values for lidar beams.
- dist_lead_follow_rel: [leading, following] lane-based distances.
- rel_prio_lidar: relative traffic priority, roughly yield/same/priority = -1/0/1.
- vehicle_type: detected obstacle classes.
- vehicle_signals: detected obstacle turn/brake/indicator signals.
- intersection_velocities: relative velocities of conflict-zone vehicles.
- intersection_distances: distances of conflict-zone vehicles.
- ego_distance_intersection: ego distance to near/far conflict-zone points.
- stop_sign, yield_sign, priority_sign, right_of_way_sign: traffic-sign features.

### REWARD DESIGN RECOMMENDATIONS

A good generated reward for highD/CommonRoad should usually contain these parts:

1. Terminal safety and success terms from `next_obs`:

   - Large negative penalty for collision: `next_obs[:, 27]`.
   - Large negative penalty for off-road: `next_obs[:, 29]`.
   - Large positive bonus for goal reached: `next_obs[:, 5]`.
   - Moderate negative penalty for timeout: `next_obs[:, 6]`.
   - Strong penalty for friction violation: `next_obs[:, 13]`.

2. Dense goal progress:

   - Reward `next_obs[:, 1]` = distance_goal_long_advance.
   - Reward `next_obs[:, 3]` = distance_goal_lat_advance.
   - Penalize remaining distance magnitude, e.g. `abs(next_obs[:, 0])`,
     `abs(next_obs[:, 2])`, and/or `abs(next_obs[:, 50])`.

3. Safe-distance terms:

   - Extract `v_ego = next_obs[:, 7]`.
   - Extract same-lane following distance: `same_follow_dist = next_obs[:, 22]`.
   - Extract same-lane leading distance: `same_lead_dist = next_obs[:, 25]`.
   - Ignore dummy distances near 500 m.
   - Penalize distances smaller than `base_distance + time_headway * v_ego` using
     smooth functions such as `softplus` or squared `clamp` violations.

4. Lane and route discipline:

   - Penalize `abs(next_obs[:, 34])` for lateral offset.
   - Penalize `abs(next_obs[:, 10])` for relative heading error.
   - Penalize `1 - next_obs[:, 52]` if outside the route projection domain.
   - Penalize lane_change `next_obs[:, 28]` mildly, not as a hard constraint.

5. Comfort and action regularization:

   - Penalize squared acceleration: `next_obs[:, 8] ** 2`.
   - Penalize squared jerk: `next_obs[:, 9] ** 2`.
   - Penalize squared yaw rate: `next_obs[:, 11] ** 2`.
   - Penalize squared slip angle: `next_obs[:, 12] ** 2`.
   - Penalize action magnitude: `(action ** 2).sum(dim=-1)`.

6. Robustness:

   - Clamp large values before squaring if necessary.
   - Avoid hard equality checks such as `x == 0` for continuous values.
   - Prefer smooth barriers over discontinuous step penalties, except for boolean
     terminal indicators that are already provided by the environment.
   - Make sure the reward is finite for all observations.
"""
