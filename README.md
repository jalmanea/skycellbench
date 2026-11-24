# SkyCellBench: a benchmark for multi-UAV aerial base station placement

Version 0.1, not frozen. The licence is not decided yet.

## The tasks

An instance gives a square area, a set of ground users with a downlink demand each, and K UAV base
stations. The controller returns one position `(x, y, altitude)` per UAV. That is the whole
interface, so an LLM agent, an RL policy and a plain algorithm all answer the same question.
There are three tasks. Each adds one difficulty to the first, and all use the same scorers.

| Task | What is added | What makes it harder |
|---|---|---|
| `place` | nothing | the baseline task |
| `constrained` | no-fly zones over the largest hotspots; every UAV must reach a ground gateway on the area edge through hops of 450 to 650 m, directly or through other UAVs | the obvious spots are forbidden and the chain cannot cover the whole area, so coverage is traded against connectivity |
| `blind` | user positions are hidden; the controller gets 30 probes | no geometry to reason from; each probe tries one placement and returns, per UAV, the users attached, their demand and the throughput delivered, plus the score |

Common to all tasks:

- Area: `[0, side] x [0, side]`, 2 km by default. Altitude between 50 and 300 m.
- All UAVs share one 20 MHz LTE carrier, so they interfere with each other.
- Each user attaches to the UAV with the lowest path loss.
- Channel: free-space loss plus an excess loss, 1 dB with line of sight and 20 dB without
  (Al-Hourani et al., 2014, urban values). An instance either stands on flat, empty ground, and
  then the excess is its mean over a line-of-sight probability that depends only on the
  elevation angle: a smooth function of position. Or it stands on a scene (next section), and
  then line of sight is computed for each link from the terrain and the buildings. Neither has
  random shadowing.
- Demand: 0.5, 1 or 2 Mbit/s per user, constant bit rate.
- A placement that breaks a rule is rejected with the rule named; it has no score.

## Scenes

A scene is a real place as geometry: building footprints with one height each, a grid of ground
heights, and the origin and coordinate system that tie them to the map (`skycellbench/scene.py`
documents the format, `skycellbench-scene/1`). An instance names its scene; instances without
one are unchanged.

There is one scene, `scenes/cincinnati-downtown`: 2 km by 2 km of downtown Cincinnati and
Over-the-Rhine, with 2,332 buildings up to 203 m tall and ground from 139 m (the Ohio River) to
257 m. Buildings come from OpenStreetMap (ODbL) and the ground from the USGS 3D Elevation
Program (public domain); `scenes/cincinnati-downtown/SOURCES.md` lists every source, its
licence and how heights were chosen. The area contains that of
[DroneRTS](https://github.com/DanMcInerney/DroneRTS), whose city is 266 OpenStreetMap footprints
on flat ground; the scene takes three tower heights from it and no geometry.

On a scene:

- Users stand outdoors: outside every building footprint and off the water, 1.5 m above the
  ground.
- Altitude is measured above the ground under the UAV, still between 50 and 300 m, and a UAV may
  not be at or below the roof of a building it is over.
- A link has line of sight when the straight line from the user to the UAV clears the surface
  (ground plus buildings, on a raster of 2 m cells) at samples 2 m apart. The path loss is the
  free-space loss plus 1 dB with line of sight and 20 dB without.
- Backhaul hops and no-fly zones are as before; hop lengths use true heights.

## The score

    score = mean over users of min(1, delivered throughput / demand)

It is the average share of demand delivered, between 0 and 1. Every user counts the same, so
leaving some users unserved costs as much as serving many users poorly. Total throughput and the
5th-percentile user are reported next to it but do not enter the score.

## Two scorers, one answer format

| | Closed-form (`score_fast`) | ns-3 LTE (`score_ns3`) |
|---|---|---|
| Time per placement | about 1 ms | about 20 s |
| Use | development, training, search | the reported score |
| How | formulas (below) | packet-level simulation, ns-3.48 LENA with EPC and UDP flows |

The path loss of every UAV-user link is computed once per placement, in Python
(`fast.path_loss_db`), and both scorers use that table. ns-3 computes no loss of its own: the
table is written into its scenario file and loaded into ns-3's `MatrixPropagationLossModel`,
which LTE accepts like any other path loss model (`ns3/uav-score.cc`). With the table of the
flat-ground channel, ns-3 returns the scores it returned when it computed the loss itself (12
stored answers rescored, all equal to four decimals).

The closed-form scorer follows the same chain as ns-3: path loss, association, SINR with
interference from each neighbour scaled by how busy that neighbour is, the rate a user would get
alone in its cell, then proportional-fair sharing of the cell capped at each user's demand. Cell
loads and SINRs depend on each other and are solved by fixed-point iteration. The rate step uses
a link curve measured from ns-3 itself (`experiments/link_curve.py`, one user, 125 SINR values),
because ns-3 chooses the modulation and coding from error curves that have no formula.

ns-3 settings that differ from the defaults: radio link failure detection is off (a user with a
weak link receives nothing, and is not disconnected part-way through the run), SRS periodicity is
320 so that a cell can hold more than 40 users, and users are attached to their cell directly.
Each run has 0.5 s to attach, 0.3 s of traffic to settle and 2 s that are measured.

## Files

- `skycellbench/instance.py`: instance and placement formats (JSON), the rules of each task, seeded
  generator.
- `skycellbench/scene.py`: scene format, the surface raster and the line-of-sight test.
  `scenes/cincinnati-downtown/`: the scene, the script that builds it and its sources.
- `skycellbench/fast.py`: the path loss table, the closed-form scorer and the probe for the blind
  task. `skycellbench/link_curve.json`: the measured link curve.
- `skycellbench/ns3.py` and `ns3/uav-score.cc`: ns-3 scorer and its Python wrapper.
- `skycellbench/baselines.py`: centre (do nothing), random, k-means, local search.
- `skycellbench/rl.py`: step-by-step environment (reset/step) and the PPO policy.
- `skycellbench/controllers.py`: the controllers that are compared, behind one call.
- `instances/public`, `instances/heldout`: 10 instances per task each, on flat ground. The
  held-out set is for the reported scores and must not be scored before the protocol is frozen.
  `instances/public-cincinnati-downtown`: the public seeds on the scene. There is no held-out
  set on the scene yet.
- `experiments/`: link-curve measurement, scorer agreement check, RL training, controller
  comparison, reference search, the two scene checks (`scene_gap.py`, `moving_scene.py`), and
  their results.
- `viewer/`: a static web page that shows a scene, an instance and an answer in 3-D (below).
- `tests/`: `python -m pytest tests`.

## Use

    NS3_BUILD=/path/to/ns-3.48/build ns3/build.sh        # once; needs an optimised build with lte
    python -m skycellbench generate --seed 7 --task constrained -o inst.json
    python -m skycellbench generate --seed 7 --scene cincinnati-downtown -o inst.json
    python -m skycellbench baseline inst.json --method kmeans -o place.json
    python -m skycellbench score inst.json place.json --tier fast
    python -m skycellbench score inst.json place.json --tier ns3

The scripts in `experiments/` are run from the repository root with `PYTHONPATH=.`.

## Viewer

    python -m skycellbench view inst.json place.json -o scored.json     # add --tier ns3 for ns-3
    python -m http.server 8000                                          # from the repository root
    # then open http://localhost:8000/viewer/?instance=../inst.json&answer=../scored.json

`view` saves the answer together with what the scorer made of it: per user the throughput, the
UAV that serves it and whether it sees that UAV. The page draws the terrain, the building
footprints at their heights, streets and water, the users coloured by the share of their demand
that was delivered, the UAVs, and a line from every user to its UAV, solid with line of sight
and faint without. No-fly zones, the gateway and the backhaul hops are drawn on the constrained
task, and answers with several epochs get a slider. It computes nothing itself. Opened without
`instance` and `answer` it offers the examples in `viewer/examples`
(`experiments/viewer_examples.py`); files can also be opened from the panel. It needs a local
web server only because browsers do not load data files from disk; there is no build step and no
network access.

The sky, lighting, free camera, drone model and building materials are code from DroneRTS (MIT),
kept in `viewer/dronerts` with its licence and a list of what was changed; Three.js (MIT) is in
`viewer/vendor/three`. The page shows the OpenStreetMap and USGS credits the scene requires.

From Python: `skycellbench.generate(seed, task, scene=None)`, `skycellbench.score_fast(inst, uavs)`,
`skycellbench.score_ns3(inst, uavs)`, and for the blind task `inst.public()` and `skycellbench.Probe(inst)`.

## Not done yet

The LLM baseline, freezing the evaluation protocol (`PROTOCOL.md` is a draft), the held-out run
with ns-3, and the licence. For the scene: a held-out set, RL policies trained on it, and the
choice of whether results are reported on flat ground, on the scene or on both.

## Do the two scorers agree?

`experiments/agreement.py` scored 120 placements on 3 instances with both scorers (random,
k-means, perturbed k-means, and local-search results at four budgets).

| Instance | Spearman, all 40 placements | Spearman, the 12 search placements | Mean gap |
|---|---|---|---|
| 5 UAVs, 200 users | 0.99 | 0.78 | 0.058 |
| 3 UAVs, 150 users | 0.99 | 0.90 | 0.037 |
| 8 UAVs, 250 users | 1.00 | 0.94 | 0.054 |

The closed-form scorer ranks placements almost as ns-3 does, and it is optimistic: on average
0.05 higher, and up to 0.13 higher for the best placements. Among good placements that differ by
a few hundredths the ranking is less reliable, so reported scores must come from ns-3.

On the public set (`experiments/compare.py public --ns3`, 170 answers from six controllers on
30 instances, generated after the check above):

| Task | Answers | Spearman | Mean gap | Largest gap | Controller pairs ranked the same |
|---|---|---|---|---|---|
| `place` | 60 | 0.99 | 0.059 | 0.165 | 147 of 150 |
| `blind` | 50 | 0.99 | 0.051 | 0.155 | 100 of 100 |
| `constrained` | 60 | 0.98 | 0.036 | 0.137 | 140 of 147 |

The gap is closed-form minus ns-3 and is never negative. A controller pair is two
controllers on one instance; ns-3 ranks them as the closed-form scorer does in 387 of 397 cases.
Repeating ns-3 with other run numbers changes little: over 15 answers with 4 runs each, the
standard deviation of the score is 0.0015 on average and 0.005 at most.

## How hard is each task?

Every controller on the 10 public instances of each task (3, 5 or 8 UAVs, 150 to 250 users),
mean ns-3 score, with the closed-form score in brackets. These are development numbers, not
reported results: the held-out set has not been scored.

| Task | Do-nothing | Random | k-means (3 calls) | RL (29 calls) | Search, 30 calls | Search, 4,000 calls | Reference, 200,000 calls |
|---|---|---|---|---|---|---|---|
| `place` | 0.19 (0.26) | 0.40 (0.43) | 0.68 (0.75) | 0.53 (0.58) | 0.62 (0.69) | 0.73 (0.81) | (0.84) |
| `blind` | 0.19 (0.26) | 0.43 (0.46) | not allowed | 0.55 (0.61) | 0.58 (0.64) | not allowed | (0.86) |
| `constrained` | 0.07 (0.10) | 0.16 (0.20) | 0.21 (0.24) | 0.25 (0.28) | 0.28 (0.31) | 0.47 (0.53) | (0.59) |

The order of the controllers is the same on both scorers in every task. The reference answers
have not been scored with ns-3, and the distances to the reference below are closed-form scores.

Controllers are compared at two budgets of scorer calls per instance: 30 (the probe budget of
the blind task) and 4,000. The RL policy is trained per task for 600,000 steps on generated
instances (`experiments/train_rl.py`), which is a cost outside the per-instance budget, and
answers with one rollout of 29 moves.

The reference (`experiments/headroom.py`) is a local search with 200,000 calls and full knowledge
of the instance, including the users on the blind task. It is the best score found, not a proven
optimum. It shows where room is left: at 4,000 calls search is within 0.03 of it on `place` and
0.06 on `constrained`; at 30 calls every controller is at least 0.09 below it on `place`, 0.22 on
`blind` and 0.28 on `constrained`. No controller beats the reference on any instance.

## On the Cincinnati scene

Everything above this section was measured on flat ground. The same checks on
`instances/public-cincinnati-downtown` (the public seeds on the scene, 30 instances):

**The scorers agree as they do on flat ground** (`experiments/compare.py
public-cincinnati-downtown --ns3`, 170 answers from the six controllers):

| Task | Answers | Spearman | Mean gap | Largest gap | Controller pairs ranked the same |
|---|---|---|---|---|---|
| `place` | 60 | 1.00 | 0.051 | 0.152 | 149 of 150 |
| `blind` | 50 | 0.99 | 0.045 | 0.166 | 99 of 100 |
| `constrained` | 60 | 0.99 | 0.033 | 0.094 | 146 of 147 |

The gap is closed-form minus ns-3. It is negative for one answer of the 170, by 0.003.

**Controllers**, mean ns-3 score with the closed-form score in brackets:

| Task | Do-nothing | Random | k-means (3 calls) | RL (29 calls) | Search, 30 calls | Search, 4,000 calls | Reference, 50,000 calls |
|---|---|---|---|---|---|---|---|
| `place` | 0.21 (0.27) | 0.45 (0.48) | 0.63 (0.69) | 0.50 (0.54) | 0.59 (0.64) | 0.72 (0.79) | (0.82) |
| `blind` | 0.22 (0.29) | 0.46 (0.49) | not allowed | 0.49 (0.53) | 0.55 (0.59) | not allowed | not run |
| `constrained` | 0.08 (0.11) | 0.22 (0.25) | 0.31 (0.35) | 0.29 (0.31) | 0.40 (0.43) | 0.52 (0.56) | (0.61) |

The RL policies are the ones trained on flat ground; they were not retrained. The reference here
is a 50,000-call search, not the 200,000 calls used on flat ground.

**Does the scene make position matter more?** `experiments/scene_gap.py` compares the k-means
answer with the reference search, on the scene and on the old channel with the same users (the
same instance with its scene removed). Closed-form scores, 10 instances per task:

| Task | Scene: k-means, reference, gap | Old channel: k-means, reference, gap | Gap larger on the scene in |
|---|---|---|---|
| `place` | 0.688, 0.821, **0.133** | 0.767, 0.843, **0.076** | 9 of 10 instances |
| `constrained` | 0.347, 0.613, **0.266** | 0.250, 0.569, **0.319** | 3 of 10 instances |

On `place` the gap nearly doubles: k-means loses 0.08 and a search wins most of it back by
finding positions that see the users. On `constrained` the gap shrinks, because k-means itself
does better on the scene. There the chain to the gateway keeps the UAVs far from many users
(650 m on average for k-means), where the old channel gives almost no line of sight to anyone,
while real streets give it to some: the mean excess loss on the serving link of the k-means
answers is 11.9 dB on the scene against 15.4 dB on the old channel.

`experiments/los_curve.py` measures the cause over the whole scene
(`experiments/results/los-curve.txt`; 1.2 million links between random outdoor points and random
UAV positions 50 to 300 m above the ground). The scene gives more line of sight than the old
channel at low elevation angles and less at steep ones:

| Elevation angle | 3 to 6° | 6 to 10° | 10 to 15° | 15 to 20° | 20 to 30° | 30 to 45° | 45 to 60° |
|---|---|---|---|---|---|---|---|
| Links with line of sight, scene | 17% | 26% | 36% | 48% | 60% | 73% | 83% |
| Line-of-sight probability, old channel | 5% | 8% | 14% | 26% | 51% | 86% | 99% |

So the scene does not make position matter more everywhere. Near its users and steeply above
them, where the `place` answers are (about 30°), a UAV is no longer almost sure to see them, and
finding the positions that do is what the search wins. Far from them and at a low angle, where
the gateway chain puts the `constrained` k-means answers (13°), the scene is kinder than the old
channel: 43% of their serving links have line of sight against the 24% the old channel expects.

### A fourth task, `moving`, was removed

It had users that move over 8 epochs, hotspots that start and end, and a limit on how far a UAV
moves per epoch. It did not test planning (`experiments/results/moving-lookahead.txt`): UAVs that
never move scored within 0.03 of UAVs that take the best position in every epoch with no speed
limit. Three redesigns (short events at more sites than UAVs, on 2, 4 and 10 km areas) raised
that difference to 0.1 at most, and not on every instance. The reason is the radio model: a UAV
at altitude reaches users kilometres away, so the score is limited by shared capacity and
interference, not by coverage, and following the users gains little. The instance format and
both scorers still accept a placement per epoch, but no task uses it.

**The test was repeated on the scene, and the task is still not added**
(`experiments/moving_scene.py`, results in `experiments/results/moving-scene.txt`; closed-form
scores, four instances of each kind, each also run on the old channel with the same users):

| Instances | Free minus static, scene | Free minus static, old channel | Best plan under a speed limit minus static, scene |
|---|---|---|---|
| hotspots that drift, start and end; 8 epochs | 0.051 (0.021 to 0.076) | 0.023 (0.014 to 0.044) | 0.022 at 100 m per epoch |
| short events at 2 to 3 sites per UAV, demand tripled; 12 epochs | 0.091 (0.084 to 0.104) | 0.057 (0.044 to 0.066) | 0.007 at 150 m per epoch |

The scene about doubles what free movement is worth. With drifting hotspots it is still at the
0.05 line and not clearly above it: the static search had fewer calls than the free one, and with
four times as many the two largest differences fell from 0.076 and 0.069 to 0.062 and 0.050
(`experiments/moving_static_check.py`). With short events it is above the line on all four
instances (0.088 and 0.086 on the two rechecked). But no plan that respects a speed limit came
close to it: the whole-trajectory search gained 0.004 on average, and following the users epoch
by epoch was as often worse than staying put as better. Whether that is the limit being too
tight for the distance between sites or the searches being too weak is not known, so there is no
evidence yet that such a task would reward planning. A first version of the event instances,
with the normal demand, was too easy to show anything: static UAVs already scored 1.0.
