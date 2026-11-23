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
- Channel: mean air-to-ground path loss of Al-Hourani et al. (2014), urban parameters. It is a
  smooth function of position, with no random shadowing.
- Demand: 0.5, 1 or 2 Mbit/s per user, constant bit rate.
- A placement that breaks a rule is rejected with the rule named; it has no score.

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
- `skycellbench/fast.py`: closed-form scorer and the probe for the blind task. `skycellbench/link_curve.json`: the measured link curve.
- `skycellbench/ns3.py` and `ns3/uav-score.cc`: ns-3 scorer and its Python wrapper.
- `skycellbench/baselines.py`: centre (do nothing), random, k-means, local search.
- `skycellbench/rl.py`: step-by-step environment (reset/step) and the PPO policy.
- `skycellbench/controllers.py`: the controllers that are compared, behind one call.
- `instances/public`, `instances/heldout`: 10 instances per task each. The held-out set is for
  the reported scores and must not be scored before the protocol is frozen.
- `experiments/`: link-curve measurement, scorer agreement check, RL training, controller
  comparison, reference search, and their results.
- `tests/`: `python -m pytest tests`.

## Use

    NS3_BUILD=/path/to/ns-3.48/build ns3/build.sh        # once; needs an optimised build with lte
    python -m skycellbench generate --seed 7 --task constrained -o inst.json
    python -m skycellbench baseline inst.json --method kmeans -o place.json
    python -m skycellbench score inst.json place.json --tier fast
    python -m skycellbench score inst.json place.json --tier ns3

From Python: `skycellbench.generate(seed, task)`, `skycellbench.score_fast(inst, uavs)`,
`skycellbench.score_ns3(inst, uavs)`, and for the blind task `inst.public()` and `skycellbench.Probe(inst)`.

## Not done yet

The LLM baseline, freezing the evaluation protocol (`PROTOCOL.md` is a draft), the held-out run
with ns-3, and the licence.

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

### A fourth task, `moving`, was removed

It had users that move over 8 epochs, hotspots that start and end, and a limit on how far a UAV
moves per epoch. It did not test planning (`experiments/results/moving-lookahead.txt`): UAVs that
never move scored within 0.03 of UAVs that take the best position in every epoch with no speed
limit. Three redesigns (short events at more sites than UAVs, on 2, 4 and 10 km areas) raised
that difference to 0.1 at most, and not on every instance. The reason is the radio model: a UAV
at altitude reaches users kilometres away, so the score is limited by shared capacity and
interference, not by coverage, and following the users gains little. The instance format and
both scorers still accept a placement per epoch, but no task uses it.
