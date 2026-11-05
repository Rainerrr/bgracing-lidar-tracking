# LiDAR Obstacle Tracking project instructions

## Product purpose

A two-container ROS 2 Jazzy pipeline that replays a KITTI raw LiDAR drive as a live sensor, detects obstacles, and tracks them in a world frame with ego-motion-compensated velocities scored against KITTI ground truth.

Primary users: BGRacing Autonomous reviewers, who clone the repo, run the containers, inspect Foxglove and `output/`, and decide whether the submission is reproducible, correct and well understood. Secondarily the author, who tunes parameters and must defend every design choice.

Do not assume that a feature request implies a particular UI, score, alert, or algorithm. See `docs/specs/PRODUCT-DIRECTION.md`.

## Project organization

- `docs/spec.md` is the main project spec. It defines the architecture and the phases, each with an end goal, scope and exit criteria.
- Work happens one phase at a time. Each phase gets its own folder `docs/specs/phase-N-<slug>/` with `FEATURE_SPEC.md`, `PLAN.md`, `REVIEW.md` and `EVIDENCE/`. In the steps below, `<feature-slug>` means that phase folder.
- A phase spec adds detail and sets thresholds. If it changes a phase's scope or exit criteria, update `docs/spec.md` too, so the main spec stays the single source.

## Before implementation

1. Read `docs/spec.md` (the current phase's section), `docs/specs/PRODUCT-DIRECTION.md` and `docs/specs/SPEC-CANVAS.md`.
2. Explore the current product and code before proposing changes.
3. Write an explicit goal, non-goals, acceptance criteria, constraints, and verification plan.
4. Fan out focused, read-only perspectives to product, architecture, and QA. Add the security expert only when the change crosses a trust boundary, handles sensitive data or external input, or expands tool/deployment access.
5. Fan in their advice into one coherent feature specification at `docs/specs/<feature-slug>/FEATURE_SPEC.md`: remove duplicates, resolve conflicts and record rejected options. Experts advise; the main agent owns synthesis and consistency. The user approves every product decision.
6. Present the plan (`docs/specs/<feature-slug>/PLAN.md`) before editing application code.

## Engineering boundaries

- Keep the project runnable with `docker compose up --build` and verifiable with `docker compose build`.
- Markers, plots and reports carry text labels and never rely on color alone for status.
- Prefer a thin end-to-end slice over many unfinished surfaces.
- Do not rewrite unrelated areas or change fixtures or seed data merely to make a feature appear successful.
- Never weaken or delete a test to satisfy the verification loop.
- Exactly two Docker containers (publisher, processor), both on Ubuntu 24.04, ROS 2 Jazzy and Python 3.12. Base images are pinned by digest.
- A fresh clone reproduces the results with the README commands only. Dependencies come from apt in the Dockerfiles, never pip (except inside a `--system-site-packages` venv). Containers use compose's default bridge network: no X11, no host networking.

## Definition of done

A feature is complete only when:

- It solves the stated user and business problem.
- Every acceptance criterion has visible evidence in `docs/specs/<feature-slug>/EVIDENCE/` or `REVIEW.md`.
- Loading, empty, error, and success states are considered where relevant.
- `docker compose build` passes.
- The current git diff has received an independent review from `change-reviewer`.
- Every review finding names severity, criterion, exact evidence, impact, action and a final `PASS / REPAIR / REPLAN / HUMAN DECISION` verdict.
- The user can explain the decision, tradeoffs, and remaining risks.

## Workflow skills and agents

- `/write-feature-spec`: fan out to `product-expert`, `architecture-expert`, `qa-expert` (+ `security-expert` when triggered) and fan in to one spec.
- `/architecture-plan`: turn the approved spec into a file-level plan.
- `/loop-engineering`: bounded build, test, review and repair loop.

## Useful commands

```bash
docker compose up --build   # run the pipeline
./scripts/verify.sh         # verify: build both images, publisher smoke import, pytest in the processor image
docker compose run --rm --no-deps processor python3 /app/scripts/dataset_stats.py   # dataset stats → output/
docker compose run --rm --no-deps processor python3 /app/scripts/run_offline.py      # detections, recall, timing → output/offline/
```
