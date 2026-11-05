---
name: loop-engineering
description: Execute a feature through a bounded build-test-review-repair loop with explicit evidence and stop conditions. Use only after the specification and implementation plan are approved.
---

# Bounded execution loop

## Inputs

- Approved goal and non-goals from `docs/specs/<feature-slug>/FEATURE_SPEC.md`
- Observable acceptance criteria
- Approved file-level plan from `docs/specs/<feature-slug>/PLAN.md`
- Verification commands and expected evidence
- Visible task list derived from the approved plan
- Maximum of three repair cycles

## Loop

1. Create or refresh the task list from the approved plan. Keep exactly one task in progress.
2. Implement the current bounded task.
3. Run the narrowest relevant check and inspect its output.
4. Mark the task complete only after that check produced evidence; otherwise repair or route the failure.
5. Run `docker compose build` before claiming completion.
6. Ask `change-reviewer` for an independent review.
7. Map every failure or finding to an acceptance criterion or engineering boundary.
8. Repair blockers and repeat verification.
9. Stop after three repair cycles or when all criteria pass.

## Stop rules

- Never delete or weaken a test to create a pass.
- Never change the goal silently.
- Never widen scope during a repair cycle.
- If the same blocker survives three cycles, stop and document the blocker, evidence, and next decision needed.

## Completion report

Write the report to `docs/specs/<feature-slug>/REVIEW.md` and store screenshots, logs or command output in `docs/specs/<feature-slug>/EVIDENCE/`.

Report what changed, evidence for every criterion, commands executed, reviewer findings resolved, remaining risks, and the recommended next step.
