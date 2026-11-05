---
name: bootstrap-setup
description: One-time setup after installing the generic workflow kit. Discovers this project's purpose, commands and structure, confirms each value with the user, replaces every {{KIT:...}} placeholder in the kit's agents, skills, CLAUDE.md and spec templates, prunes parts that do not apply, self-checks the result, then removes itself.
disable-model-invocation: true
---

# Bootstrap the workflow kit for this project

The kit's agents, skills and templates are generic. They contain `{{KIT:NAME}}` placeholders that this skill replaces with values specific to this project. Run it once, right after `install.sh`.

The user is the product owner: propose values with evidence, but never write a value they have not confirmed.

## Placeholders

| Token | Meaning | Typical evidence |
|---|---|---|
| `PRODUCT_NAME` | Product or repository name | README title, package manifest `name` |
| `PRODUCT_PURPOSE` | 1–3 sentences: what the product does and for whom | README, docs, landing copy |
| `PRIMARY_USERS` | Who uses it and which decisions they make | README, docs, domain models, routes |
| `DEV_COMMAND` | Command that runs the project locally | `scripts.dev`/`start`, Makefile, Procfile |
| `TYPECHECK_COMMAND` | Static type or lint check | `tsc`, `mypy`, `cargo check`, `go vet` |
| `TEST_COMMAND` | Test command | `npm test`, `pytest`, `cargo test`, `go test ./...` |
| `BUILD_COMMAND` | Production build | `npm run build`, `cargo build --release` |
| `VERIFY_COMMAND` | One command that runs typecheck + tests + build | `npm run verify`, `make verify`, CI job |
| `SPEC_DIR` | Folder for direction, canvas and per-feature specs | default `docs/specs` |
| `ENGINEERING_BOUNDARIES` | Extra project rules as `- ` bullet lines | existing CLAUDE.md, CONTRIBUTING, ADRs |
| `DESIGN_SYSTEM_REF` | Where design tokens/components live, or `none` | `tokens.css`, `theme.ts`, `components/ui/` |

## Process

### 1. Preflight

- Run `grep -rn "{{KIT:" .claude CLAUDE.md CLAUDE.workflow.md docs/specs 2>/dev/null | grep -v "skills/bootstrap-setup/"`. This skill's own folder is excluded because it documents the tokens.
  - No matches: report "Workflow kit already bootstrapped" and stop.
  - Otherwise list which files and tokens remain.
- Run `git status --short`. If there are unrelated uncommitted changes, recommend committing or stashing them first, so the bootstrap diff can be reviewed on its own. Continue only if the user agrees.
- Note whether `CLAUDE.workflow.md` exists. If it does, the project already had a `CLAUDE.md` and the two must be merged in step 5.

### 2. Discover (read-only)

Read only what is needed to propose values:

- `README*`, the existing `CLAUDE.md`, `CONTRIBUTING*`, `docs/` index pages.
- Package and build manifests: `package.json`, `pyproject.toml`, `requirements*.txt`, `Makefile`, `Cargo.toml`, `go.mod`, `pom.xml`, `build.gradle`, `Gemfile`, `composer.json`.
- Test, lint and type configuration, and CI workflows (`.github/workflows/`, `.gitlab-ci.yml`). CI often reveals the real verify sequence.
- Frontend presence: framework dependencies, `src/**/*.{tsx,jsx,vue,svelte}`, templates, CSS. Note design tokens or a component library.
- Trust boundaries: auth, payments, PII, external APIs, file uploads, deployment config. These inform `ENGINEERING_BOUNDARIES` and when `security-expert` matters.

For every token, prepare a proposed value and the file that supports it. Mark a value `UNKNOWN` instead of guessing.

### 3. Confirm with the user

Use AskUserQuestion in small batches. Put the evidence-backed proposal first and mark it "(Recommended)":

1. Product name, purpose and primary users.
2. Dev, typecheck, test, build and verify commands.
3. Spec folder (`docs/specs` or another path).
4. Engineering boundaries (for example: no new external services, no secrets in code, no schema changes without a migration, supported browsers or runtimes).
5. Does the project have a user interface? If yes, confirm `DESIGN_SYSTEM_REF`.

Questions with no good proposal (often purpose and users) must still be answered by the user. Never write invented product strategy.

### 4. Ensure a single verify command

If no single command runs typecheck + tests + build:

- Propose the smallest addition that fits the stack, for example `"verify": "npm run typecheck && npm test && npm run build"` or a `make verify` target.
- Add it only after the user approves, and run it once to confirm it works.
- If the user declines, use the full chained command as `VERIFY_COMMAND`.

### 5. Apply

- Replace every `{{KIT:NAME}}` token in `.claude/agents/`, `.claude/skills/`, `CLAUDE.md` / `CLAUDE.workflow.md` and `docs/specs/` with the confirmed values. `ENGINEERING_BOUNDARIES` becomes `- ` bullet lines; if none are given, delete its line.
  - Never touch `.claude/skills/bootstrap-setup/`, so the placeholder table above stays intact.
- Remove command lines in "Useful commands" for commands the project does not have, such as typecheck in a plain-JS project.
- If `CLAUDE.workflow.md` exists:
  - Merge it into the existing `CLAUDE.md`.
  - Keep every existing instruction.
  - Add the workflow sections ("Before implementation", "Engineering boundaries", "Definition of done", "Workflow skills and agents"). Combine overlapping sections instead of duplicating them.
  - Show the proposed merged file and get approval before writing it, then delete `CLAUDE.workflow.md`.
- If the user chose a spec folder other than `docs/specs`, move the folder with `git mv`, or `mv` if the files are untracked. Remove `docs/` if it is left empty.

### 6. Prune what does not apply

If the project has no user interface, offer to:

- Delete `.claude/agents/design-expert.md`, `.claude/skills/ui-ux-review/` and `.claude/skills/drop-generic-design/`.
- Remove `design-expert` from the fan-out list in `.claude/skills/write-feature-spec/SKILL.md`, from step 4 and "Workflow skills and agents" in `CLAUDE.md`, and the "Design" line in `SPEC-CANVAS.md` section 7.
- Remove the accessibility bullet in Engineering boundaries and the "Loading, empty, error, and success states" bullet in Definition of done, but only if the user confirms that they do not apply (an API can still have error states).

Prune only after the user agrees.

### 7. Self-check

Report each check's result honestly:

1. `grep -rn "{{KIT:" .claude CLAUDE.md docs 2>/dev/null | grep -v "skills/bootstrap-setup/"` (or the chosen spec folder) returns nothing.
2. Every agent's `skills:` entry names a folder that exists in `.claude/skills/`.
3. Every agent named in `write-feature-spec` and `loop-engineering` exists in `.claude/agents/`.
4. The spec folder contains `PRODUCT-DIRECTION.md`, `SPEC-CANVAS.md` and `README.md`, and the paths in the skills point to it.
5. Run the verify command once and report pass or fail with the relevant output. A failing verify is a finding to report, not something to fix by weakening tests.

### 8. Finish

- Summarize what changed, using `git status --short` and `git diff --stat`.
- List the confirmed values and any assumptions still marked as such.
- Suggest next steps:
  1. Review and fill in `PRODUCT-DIRECTION.md` (decisions, current focus, out of scope).
  2. Restart Claude Code or run `/agents` so the agents are loaded.
  3. Try the flow on a small feature: `/write-feature-spec <request>`.
- Ask the user to confirm deleting `.claude/skills/bootstrap-setup/`, so the skill truly runs once. Delete it only on a yes. If the user says no, tell them the skill stops on its own when no placeholders remain.
