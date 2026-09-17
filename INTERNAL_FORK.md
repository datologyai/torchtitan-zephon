# DatologyAI integration fork

This repository is a private copy of PyTorch TorchTitan with opt-in Zephon
training and validation dataloaders. It is intentionally outside GitHub's
public fork network. The intended end state is a public reference integration
that demonstrates how to replace an existing framework dataloader without
changing its model or training stack.

User documentation belongs in the [Zephon example guide](examples/zephon/README.md)
and [integration reference](docs/zephon.md). This file is for maintainers of the
fork.

## Repository policy

- `origin` is `datologyai/torchtitan-zephon`; `upstream` is
  `pytorch/torchtitan`.
- Keep `main` as the synchronized upstream reference. Review Zephon changes on
  a feature branch and merge them through a pull request.
- Preserve upstream copyright and licensing notices.
- Follow TorchTitan's contribution policy, including DCO sign-off and the
  repository's cryptographic-signing requirements.
- Keep Zephon-specific behavior confined to the override, examples, focused
  tests, dependency pin, and documentation so upstream synchronization remains
  tractable.

## Continuous integration

GitHub Actions is disabled for this private fork. The upstream workflow files
remain in the tree to minimize synchronization conflicts, but no hosted jobs
should run. Do not enable scheduled, release, package-publishing, container,
notification, or upstream-internal workflows in this repository.

While Zephon is private, run the release-facing validation locally from a
TorchTitan development environment with GitHub access to `datologyai/zephon`:

```bash
scripts/validate_zephon_install.sh
```

The script builds a clean temporary environment, installs the pinned Zephon
release, runs the focused adapter tests, and executes the CPU elastic demo. Set
`ZEPHON_WHEEL=/path/to/zephon.whl` to validate a release candidate wheel.

Before opening a pull request, also run:

```bash
pre-commit run --all-files
```

Pyrefly may require the expected Linux/PyTorch development environment; record
any intentional local skip in the pull request.

## Updating from PyTorch

```bash
git fetch upstream
git switch main
git merge --ff-only upstream/main
git push origin main
```

After an upstream sync, rebase the Zephon integration branch and resolve any
changes around the dataloader configuration and override seam. Do not restore
or enable upstream automation unless its behavior is deliberately adopted by
this project.

## Public-release checklist

Complete every item before changing the repository visibility to public.

### Dependency and access

- [ ] Publish an approved public Zephon release and replace the private Git pin
      in `requirements-zephon.txt` with its public installation source.
- [ ] Confirm a clean machine without DatologyAI GitHub or package credentials
      can install every dependency used by the examples and tests.
- [ ] Remove private package indexes, repository URLs, credentials, internal
      hostnames, account identifiers, and employee-specific paths from tracked
      files and GitHub configuration.
- [ ] Search the complete Git history, not only the current tree, for secrets
      or private artifacts; rewrite history if anything sensitive was committed.

### Licensing and provenance

- [ ] Obtain approval to publish the integration, fixtures, scripts, and Zephon
      API examples.
- [ ] Verify that all added files have appropriate copyright and license
      treatment and that all fixtures are redistributable.
- [ ] Preserve attribution to PyTorch TorchTitan and document the upstream
      repository and synchronization model.
- [ ] Confirm that the repository name, description, topics, and license do not
      imply endorsement by PyTorch or upstream ownership of Zephon.

### Documentation and usability

- [ ] Replace every statement that Zephon is private with public installation
      and support information.
- [ ] Run the CPU elastic demo from a fresh clone using only documented commands.
- [ ] Run the full GPU checkpoint/resume demo from a fresh clone, including a
      two-GPU to one-GPU topology change.
- [ ] Verify all Markdown links, commands, paths, expected output, and current
      limitations in `README.md`, `examples/zephon/README.md`, and
      `docs/zephon.md`.
- [ ] Ensure the Megatron and TorchTitan guides use the same terminology for
      recipes, token proportions, token estimation, packing, canonical lanes,
      and elastic-resume guarantees.

### Tests and GitHub configuration

- [ ] Enable release-facing Zephon tests in hosted CI without repository
      secrets and confirm they pass on pull requests.
- [ ] Review workflow permissions, third-party actions, branch protection,
      required checks, CODEOWNERS, issue settings, and pull-request settings for
      a public repository.
- [ ] Run secret scanning, dependency review, and an appropriate source/security
      scan; resolve or explicitly accept every finding.
- [ ] Remove, disable, or scope upstream workflows that can publish packages,
      containers, releases, or notifications before enabling GitHub Actions.
- [ ] Confirm the public default branch contains the intended CI policy and that
      the Zephon integration pull request has a reviewable, signed history.

### Publication

- [ ] Decide whether to retain this standalone repository or recreate it as a
      GitHub fork, and document the consequence for upstream synchronization.
- [ ] Add public issue/support guidance and identify maintainers responsible for
      the demo.
- [ ] Record the final private commit SHA, create a release tag for the reviewed
      public baseline, and save the successful clean-clone validation results.
- [ ] Have a second maintainer review this checklist and the visibility change.
- [ ] Change visibility only after every preceding item is complete.
