# Contributing to Megatron-DeepSpeed

Thanks for your interest in improving Megatron-DeepSpeed!  This document summarizes the expectations for opening issues and
submitting pull requests.  Following these guidelines helps us keep the project healthy and makes review smoother for everyone.

## Code of conduct

Be respectful and professional.  We follow the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

## Getting started

1. **Fork and clone** the repository, then create a topic branch for your work.
2. **Install the development dependencies** using the instructions in the
   [Megatron-DeepSpeed Quickstart](docs/deepspeed_quickstart.md).  This ensures local environments match the CI workflow.
3. **Run the test suites** locally before sending a pull request:

   ```bash
   pytest tests/unit_tests
   pytest tests/transformer
   ```

   Additional integration or functional tests should be added when a change warrants them.

## Development guidelines

* **Coding style.**  Follow the conventions already used in the surrounding files.  We rely on reviewers to call out
  inconsistencies during code review.
* **Documentation.**  Update or create documentation whenever behavior changes.  The `docs/` directory contains user guides like
  the quickstart.  README updates are welcome when they make the entry points clearer.
* **Dependencies.**  Runtime requirements are managed via `megatron/core/requirements.txt`.  Testing tools belong in
  `requirements-dev.txt`.  Keep the lists minimal and justified by concrete imports.
* **Large features.**  For substantial work, consider opening an issue or discussion first so the community can weigh in on the
  approach.

## Pull request checklist

Before requesting a review, please confirm:

- [ ] Tests pass locally (`pytest tests/unit_tests` and `pytest tests/transformer`).
- [ ] New code paths include adequate automated test coverage.
- [ ] Documentation and example scripts are updated to reflect the change.
- [ ] Commits are logically organized and include clear messages.

We appreciate your contributions and look forward to collaborating!
