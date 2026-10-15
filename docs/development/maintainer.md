# Maintainers notes

> Workflow information mainly relevant to maintainers

## PR review workflow

- Submit a PR (mark as draft if your feature isn't ready yet, but still want to share your work)
- Request PR to be reviewed by at least one maintainer. Other users are also welcome to submit reviews on PRs.
- Implement or discuss suggested edits
- Once PR is approved:
  - Original author merges the PR (if original author has sufficient permissions)
  - Wait for maintainer to merge
  - If more edits are required: Implement edits and re-request review if changes are significant
- Close linked issue

---

- If PR is automated (i.e., from dependabot or similar), maintainer can review and merge.

## Release checklist

Main workflow:

- Run the validation test suite (`pixi run tests-validation`)
- Go to GitHub, draft new release. Enter name of version and "create new tag" if it doesn't already exist. Click "Generate Release Notes". Currate release notes as needed. Look at a previous version release to match the format (title, header, section organisation etc.)
- (optional - for fast conda releases) Go to [conda-forge/parcels-feedstock](https://github.com/conda-forge/parcels-feedstock), create a new issue (select the "Bot Commands" issue from the menu) with title `@conda-forge-admin, please update version`
  - Approve PR and merge on green
- Check ["publish to PyPI" workflow](https://github.com/Parcels-code/Parcels/actions/workflows/pypi-release.yml) succeeded

Added considerations:

- Update version, DOI, and release date in `CITATION.cff` file (use [Parcels Zenodo entry](https://zenodo.org/records/14001000) as reference)
- Update parcels-code.org
  - Parcels development status
  - Check feature tiles
- (once package is available on conda) Re-build the Binder
- Ask for the shared Parcels environment on [Lorenz](https://github.com/IMAU-oceans/Lorenz) to be updated
