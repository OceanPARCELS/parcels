# Policies

## Versioning

Parcels follows [Intended Effort Versioning (EffVer)](https://jacobtomlinson.dev/effver/), where the version number (e.g., v2.1.0) is thought of as `MACRO.MESO.MICRO`.

> MACRO version - you will need to dedicate time to upgrading to this version<br>
> MESO version - some small effort may be required for you to upgrade to this version<br>
> MICRO version - no effort is intended for you to upgrade to this version<br>

When making backward incompatible changes, we will make sure these changes and instructions to upgrade are communicated to the user via change logs or migration guides, and (where applicable) informative error messaging.

Note when conducting research we highly recommend documenting which version of Parcels (and other packages) you are using. This can be as easy as doing `conda env export > environment.yml` alongside your project code. The Parcels version used to generate an output file is also stored as metadata entry in the `.parquet` output file.

### Changes in versioning policies

- In v4.0.0 of Parcels, adopted EffVer which formalises this "SemVer-like" variant we were following - and we adjusted our deprecation policy.
- In [v3.1.0](https://docs.parcels-code.org/en/v3.1.0/community/policies.html) of Parcels, we adopted SemVer-like versioning and deprecation policies

## AI usage policy

_This policy has been adapted from Xarray's AI policy._

**Note:** Some Parcels developers use AI tools as part of their development workflow.
We assume this is now common. Tools, patterns, and norms are evolving fast — this
policy aims to avoid restricting contributors' choice of tooling while ensuring that:

- Reviewers are not overburdened
- Contributions can be maintained
- The submitter can vouch for and explain all changes
- Developers can acquire new skills[^1]

To that end this policy applies regardless of whether the code was written by hand, with
AI assistance, or generated entirely by an AI tool.

[^1]:
    Over-reliance on AI tools has been shown to
    [hinder skill formation amongst software developers](https://arxiv.org/abs/2601.20245).

### Core principle: changes

If you submit a pull request, you are responsible for understanding and having fully reviewed
the changes. You must be able to explain why each change is correct[^2] and how it fits into
the project. Strive to minimize changes to ease the burden on reviewers — avoid
including unnecessary or loosely related changes.

[^2]:
    You may also open a draft PR with changes in order to discuss and receive feedback on the
    best approach if you are not sure what the best way forward is.

### Core principle: communication

PR descriptions, issue comments, and review responses must be your own words. The
substance and reasoning must come from you. Do not paste AI-generated text as
comments or review responses. Please attempt to be concise.

PR descriptions should follow the provided template.

Using AI to improve the language of your writing (grammar, phrasing, spelling, etc.) is
acceptable. Be careful that it does not introduce inaccurate details in the process.

Maintainers reserve the right to delete or hide comments that violate our AI policy or code of conduct.

### Code and tests

#### Review every line

You must have personally reviewed and understood all changes before submitting.

If you used AI to generate code, you are expected to have read it critically and
tested it. As with the (hand-written) PR, the description should explain the approach
and reasoning behind the changes. Do not leave it to reviewers to figure out what
the code does and why.

##### Not acceptable

> I pointed an agent at the issue and here are the changes

> This is what Claude came up with. 🤷

##### Acceptable

> I iterated multiple times with an agent to produce this. The agent wrote the code at my direction,
> and I have fully read and validated the changes.

> I pointed an agent at the issue and it generated a first draft. I reviewed the changes thoroughly and understand the implementation well.

#### Large AI-assisted contributions

Generating code with agents is fast and easy. Reviewing it is not. Making a PR with a large diff
shifts the burden from the contributor to the reviewer. To guard against this asymmetry:

If you are planning a large AI-assisted contribution (e.g., a significant refactor, a
framework migration, or a new subsystem), **open an issue first** to discuss the scope
and approach with maintainers. This helps us decide if the change is worthwhile, how
it should be structured, and any other important decisions.

```{important}
Maintainers reserve the right to close PRs for whatever reason.
This could be because where the scope makes meaningful review
impractical, when they suspect this policy has been violated, or
if the PR is a "drive-by PR" and takes more energy to review than to work on seperately.
Similarly they may request
that large changes be broken into smaller, reviewable pieces.
```

### Documentation

The same core principles apply to both code and documentation. You must review the result
for accuracy and are ultimately responsible for all changes made. Parcels has domain-specific
semantics that AI tools frequently get wrong. Do not submit documentation that you
haven't carefully read and verified.
