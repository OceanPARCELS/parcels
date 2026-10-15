# Documentation Notes

## Vision

We believe a clear documentation is important to community building, reproducibility, and transparency in our open-source
project. To make it easier to write our documentation in a consistent way, here we outline a brief vision for our
documentation based heavily on a few common resources.

```{note}
TODO: outline functions of the documentation based on resources
```

### Resources

- [Divio Documentation System](https://docs.divio.com/documentation-system/)
- [PyOpenSci Documentation Guide](https://www.pyopensci.org/python-package-guide/documentation/index.html#)
- [Write the Docs Guide](https://www.writethedocs.org/guide/)
- [NumPy Documentation Article](https://labs.quansight.org/blog/documentation-as-a-way-to-build-community)

## Notebook execution

We run the notebooks in our documentation using [MyST-NB](https://myst-nb.readthedocs.io/en/latest/index.html). Here is
a table showing the latest notebook execution:

```{nb-exec-table}

```

## Style guide

- **Prefer `import parcels` over `from parcels import class` in tutorials and how-to guides** so its obvious in later
  code cells which classes and methods are part of Parcels.
- [**Avoid too much Repitition In Documentation**](https://www.writethedocs.org/guide/writing/docs-principles/#arid):
  tutorials and how-to guides notebooks will often have repetition of the general Parcels steps, (e.g., imports ) -
  this is needed so that users have complete examples that they can copy and experiment with.`. We try to limit each page
  in the documentation to a small number of examples.
- Introduce links and cross-references to maximize discoverability of documentation. This also reduces the necessity for
  repetition in notebooks.
- **Import packages at the top of the section in which they are first used** to show what they are used for.
- **Write documentation in first person plural ("we").** In our open source code, tutorials and guides can be written
  by any developer or user, so the documentation teaches all of us how to do something with Parcels. Sometimes it can be
  more natural to take on the tone of a teacher, writing to a student/learner, in which case it is okay to use "you".
  Please refrain from using impersonal subjects such as "the user".
- Use British English in tutorial prose. This is validated by a spellchecker that is run in CI.
- We recommend hard wrapping prose in markdown so that reading it becomes easier in any editor.
