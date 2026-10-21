# Parcels documentation

Welcome to the documentation of Parcels. **Parcels** provides a set of Python classes and methods to create customisable particle tracking simulations using gridded output from (ocean) circulation models. Parcels can be used to track passive and active particulates such as water, plankton, plastic, and fish.

```{figure} _static/homepage.gif
:class: dark-light
```

_Animation of virtual particles carried by ocean surface flow in the global oceans. The particles are advected with Parcels in data from the [Copernicus Marine Service](https://marine.copernicus.eu). See [here](./user_guide/examples/tutorial_homepage_animation.md) for the script to create this animation yourself._

```{note}
You are viewing the documentation for Parcels {{ version }}. You can browse the documentation for older versions by using the version switcher in the bottom right.
```

**Useful links**: [Installation instructions](user_guide/getting_started/installation) | [Discussions on GitHub](https://github.com/Parcels-code/parcels/discussions) | [Issue on GitHub](https://github.com/Parcels-code/parcels/issues) | [Parcels website](https://parcels-code.org/) | [CLAM community website](https://clam-community.github.io/) | [API reference](reference/parcels/index)

`````{grid} 1 2 2 2
:gutter: 4
:padding: 2 2 0 0
:class-container: sd-text-center

````{grid-item-card} Getting started
:shadow: md

New to Parcels? Check out the [installation instructions](user_guide/getting_started/installation), run the [quickstart tutorial](user_guide/getting_started/tutorial_quickstart), and learn the [key concepts](user_guide/getting_started/explanation_concepts) to understand the package.

+++

```{button-ref} user_guide/index
:ref-type: doc
:color: secondary
:expand:

Get started!
```
````
````{grid-item-card} How to?
:shadow: md

Wondering how to load a `FieldSet` or write a `Kernel`? Find tutorials and explainers to these and other questions here.

+++

```{button-ref} user_guide/index
:ref-type: doc
:color: secondary
:expand:

To the user guide
```
````
````{grid-item-card} Development
:shadow: md

We encourage anyone to help improve Parcels: read our guidelines to get started!

+++

```{button-ref} development/index
:ref-type: doc
:color: secondary
:expand:

Contributing guidelines
```
````
````{grid-item-card} Community
:shadow: md

Want to interact with other users and Parcels developers?

+++

```{button-ref} community/index
:ref-type: doc
:color: secondary
:expand:

Connect with our community!
```
````
`````

```{toctree}
:maxdepth: 2
:hidden:

Home <self>
User guide <user_guide/index>
Community <community/index>
Development <development/index>
API reference <reference/parcels/index>
Parcels website <https://parcels-code.org/>
```
