__all__ = ["FieldSetWarning", "FileWarning", "KernelWarning", "ParticleSetWarning"]


class FieldSetWarning(UserWarning):
    """Warning that is raised when there are issues in the construction of the FieldSet or its Grid.

    These warnings are often caused by issues in the input data dimensions
    or options selected when loading data into a FieldSet.
    """

    pass


class ParticleSetWarning(UserWarning):
    """Warning that is raised when there are issues in the construction of the ParticleSet."""

    pass


class FileWarning(UserWarning):
    """Warning that is raised when there are issues with input or output files.

    These warnings can be related to file chunking, naming, or decoding issues.
    Chunking issues in particular may negatively impact performance
    """

    pass


class FieldEvalWarning(UserWarning):
    """Warning that is raised when there are issues during the evaluation of a Field.

    These warnings can be related to out-of-bounds indices during interpolation,
    or other issues that arise during the evaluation of a Field at particle positions.
    """

    pass


class KernelWarning(RuntimeWarning):
    """Warning that is raised when there are issues with the Kernel.

    These warnings often result from issues in the FieldSet or user-defined Kernel
    that are passed into the Parcels Kernel loop.
    """

    pass
