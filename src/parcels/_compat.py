"""Import helpers for compatability between installations."""


# for compat with v3 of Parcels when users provide `initial=attrgetter("...")` to a Variable
# so that particle initial state matches another variable
class _AttrgetterHelper:
    """
    Example usage

    >>> _attrgetter_helper = _AttrgetterHelper()
    >>> _attrgetter_helper.some_attribute
    'some_attribute'
    >>> from operator import attrgetter
    >>> attrgetter("some_attribute")(_attrgetter_helper)
    'some_attribute'
    """

    def __getattr__(self, name):
        return name


_attrgetter_helper = _AttrgetterHelper()
