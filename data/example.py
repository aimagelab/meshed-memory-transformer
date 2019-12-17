
class Example(object):
    """Defines a single training or test example.
    Stores each column of the example as an attribute.
    """
    @classmethod
    def fromdict(cls, data):
        ex = cls(data)
        return ex

    def __init__(self, data):
        for key, val in data.items():
            super(Example, self).__setattr__(key, val)

    def __setattr__(self, key, value):
        raise AttributeError

    def __hash__(self):
        return hash(tuple(x for x in self.__dict__.values()))

    def __eq__(self, other):
        this = tuple(x for x in self.__dict__.values())
        other = tuple(x for x in other.__dict__.values())
        return this == other

    def __ne__(self, other):
        return not self.__eq__(other)
