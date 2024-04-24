import autogen.trace as trace
import random

seed = 0
random.seed(seed)
x = random.random()


def test():
    x = random.random()
    return x


random.seed(seed)
x1 = test()
random.seed(seed)
x2 = test()
assert x1 == x2


obj = 1
print("outside obj id", id(obj))


@trace.trace_op(trainable=True)
def test():
    x = random.random()
    x = obj + x
    print("inside obj id", id(obj))
    return x


random.seed(seed)
x1 = test()
random.seed(seed)
x2 = test()
assert x1 == x2
