from autogen import trace
from autogen.trace.trace_ops import trace_op, TraceMissingInputsError
from autogen.trace.nodes import Node, node
from autogen.trace.utils import for_all_methods, contain

x = Node(1, name="node_x")
y = Node(2, name="node_y")
condition = Node(True)


# Test node_dict==None
@trace_op("[auto_cond] This selects x if condition is True, otherwise y.", node_dict=None)
def auto_cond(condition: Node, x: Node, y: Node):
    """
    A function that selects x if condition is True, otherwise y.
    """
    # You can type comments in the function body
    x, y, condition = x, y, condition  # This makes sure all data are read
    return x if condition else y


output = auto_cond(condition, x, y)
assert output.name.split(":")[0] == "auto_cond"
assert output._inputs[x.name] is x and output._inputs[y.name] is y and output._inputs[condition.name] is condition


# Test node_dict=='auto'
# here we use the signature to get the keys of message_node._inputs
@trace_op("[cond] This selects x if condition is True, otherwise y.", node_dict="auto")
def cond(condition: Node, x: Node, y: Node):
    x, y, condition = x, y, condition  # This makes sure all data are read
    return x if condition else y


output = cond(condition, x, y)
assert output.name.split(":")[0] == "cond"
assert output._inputs["x"] is x and output._inputs["y"] is y and output._inputs["condition"] is condition


# Test dot is okay for operator name
@trace_op("[fancy.cond] This selects x if condition is True, otherwise y.", node_dict="auto")
def fancy_cond(condition: Node, x: Node, y: Node):
    x, y, condition = x, y, condition  # This makes sure all data are read
    return x if condition else y


output = fancy_cond(condition, x, y)
assert output.name.split(":")[0] == "fancy.cond"
assert output._inputs["x"] is x and output._inputs["y"] is y and output._inputs["condition"] is condition


# Test wrapping a function that returns a node
@trace_op("[add_1] Add input x and input y")
def foo(x, y):
    z = x + y
    return z


z = foo(x, y)
assert z.data == 3
assert set(z.parents) == {x, y}


# Test tracing class method
class Foo:
    @trace_op("[Foo.add] Add input x and input y")
    def add(self, x, y):
        z = x + y
        return z


foo = Foo()
z = foo.add(x, y)


# Test composition of trace_op with for all_all_methods
@for_all_methods
def test_cls_decorator(fun):
    def wrapper(*args, **kwargs):
        return fun(*args, **kwargs)

    return wrapper


@test_cls_decorator
class Foo:
    # Test automatic description generation
    @trace_op()
    def add(self, x, y):
        z = x + y
        return z


foo = Foo()
z = foo.add(x, y)


# Test functions with *args and *kwargs and node_dict=None
@trace_op(node_dict=None, unpack_input=False)
def fun(a, args, kwargs, *_args, **_kwargs):
    print(a.data)
    print(args.data)
    print(kwargs.data)
    return a


x = fun(
    node(1), node("args"), node("kwargs"), node("_args_1"), node("_args_2"), b=node("_kwargs_b"), c=node("_kwargs_c")
)
print(x, x.inputs)
assert len(x.inputs) == 3


# Test functions with *args and *kwargs and node_dict='auto'
@trace_op(node_dict="auto")  # This is the default behavior
def fun(a, args, kwargs, *_args, **_kwargs):
    print(a)
    print(args)
    print(kwargs)
    for v in _args:
        print(v)
    for k, v in _kwargs.items():
        print(v)
    return a


x = fun(
    node(1), node("args"), node("kwargs"), node("_args_1"), node("_args_2"), b=node("_kwargs_b"), c=node("_kwargs_c")
)
print(x, x.inputs)


# Test stop_tracing
x = node(1)
y = node(2)
with trace.stop_tracing():
    z = x + y
assert z.inputs == {}
assert z == 3


# Test trace_op as an inline decorator
def fun(a, b):
    return a + b


tfun = trace_op()(fun)
assert tfun(node(1), node(2)) == 3


# Test multi-output function
@trace_op(n_outputs=2)
def fun(a, b):
    return a + b, a - b


x, y = fun(node(1), node(2))


@trace_op()  # single output
def fun(a, b):
    return a + b, a - b


x_y = fun(node(1), node(2))
assert isinstance(x_y, Node) and len(x_y) == 2
assert isinstance(x, Node)
assert isinstance(y, Node)

assert x == x_y[0] and y == x_y[1]


# Test trace codes using nodes


@trace_op(traceable_code=True)  # set unpack_input=False to run node-based codes
def test(a: Node, b: Node):
    """Complex function."""
    return a + b + 10


x = node(1)
y = node(2)
z = test(x, y)
assert z == (x + y + 10)
assert contain(z.parents, x) and contain(z.parents, y)
assert "test" in z.name

z0 = z.info["output"]  # This is the original output
assert z == z0
assert not contain(z0.parents, x) and not contain(z0.parents, y)
assert "add" in z0.name


# Test external dependencies

external_var = node(0)


@trace_op()  # set unpack_input=False to run node-based codes
def test(a: Node, b: Node):
    """Complex function."""
    return a + b + 10 + external_var.data


x = node(1)
y = node(2)
try:
    z = test(x, y)
except TraceMissingInputsError:
    print("This usage throws an error because external_var is not provided as part of the inputs")


@trace_op(node_dict={"x": external_var})
def test(a: Node, b: Node):
    """Complex function."""
    return a + b + 10 + external_var.data


z = test(x, y)
assert z == (x + y + 10 + external_var.data)
assert contain(z.parents, x) and contain(z.parents, y) and contain(z.parents, external_var)
assert "a" in z.inputs and "b" in z.inputs and "x" in z.inputs


@trace_op(allow_external_dependencies=True)
def test(a: Node, b: Node):
    """Complex function."""
    return a + b + 10 + external_var.data


z = test(x, y)
assert z == (x + y + 10 + external_var.data)
assert contain(z.parents, x) and contain(z.parents, y)
assert contain(z.info["external_dependencies"], external_var)
assert "a" in z.inputs and "b" in z.inputs

# Test get attribute and call


class Foo:
    def __init__(self):
        self.node = node(1)
        self.non_node = 2

    def trace_fun(self, x: Node):
        print(x.data)
        return self.node * 2

    def non_trace_fun(self):
        return self.non_node * 2


foo = node(Foo())
x = node("x")
try:
    foo.node
    foo.trace_fun()
except AttributeError:
    print("The attribute of the wrapped object cannot be directly accessed. Instead use getattr() or call()")


attr = foo.getattr("node")
print(f"foo_node: {attr}\nparents {[(p.name, p.data) for p in attr.parents]}")


attr = foo.getattr("non_node")
print(f"non_node: {attr}\nparents {[(p.name, p.data) for p in attr.parents]}")


fun = foo.getattr("non_trace_fun")
y = fun()
print(f"output: {y}\nparents {[(p.name, p.data) for p in y.parents]}")

fun = foo.getattr("trace_fun")
y = fun(x)

y = foo.call("non_trace_fun")
print(f"output: {y}\nparents {[(p.name, p.data) for p in y.parents]}")

y = foo.call("trace_fun", x)
print(f"output: {y}\nparents {[(p.name, p.data) for p in y.parents]}")


class Foo:
    def __init__(self):
        self.x = node(1)

    def add(self, y):
        return y + 1 + self.x  # node


node_F = node(Foo())
y = node_F.getattr("x")
assert len(y.parents) == 2
assert "getattr" in y.name
assert y == node_F.data.x  # value

add = node_F.getattr("add")
z = add(node(2))
assert len(z.parents) == 2
assert contain(z.parents, add)
assert contain(z.parents[0].parents, node_F)

z2 = node_F.call("add", 2)
assert z2 == z
assert contain(z2.parents[0].parents, node_F)

z2 = node_F.call("add", node(2))
assert z2 == z
assert contain(z2.parents[0].parents, node_F)
