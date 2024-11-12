from dataclasses import dataclass
from hypothesis import given, settings
import hypothesis.strategies as st
import flint.arrow as fa
from datetime import timedelta


@dataclass
class Person:
    name: str
    age: int
    gender: bool


strategies = {
    "int8": st.integers(-2**7, 2**7 - 1),
    "int16": st.integers(-2**15, 2**15 - 1),
    "int32": st.integers(-2**31, 2**31 - 1),
    "int64": st.integers(-2**63, 2**63 - 1),
    "string": st.text(),
    "list[int64]": st.lists(st.integers(-2**63, 2**63 - 1)),
    "map[string, int64]": st.dictionaries(st.text(), st.integers(-2**63, 2**63 - 1)),
    "Person": st.builds(Person, name=st.text(), age=st.integers(0, 128), gender=st.booleans())
}

marshallers = {
    "int8": fa.int8,
    "int16": fa.int16,
    "int32": fa.int32,
    "int64": fa.int64,
    "string": fa.string,
    "list[int64]": fa.derive_arrow_marshaller(list[int]),
    "map[string, int64]": fa.derive_arrow_marshaller(dict[str, int]),
    "Person": fa.derive_arrow_marshaller(Person)
}


@given(t=st.sampled_from(list(marshallers.keys())))
@settings(max_examples=128, deadline=timedelta(milliseconds=2000))
def test_type(t):
    print(f"Testing for type {t}")
    marshaller = marshallers[t]

    @given(x=strategies[t])
    def test_elem(x):
        a = marshaller.to_arrow(x)
        assert a.type == marshaller.arrow_dtype
        assert marshaller.from_arrow(a) == x

    @given(x=st.lists(strategies[t]))
    def test_list(x):
        a = marshaller.to_arrow_array(x)
        assert a.type == marshaller.arrow_dtype
        assert marshaller.from_arrow_array(a) == x

    test_elem()
    test_list()

