from dataclasses import is_dataclass, Field
from abc import abstractmethod, ABC
from collections.abc import Sequence, Mapping

from typing import TypeVar, Generic, get_origin, get_args, Any
import pyarrow as pa


T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')
AS = TypeVar('AS', bound=pa.Scalar)
AA = TypeVar('AA', bound=pa.Array)


class ArrowMarshaller(Generic[T, AS, AA], ABC):

    def __init__(self, py_type: type[T], arrow_dtype: pa.DataType):
        self.py_type = py_type
        self.arrow_dtype = arrow_dtype

    @abstractmethod
    def to_arrow(self, py: T) -> AS:
        raise NotImplementedError

    @abstractmethod
    def from_arrow(self, arrow: AS) -> T:
        raise NotImplementedError

    @abstractmethod
    def to_arrow_array(self, py: Sequence[T]) -> AA:
        raise NotImplementedError

    @abstractmethod
    def from_arrow_array(self, arrow: AA) -> Sequence[T]:
        raise NotImplementedError


class BasicTypeArrowMarshaller(ArrowMarshaller[T, AS, AA]):

        def __init__(self, py_type: type[T], arrow_type: pa.DataType):
            super().__init__(py_type, arrow_type)

        def to_arrow(self, py: T) -> AS:
            return pa.scalar(py, type=self.arrow_dtype)

        def from_arrow(self, arrow: AS) -> T:
            return arrow.as_py()

        def to_arrow_array(self, py: Sequence[T]) -> AA:
            return pa.array(py, type=self.arrow_dtype)

        def from_arrow_array(self, arrow: AA) -> Sequence[T]:
            return arrow.to_pylist()

null = BasicTypeArrowMarshaller[type(None), pa.NullScalar, pa.NullArray](type(None), pa.null())
boolean = BasicTypeArrowMarshaller[bool, pa.BooleanScalar, pa.BooleanArray](bool, pa.bool_())
int8 = BasicTypeArrowMarshaller[int, pa.Int8Scalar, pa.Int8Array](int, pa.int8())
int16 = BasicTypeArrowMarshaller[int, pa.Int16Scalar, pa.Int16Array](int, pa.int16())
int32 = BasicTypeArrowMarshaller[int, pa.Int32Scalar, pa.Int32Array](int, pa.int32())
int64 = BasicTypeArrowMarshaller[int, pa.Int64Scalar, pa.Int64Array](int, pa.int64())
uint8 = BasicTypeArrowMarshaller[int, pa.UInt8Scalar, pa.UInt8Array](int, pa.uint8())
uint16 = BasicTypeArrowMarshaller[int, pa.UInt16Scalar, pa.UInt16Array](int, pa.uint16())
uint32 = BasicTypeArrowMarshaller[int, pa.UInt32Scalar, pa.UInt32Array](int, pa.uint32())
uint64 = BasicTypeArrowMarshaller[int, pa.UInt64Scalar, pa.UInt64Array](int, pa.uint64())
float16 = BasicTypeArrowMarshaller[float, pa.HalfFloatScalar, pa.HalfFloatArray](float, pa.float16())
float32 = BasicTypeArrowMarshaller[float, pa.FloatScalar, pa.FloatArray](float, pa.float32())
float64 = BasicTypeArrowMarshaller[float, pa.DoubleScalar, pa.DoubleArray](float, pa.float64())
string = BasicTypeArrowMarshaller[str, pa.StringScalar, pa.StringArray](str, pa.string())
binary = BasicTypeArrowMarshaller[bytes, pa.BinaryScalar, pa.BinaryArray](bytes, pa.binary())
# TODO:
# TODO: time32, time64, timestamp, date32, date64, duration, month_day_nano_interval
# TODO: large_binary, large_string
# TODO: binary_view, string_view
# TODO: decimal128, decimal256


class ListArrowMarshaller(ArrowMarshaller[Sequence[T], pa.ListScalar, pa.ListArray]):

    def __init__(self, elem: ArrowMarshaller[T, AS, AA]):
        self.elem = elem
        super().__init__(Sequence[T], pa.list_(elem.arrow_dtype))

    def to_arrow(self, py: Sequence[T]) -> pa.ListScalar:
        if isinstance(self.elem, BasicTypeArrowMarshaller):
            s = pa.scalar(py, type=self.arrow_dtype)
        else:
            s = pa.scalar([self.elem.to_arrow(sub) for sub in py], type=self.arrow_dtype)
        assert isinstance(s, pa.ListScalar)
        return s

    def from_arrow(self, arrow: pa.ListScalar) -> Sequence[T]:
        if isinstance(self.elem, BasicTypeArrowMarshaller):
            return arrow.as_py()
        else:
            return [self.elem.from_arrow(sub) for sub in arrow]

    def to_arrow_array(self, py: Sequence[Sequence[T]]) -> pa.ListArray:
        if isinstance(self.elem, BasicTypeArrowMarshaller):
            a = pa.array(py, type=self.arrow_dtype)
        else:
            a = pa.array([self.to_arrow(sub) for sub in py], type=self.arrow_dtype)
        assert isinstance(a, pa.ListArray)
        return a

    def from_arrow_array(self, arrow: pa.ListArray) -> Sequence[Sequence[T]]:
        if isinstance(self.elem, BasicTypeArrowMarshaller):
            return arrow.to_pylist()
        return [self.from_arrow(sub) for sub in arrow]


class MapArrowMarshaller(ArrowMarshaller[Mapping[K, V], pa.MapScalar, pa.MapArray]):
    def __init__(self, key: ArrowMarshaller[K, Any, Any], value: ArrowMarshaller[V, Any, Any]):
        self.key = key
        self.value = value
        super().__init__(Mapping[K, V], pa.map_(self.key.arrow_dtype, self.value.arrow_dtype))

    def to_arrow(self, py: T) -> pa.MapScalar:
        s = pa.scalar(py, type=self.arrow_dtype)
        assert isinstance(s, pa.MapScalar)
        return s

    def from_arrow(self, arrow: pa.MapScalar) -> T:
        if isinstance(self.key, BasicTypeArrowMarshaller) and isinstance(self.value, BasicTypeArrowMarshaller):
            return {
                k: v for k, v in arrow
            }
        return {
            self.key.from_arrow(arrow[i][0]): self.value.from_arrow(arrow[i][1])
            for i in range(len(arrow))
        }

    def to_arrow_array(self, py: Sequence[Mapping[K, V]]) -> pa.MapArray:
        if isinstance(self.key, BasicTypeArrowMarshaller) and isinstance(self.value, BasicTypeArrowMarshaller):
            a = pa.array(py, type=self.arrow_dtype)
        else:
            a = pa.array([self.to_arrow(sub) for sub in py], type=self.arrow_dtype)
        assert isinstance(a, pa.MapArray)
        return a

    def from_arrow_array(self, arrow: pa.MapArray) -> Sequence[T]:
        return [self.from_arrow(sub) for sub in arrow]


class StructArrowMarshaller(ArrowMarshaller[T, pa.StructScalar, pa.StructArray]):
    def __init__(self, cls: type[T]):
        assert is_dataclass(cls)
        self.cls = cls
        self.fields = {
            f.name: derive_arrow_marshaller_for_field(f)
            for f in cls.__dataclass_fields__.values()
        }
        arrow_dtype = pa.struct([
            (name, marshaller.arrow_dtype) for name, marshaller in self.fields.items()
        ])
        super().__init__(cls, arrow_dtype)

    def to_arrow(self, py: T) -> pa.StructScalar:
        a = pa.scalar(py.__dict__, type=self.arrow_dtype)
        assert isinstance(a, pa.StructScalar)
        return a

    def from_arrow(self, arrow: pa.StructScalar) -> T:
        return self.cls(**{
            f: self.fields[f].from_arrow(v)
            for f, v in arrow.items()
        })

    def to_arrow_array(self, py: Sequence[T]) -> pa.StructArray:
        a = pa.array([self.to_arrow(sub) for sub in py], type=self.arrow_dtype)
        assert isinstance(a, pa.StructArray)
        return a

    def from_arrow_array(self, arrow: pa.StructArray) -> Sequence[T]:
        return [self.from_arrow(sub) for sub in arrow]


def derive_arrow_marshaller_for_field(f: Field) -> ArrowMarshaller:
    if f.metadata is not None and f.metadata.get('arrow_marshaller') is not None:
        return f.metadata['arrow_marshaller']
    return derive_arrow_marshaller(f.type)


def derive_arrow_marshaller(cls: type) -> ArrowMarshaller:
    if cls is type(None):
        return null
    elif cls is bool:
        return boolean
    elif cls is int:
        return int64
    elif cls is float:
        return float64
    elif cls is str:
        return string
    elif cls is bytes:
        return binary
    elif is_dataclass(cls):
        return StructArrowMarshaller(cls)
    elif get_origin(cls) is not None and issubclass(get_origin(cls), Sequence):
        return ListArrowMarshaller(derive_arrow_marshaller(get_args(cls)[0]))
    elif get_origin(cls) is not None and issubclass(get_origin(cls), Mapping):
        return MapArrowMarshaller(derive_arrow_marshaller(get_args(cls)[0]), derive_arrow_marshaller(get_args(cls)[1]))
    raise NotImplementedError(f"Cannot derive ArrowMarshaller for {cls}")
