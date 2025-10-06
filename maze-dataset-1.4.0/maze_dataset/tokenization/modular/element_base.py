"""provides the base `_TokenizerElement` class and related functionality for modular maze tokenization

see the code in `maze_dataset.tokenization.modular.elements` for examples of subclasses of `_TokenizerElement`
"""

import abc
from typing import (
	Any,
	Callable,
	Literal,
	TypeVar,
)

from muutils.json_serialize import (
	SerializableDataclass,
	serializable_dataclass,
	serializable_field,
)
from muutils.json_serialize.util import _FORMAT_KEY
from muutils.misc import flatten
from zanj.loading import load_item_recursive

from maze_dataset.tokenization.modular.hashing import _hash_tokenizer_name

# from maze_dataset import SolvedMaze


@serializable_dataclass(frozen=True, kw_only=True)
class _TokenizerElement(SerializableDataclass, abc.ABC):
	"""Superclass for tokenizer elements.

	Subclasses contain modular functionality for maze tokenization.

	# Development
	> [!TIP]
	> Due to the functionality of `get_all_tokenizers()`, `_TokenizerElement` subclasses
	> may only contain fields of type `utils.FiniteValued`.
	> Implementing a subclass with an `int` or `float`-typed field, for example, is not supported.
	> In the event that adding such fields is deemed necessary, `get_all_tokenizers()` must be updated.

	"""

	# TYPING: type hint `v` more specifically
	@staticmethod
	def _stringify(k: str, v: Any) -> str:  # noqa: ANN401
		if isinstance(v, bool):
			return f"{k}={str(v)[0]}"
		if isinstance(v, _TokenizerElement):
			return v.name
		if isinstance(v, tuple):
			return f"{k}={''.join(['(', *[str(x) + ', ' for x in v], ')'])}"
		else:
			return f"{k}={v}"

	@property
	def name(self) -> str:
		members_str: str = ", ".join(
			[self._stringify(k, v) for k, v in self.__dict__.items() if k != "_type_"],
		)
		output: str = f"{type(self).__name__}({members_str})"
		if "." in output and output.index("(") > output.index("."):
			return "".join(output.split(".")[1:])
		else:
			return output

	def __str__(self) -> str:
		return self.name

	# TYPING: type hints for `__init_subclass__`?
	def __init_subclass__(cls, **kwargs):  # noqa: ANN204
		"""Hack: dataclass hashes don't include the class itself in the hash function inputs.

		This causes dataclasses with identical fields but different types to hash identically.
		This hack circumvents this by adding a slightly hidden field to every subclass with a value of `repr(cls)`.
		To maintain compatibility with `all_instances`, the static type of the new field can only have 1 possible value.
		So we type it as a singleton `Literal` type.
		muutils 0.6.1 doesn't support `Literal` type validation, so `assert_type=False`.
		Ignore Pylance complaining about the arg to `Literal` being an expression.
		"""
		super().__init_subclass__(**kwargs)
		# we are adding a new attr here intentionally
		cls._type_ = serializable_field(  # type: ignore[attr-defined]
			init=True,
			repr=False,
			default=repr(cls),
			assert_type=False,
		)
		cls.__annotations__["_type_"] = Literal[repr(cls)]

	def __hash__(self) -> int:
		"Stable hash to identify unique `MazeTokenizerModular` instances. uses name"
		return _hash_tokenizer_name(self.name)

	@classmethod
	def _level_one_subclass(cls) -> type["_TokenizerElement"]:
		"""Returns the immediate subclass of `_TokenizerElement` of which `cls` is an instance."""
		return (
			set(cls.__mro__).intersection(set(_TokenizerElement.__subclasses__())).pop()
		)

	def tokenizer_elements(self, deep: bool = True) -> list["_TokenizerElement"]:
		"""Returns a list of all `_TokenizerElement` instances contained in the subtree.

		Currently only detects `_TokenizerElement` instances which are either direct attributes of another instance or
		which sit inside a `tuple` without further nesting.

		# Parameters
		- `deep: bool`: Whether to return elements nested arbitrarily deeply or just a single layer.
		"""
		if not any(type(el) == tuple for el in self.__dict__.values()):  # noqa: E721
			return list(
				flatten(
					[
						[el, *el.tokenizer_elements()]
						for el in self.__dict__.values()
						if isinstance(el, _TokenizerElement)
					],
				)
				if deep
				else filter(
					lambda x: isinstance(x, _TokenizerElement),
					self.__dict__.values(),
				),
			)
		else:
			non_tuple_elems: list[_TokenizerElement] = list(
				flatten(
					[
						[el, *el.tokenizer_elements()]
						for el in self.__dict__.values()
						if isinstance(el, _TokenizerElement)
					]
					if deep
					else filter(
						lambda x: isinstance(x, _TokenizerElement),
						self.__dict__.values(),
					),
				),
			)
			tuple_elems: list[_TokenizerElement] = list(
				flatten(
					[
						(
							[
								[tup_el, *tup_el.tokenizer_elements()]
								for tup_el in el
								if isinstance(tup_el, _TokenizerElement)
							]
							if deep
							else filter(lambda x: isinstance(x, _TokenizerElement), el)
						)
						for el in self.__dict__.values()
						if isinstance(el, tuple)
					],
				),
			)
			non_tuple_elems.extend(tuple_elems)
			return non_tuple_elems

	def tokenizer_element_tree(self, depth: int = 0, abstract: bool = False) -> str:
		"""Returns a string representation of the tree of tokenizer elements contained in `self`.

		# Parameters
		- `depth: int`: Current depth in the tree. Used internally for recursion, no need to specify.
		- `abstract: bool`: Whether to print the name of the abstract base class or the concrete class for each `_TokenizerElement` instance.
		"""
		name: str = "\t" * depth + (
			type(self).__name__
			if not abstract
			else type(self)._level_one_subclass().__name__
		)
		return (
			name
			+ "\n"
			+ "".join(
				el.tokenizer_element_tree(depth + 1, abstract)
				for el in self.tokenizer_elements(deep=False)
			)
		)

	def tokenizer_element_dict(self) -> dict:
		"""Returns a dictionary representation of the tree of tokenizer elements contained in `self`."""
		return {
			type(self).__name__: {
				key: (
					val.tokenizer_element_dict()
					if isinstance(val, _TokenizerElement)
					else (
						val
						if not isinstance(val, tuple)
						else [
							(
								el.tokenizer_element_dict()
								if isinstance(el, _TokenizerElement)
								else el
							)
							for el in val
						]
					)
				)
				for key, val in self.__dict__.items()
				if key != "_type_"
			},
		}

	@classmethod
	@abc.abstractmethod
	def attribute_key(cls) -> str:
		"""Returns the binding used in `MazeTokenizerModular` for that type of `_TokenizerElement`."""
		raise NotImplementedError

	def to_tokens(self, *args, **kwargs) -> list[str]:
		"""Converts a maze element into a list of tokens.

		Not all `_TokenizerElement` subclasses produce tokens, so this is not an abstract method.
		Those subclasses which do produce tokens should override this method.
		"""
		raise NotImplementedError

	@abc.abstractmethod
	def is_valid(self, do_except: bool = False) -> bool:
		"""Returns if `self` contains data members capable of producing an overall valid `MazeTokenizerModular`.

		Some `_TokenizerElement` instances may be created which are not useful despite obeying data member type hints.
		`is_valid` allows for more precise detection of invalid `_TokenizerElement`s beyond type hinting alone.
		If type hints are sufficient to constrain the possible instances of some subclass, then this method may simply `return True` for that subclass.

		# Types of Invalidity
		In nontrivial implementations of this method, each conditional clause should contain a comment classifying the reason for invalidity and one of the types below.
		Invalidity types, in ascending order of invalidity:
		- Uninteresting: These tokenizers might be used to train functional models, but the schemes are not interesting to study.
		E.g., `_TokenizerElement`s which are strictly worse than some alternative.
		- Duplicate: These tokenizers have identical tokenization behavior as some other valid tokenizers.
		- Untrainable: Training functional models using these tokenizers would be (nearly) impossible.
		- Erroneous: These tokenizers might raise exceptions during use.

		# Development
		`is_invalid` is implemented to always return `True` in some abstract classes where all currently possible subclass instances are valid.
		When adding new subclasses or data members, the developer should check if any such blanket statement of validity still holds and update it as neccesary.

		## Nesting
		In general, when implementing this method, there is no need to recursively call `is_valid` on nested `_TokenizerElement`s contained in the class.
		In other words, failures of `is_valid` need not bubble up to the top of the nested `_TokenizerElement` tree.
		`MazeTokenizerModular.is_valid` calls `is_valid` on each of its `_TokenizerElement`s individually, so failure at any level will be detected.

		## Types of Invalidity
		If it's judged to be useful, the types of invalidity could be implemented with an Enum or similar rather than only living in comments.
		This could be used to create more or less stringent filters on the valid `_TokenizerElement` instances.
		"""
		raise NotImplementedError


T = TypeVar("T", bound=_TokenizerElement)


def _unsupported_is_invalid(self, do_except: bool = False) -> bool:  # noqa: ANN001
	"""Default implementation of `is_valid` for `mark_as_unsupported`-decorated classes"""
	if do_except:
		err_msg: str = (
			f"Class `{type(self).__name__ = }, marked as unsupported, is not valid."
			f"{type(self) = }, {self = }"
		)
		raise ValueError(err_msg)

	return False


# TYPING: better type hints for this function
def mark_as_unsupported(is_valid: Callable[[T, bool], bool]) -> Callable[[T], T]:
	"""mark a _TokenizerElement as unsupported.

	Classes marked with this decorator won't show up in `get_all_tokenizers()` and thus wont be tested.
	The classes marked in release 1.0.0 did work reliably before being marked, but they can't be instantiated since the decorator adds an abstract method.
	The decorator exists to prune the space of tokenizers returned by `all_instances` both for testing and usage.
	Previously, the space was too large, resulting in impractical runtimes.
	These decorators could be removed in future releases to expand the space of possible tokenizers.
	"""

	def wrapper(cls: T) -> T:
		# intentionally modifying method here
		# idk why it things `T`/`self` should not be an argument
		cls.is_valid = is_valid  # type: ignore[assignment, method-assign]
		return cls

	return wrapper


# TODO: why noqa here? `B024 `__TokenizerElementNamespace` is an abstract base class, but it has no abstract methods or properties`
class __TokenizerElementNamespace(abc.ABC):  # noqa: B024
	"""ABC for namespaces

	# Properties
	- key: The binding used in `MazeTokenizerModular` for instances of the classes contained within that `__TokenizerElementNamespace`.
	"""

	# HACK: this is not the right way of doing this lol
	key: str = NotImplementedError  # type: ignore[assignment]


def _load_tokenizer_element(
	data: dict[str, Any],
	namespace: type[__TokenizerElementNamespace],
) -> _TokenizerElement:
	"""Loads a `TokenizerElement` stored via zanj."""
	key: str = namespace.key
	format_: str = data[key][_FORMAT_KEY]
	cls_name: str = format_.split("(")[0]
	cls: type[_TokenizerElement] = getattr(namespace, cls_name)
	kwargs: dict[str, Any] = {
		k: load_item_recursive(data[key][k], tuple()) for k, v in data[key].items()
	}
	if _FORMAT_KEY in kwargs:
		kwargs.pop(_FORMAT_KEY)
	return cls(**kwargs)
