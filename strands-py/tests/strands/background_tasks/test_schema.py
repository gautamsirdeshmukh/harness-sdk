"""Tests for background-task tool schema selection."""

from copy import deepcopy
from typing import cast

import pytest

from strands.background_tasks._schema import add_background_selection, strip_background_selection
from strands.types.tools import ToolSpec

_ABSENT = object()
_BACKGROUND_PROPERTY_SCHEMA = {
    "type": "boolean",
    "description": (
        "Run this tool call in the background. Acknowledgement is immediate; continue without waiting or polling. "
        "The final result will be delivered automatically at a later Agent boundary."
    ),
}


def _tool_spec(input_schema: object = _ABSENT) -> ToolSpec:
    tool_spec: dict[str, object] = {
        "name": "work",
        "description": "Perform work.",
    }
    if input_schema is not _ABSENT:
        tool_spec["inputSchema"] = input_schema
    return cast(ToolSpec, tool_spec)


def test_add_background_selection_supports_absent_schema() -> None:
    original = _tool_spec()

    tru_result = add_background_selection(original)
    exp_result = {
        "compatible": True,
        "tool_spec": {
            "name": "work",
            "description": "Perform work.",
            "inputSchema": {
                "type": "object",
                "properties": {"_background": _BACKGROUND_PROPERTY_SCHEMA},
            },
        },
    }

    assert tru_result == exp_result
    assert original == {"name": "work", "description": "Perform work."}
    assert tru_result["compatible"] is True
    assert tru_result["tool_spec"] is not original


def test_add_background_selection_supports_empty_schema() -> None:
    input_schema: dict[str, object] = {}
    original = _tool_spec(input_schema)

    tru_result = add_background_selection(original)
    exp_result = {
        "compatible": True,
        "tool_spec": {
            "name": "work",
            "description": "Perform work.",
            "inputSchema": {
                "type": "object",
                "properties": {"_background": _BACKGROUND_PROPERTY_SCHEMA},
            },
        },
    }

    assert tru_result == exp_result
    assert original["inputSchema"] == {}
    assert tru_result["compatible"] is True
    assert tru_result["tool_spec"]["inputSchema"] is not input_schema


def test_add_background_selection_supports_direct_object_schema_without_mutation() -> None:
    input_schema = {
        "type": "object",
        "properties": {
            "value": {
                "type": "object",
                "properties": {
                    "_background": {"type": "string"},
                    "nestedReference": {"$ref": "#/$defs/value"},
                },
            },
        },
        "required": ["value"],
        "additionalProperties": False,
    }
    original = _tool_spec(input_schema)
    exp_original = deepcopy(original)

    tru_result = add_background_selection(original)
    exp_result = {
        "compatible": True,
        "tool_spec": {
            "name": "work",
            "description": "Perform work.",
            "inputSchema": {
                **input_schema,
                "properties": {
                    **input_schema["properties"],
                    "_background": _BACKGROUND_PROPERTY_SCHEMA,
                },
            },
        },
    }

    assert tru_result == exp_result
    assert original == exp_original
    assert tru_result["compatible"] is True
    assert tru_result["tool_spec"]["inputSchema"] is not input_schema


def test_add_background_selection_supports_python_tool_schema_wrapper_without_mutation() -> None:
    input_schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    wrapped_schema = {"json": input_schema}
    original = _tool_spec(wrapped_schema)
    exp_original = deepcopy(original)

    tru_result = add_background_selection(original)
    exp_result = {
        "compatible": True,
        "tool_spec": {
            "name": "work",
            "description": "Perform work.",
            "inputSchema": {
                "json": {
                    **input_schema,
                    "properties": {
                        **input_schema["properties"],
                        "_background": _BACKGROUND_PROPERTY_SCHEMA,
                    },
                }
            },
        },
    }

    assert tru_result == exp_result
    assert original == exp_original
    assert tru_result["compatible"] is True
    assert tru_result["tool_spec"]["inputSchema"] is not wrapped_schema


@pytest.mark.parametrize(
    ("input_schema", "reason"),
    [
        ([], "input schema must be a direct object schema"),
        ({"type": "string"}, "root schema type must be 'object'"),
        ({"oneOf": [{"type": "object"}]}, "unsupported root schema keyword 'oneOf'"),
        ({"properties": []}, "root schema properties must be an object"),
        (
            {"additionalProperties": {"type": "string"}},
            "schema-valued additionalProperties is not supported",
        ),
        ({"required": "value"}, "root schema required must be an array"),
        (
            {"required": ["value", "value"]},
            "root schema required must contain unique property names",
        ),
        (
            {"required": ["value", 1]},
            "root schema required must contain unique property names",
        ),
        (
            {"properties": {"_background": {"type": "boolean"}}},
            "schema already defines reserved property '_background'",
        ),
        (
            {"required": ["_background"]},
            "schema requires reserved property '_background'",
        ),
    ],
)
def test_add_background_selection_rejects_incompatible_root_schema(input_schema: object, reason: str) -> None:
    original = _tool_spec(input_schema)
    exp_original = deepcopy(original)

    tru_result = add_background_selection(original)
    exp_result = {"compatible": False, "reason": reason}

    assert tru_result == exp_result
    assert original == exp_original


@pytest.mark.parametrize("selected", [True, False])
def test_strip_background_selection_removes_selector_from_copy(selected: bool) -> None:
    original = {"value": "x", "_background": selected}

    tru_result = strip_background_selection(original)
    exp_result = {"input": {"value": "x"}, "selected": selected}

    assert tru_result == exp_result
    assert original == {"value": "x", "_background": selected}
    assert tru_result["input"] is not original


def test_strip_background_selection_copies_dict_without_selector() -> None:
    original = {"value": "x"}

    tru_result = strip_background_selection(original)
    exp_result = {"input": {"value": "x"}}

    assert tru_result == exp_result
    assert tru_result["input"] is not original


@pytest.mark.parametrize("original", [None, "value", 1, ["value"]])
def test_strip_background_selection_preserves_non_dict_input(original: object) -> None:
    tru_result = strip_background_selection(original)
    exp_result = {"input": original}

    assert tru_result == exp_result
    assert tru_result["input"] is original


@pytest.mark.parametrize("selector", [None, "true", 1, 0, [], {}])
def test_strip_background_selection_rejects_malformed_selector(selector: object) -> None:
    with pytest.raises(TypeError, match="'_background' must be a boolean"):
        strip_background_selection({"value": "x", "_background": selector})
