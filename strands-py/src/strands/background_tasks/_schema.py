"""Internal schema helpers for model-selected background execution."""

from copy import deepcopy
from typing import Any, Literal, cast

from typing_extensions import NotRequired, TypedDict

from ..types.tools import JSONSchema, ToolSpec

_BACKGROUND_PROPERTY = "_background"
_BACKGROUND_PROPERTY_DESCRIPTION = (
    "Run this tool call in the background. Acknowledgement is immediate; continue without waiting or polling. "
    "The final result will be delivered automatically at a later Agent boundary."
)
_ALLOWED_ROOT_KEYS = frozenset(
    {
        "$id",
        "$schema",
        "title",
        "description",
        "default",
        "examples",
        "type",
        "properties",
        "required",
        "additionalProperties",
    }
)


class _CompatibleBackgroundSchema(TypedDict):
    compatible: Literal[True]
    tool_spec: ToolSpec


class _IncompatibleBackgroundSchema(TypedDict):
    compatible: Literal[False]
    reason: str


class _StrippedBackgroundSelection(TypedDict):
    input: Any
    selected: NotRequired[bool]


_BackgroundSchemaResult = _CompatibleBackgroundSchema | _IncompatibleBackgroundSchema


def add_background_selection(tool_spec: ToolSpec) -> _BackgroundSchemaResult:
    """Add the model-controlled background selector to a compatible tool schema.

    Args:
        tool_spec: Tool specification to copy and extend.

    Returns:
        A copied tool specification when compatible, or an incompatibility reason.
    """
    try:
        copied = cast(dict[str, object], deepcopy(tool_spec))
    except Exception as error:
        return {"compatible": False, "reason": str(error)}

    if "inputSchema" not in copied:
        copied["inputSchema"] = _object_schema_with_background({})
        return {"compatible": True, "tool_spec": cast(ToolSpec, copied)}

    input_schema = copied["inputSchema"]
    if not isinstance(input_schema, dict):
        return {"compatible": False, "reason": "input schema must be a direct object schema"}

    wrapped = set(input_schema) == {"json"}
    schema = input_schema["json"] if wrapped else input_schema
    if not isinstance(schema, dict):
        return {"compatible": False, "reason": "input schema must be a direct object schema"}

    for key in schema:
        if key not in _ALLOWED_ROOT_KEYS:
            return {"compatible": False, "reason": f"unsupported root schema keyword '{key}'"}

    if "type" in schema and schema["type"] != "object":
        return {"compatible": False, "reason": "root schema type must be 'object'"}

    properties = schema.get("properties")
    if "properties" in schema and not isinstance(properties, dict):
        return {"compatible": False, "reason": "root schema properties must be an object"}

    if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], bool):
        return {"compatible": False, "reason": "schema-valued additionalProperties is not supported"}

    required = schema.get("required")
    if "required" in schema:
        if not isinstance(required, list):
            return {"compatible": False, "reason": "root schema required must be an array"}

        seen_properties: set[str] = set()
        for property_name in required:
            if not isinstance(property_name, str) or property_name in seen_properties:
                return {
                    "compatible": False,
                    "reason": "root schema required must contain unique property names",
                }
            seen_properties.add(property_name)

    if isinstance(properties, dict) and _BACKGROUND_PROPERTY in properties:
        return {
            "compatible": False,
            "reason": f"schema already defines reserved property '{_BACKGROUND_PROPERTY}'",
        }

    if isinstance(required, list) and _BACKGROUND_PROPERTY in required:
        return {
            "compatible": False,
            "reason": f"schema requires reserved property '{_BACKGROUND_PROPERTY}'",
        }

    transformed_schema = _object_schema_with_background(schema)
    copied["inputSchema"] = {"json": transformed_schema} if wrapped else transformed_schema
    return {"compatible": True, "tool_spec": cast(ToolSpec, copied)}


def strip_background_selection(tool_input: Any) -> _StrippedBackgroundSelection:
    """Remove the background selector from a copied tool input.

    Args:
        tool_input: Input generated for a tool call.

    Returns:
        The copied input and the selected mode when one was provided.

    Raises:
        TypeError: If the selector is present but is not a boolean.
    """
    if not isinstance(tool_input, dict):
        return {"input": tool_input}

    copied = tool_input.copy()
    if _BACKGROUND_PROPERTY not in copied:
        return {"input": copied}

    selected = copied.pop(_BACKGROUND_PROPERTY)
    if not isinstance(selected, bool):
        raise TypeError(f"'{_BACKGROUND_PROPERTY}' must be a boolean")

    return {"input": copied, "selected": selected}


def _object_schema_with_background(schema: JSONSchema) -> JSONSchema:
    properties = schema.get("properties", {})
    return {
        **schema,
        "type": "object",
        "properties": {
            **properties,
            _BACKGROUND_PROPERTY: {
                "type": "boolean",
                "description": _BACKGROUND_PROPERTY_DESCRIPTION,
            },
        },
    }
