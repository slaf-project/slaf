# SLAF Docstring Template

This document defines the standard docstring format for all SLAF classes and methods.

## Class Docstring Template

```python
class ClassName:
    """
    Brief description of what the class does.

    Longer description explaining the purpose, key features, and use cases.
    Include important design decisions or architectural notes.

    Examples:
        >>> # Basic instantiation
        >>> obj = ClassName(param1="value1", param2="value2")
        >>> print(f"Object created: {obj}")
        Object created: <ClassName object>

        >>> # Common usage pattern
        >>> obj = ClassName(param1="value1")
        >>> result = obj.method1(param2="value2")
        >>> print(f"Result: {result}")
        Result: <expected output>

        >>> # Advanced usage with multiple methods
        >>> obj = ClassName(param1="value1")
        >>> obj.configure(option1=True, option2=False)
        >>> results = obj.process_data()
        >>> print(f"Processed {len(results)} items")
        Processed 100 items
    """
```

## `__init__` Method Template

```python
def __init__(self, param1: type, param2: type = default_value):
    """
    Initialize the class instance.

    Args:
        param1: Description of the first parameter. Include any constraints,
                expected format, or important notes about the parameter.
        param2: Description of the second parameter with default value.
                Explain when to use the default vs custom values.

    Raises:
        ValueError: When parameters are invalid or missing required data.
        FileNotFoundError: When required files don't exist.
        TypeError: When parameter types are incorrect.

    Examples:
        >>> # Basic instantiation
        >>> obj = ClassName("path/to/data")
        >>> print(f"Initialized with shape: {obj.shape}")
        Initialized with shape: (1000, 20000)

        >>> # With custom parameters
        >>> obj = ClassName("path/to/data", param2="custom_value")
        >>> print(f"Custom config: {obj.param2}")
        Custom config: custom_value

        >>> # Error handling
        >>> try:
        ...     obj = ClassName("nonexistent/path")
        ... except FileNotFoundError as e:
        ...     print(f"Error: {e}")
        Error: SLAF config not found at nonexistent/path/config.json
    """
```

## Method Docstring Template

```python
def method_name(self, param1: type, param2: type = default_value) -> ReturnType:
    """
    Brief description of what the method does.

    Longer description explaining the method's purpose, behavior, and any
    important implementation details or side effects.

    Args:
        param1: Description of the first parameter. Include constraints,
                expected format, or important notes.
        param2: Description of the second parameter with default value.
                Explain when to use default vs custom values.

    Returns:
        Description of the return value, including its type and structure.
        If the method returns multiple types, explain when each occurs.

    Raises:
        ValueError: When parameters are invalid or data is malformed.
        KeyError: When required keys are missing.
        RuntimeError: When the operation cannot be completed.

    Examples:
        >>> # Basic usage
        >>> obj = ClassName("path/to/data")
        >>> result = obj.method_name("value1")
        >>> print(f"Result: {result}")
        Result: <expected output>

        >>> # With multiple parameters
        >>> result = obj.method_name("value1", param2="custom_value")
        >>> print(f"Custom result: {result}")
        Custom result: <expected output>

        >>> # Error handling
        >>> try:
        ...     result = obj.method_name("invalid_value")
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        Error: Invalid value provided

        >>> # Chaining methods
        >>> obj = ClassName("path/to/data")
        >>> filtered = obj.filter_cells(cell_type="T cells")
        >>> subset = filtered.query("SELECT * FROM cells LIMIT 10")
        >>> print(f"Filtered subset: {len(subset)} cells")
        Filtered subset: 10 cells
    """
```

## Property Docstring Template

```python
@property
def property_name(self) -> ReturnType:
    """
    Brief description of the property.

    Longer description explaining what the property represents,
    when it's computed, and any important notes about its behavior.

    Returns:
        Description of the property value and its type.

    Examples:
        >>> # Accessing the property
        >>> obj = ClassName("path/to/data")
        >>> print(f"Shape: {obj.shape}")
        Shape: (1000, 20000)

        >>> # Property behavior
        >>> obj = ClassName("path/to/data")
        >>> print(f"Initial shape: {obj.shape}")
        Initial shape: (1000, 20000)
        >>> # After filtering
        >>> obj.filter_cells(cell_type="T cells")
        >>> print(f"After filtering: {obj.shape}")
        After filtering: (250, 20000)
    """
```

## Module-Level Function Template

```python
def function_name(param1: type, param2: type = default_value) -> ReturnType:
    """
    Brief description of what the function does.

    Longer description explaining the function's purpose, behavior,
    and any important implementation details.

    Args:
        param1: Description of the first parameter.
        param2: Description of the second parameter with default value.

    Returns:
        Description of the return value and its type.

    Raises:
        ValueError: When parameters are invalid.
        FileNotFoundError: When required files don't exist.

    Examples:
        >>> # Basic usage
        >>> result = function_name("input_data")
        >>> print(f"Result: {result}")
        Result: <expected output>

        >>> # With custom parameters
        >>> result = function_name("input_data", param2="custom_value")
        >>> print(f"Custom result: {result}")
        Custom result: <expected output>

        >>> # Error handling
        >>> try:
        ...     result = function_name("invalid_input")
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        Error: Invalid input provided
    """
```

## Guidelines for Examples

### 1. Instantiation Examples

- Always show the basic instantiation pattern
- Include common parameter combinations
- Show error handling for invalid inputs
- Demonstrate different ways to create the object

### 2. Method Usage Examples

- Show the object being instantiated first
- Demonstrate the method call with realistic parameters
- Include chaining of multiple methods
- Show error handling for invalid method calls

### 3. Realistic Data

- Use realistic parameter values
- Show expected output formats
- Include common use cases and edge cases
- Demonstrate the full workflow from instantiation to result

### 4. Error Handling

- Show common error scenarios
- Demonstrate proper exception handling
- Include validation examples
- Show how to debug common issues

## Google Style Docstring Format

All docstrings should follow Google style format:

```python
def function_name(param1: str, param2: int = 10) -> bool:
    """
    Brief description.

    Longer description with more details about the function's behavior,
    implementation details, and important notes.

    Args:
        param1: Description of the first parameter.
        param2: Description of the second parameter with default value.

    Returns:
        Description of the return value.

    Raises:
        ValueError: When parameters are invalid.
        FileNotFoundError: When required files don't exist.

    Examples:
        >>> # Example usage
        >>> result = function_name("test", 5)
        >>> print(result)
        True
    """
```

## Implementation Checklist

When updating docstrings, ensure:

- [ ] Class docstrings include purpose, features, and examples
- [ ] `__init__` methods document all parameters with types
- [ ] All methods include Args, Returns, Raises sections
- [ ] Examples show instantiation + method usage
- [ ] Error handling examples are included
- [ ] Realistic parameter values and expected outputs
- [ ] Google style format is followed consistently
- [ ] Type hints are included for all parameters and returns
