"""
Unit tests for DataSchema class.

Tests focus on initialization and validation of the DataSchema class.
"""

import pytest

from slaf.distributed.processor import DataSchema


class TestDataSchema:
    """Test cases for DataSchema initialization and validation."""

    def test_data_schema_initialization(self):
        """Test that all required fields are set."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )

        assert schema.group_key == "group_id"
        assert schema.item_key == "item_id"
        assert schema.value_key == "value"
        assert schema.group_key_out is None  # Defaults to None
        assert schema.item_list_key == "item_list"  # Default value
        assert schema.value_list_key is None  # Defaults to None

    def test_data_schema_no_defaults(self):
        """Verify no defaults for required fields (forces explicit specification)."""
        # This test ensures that required fields must be explicitly provided
        # If we try to create without required fields, it should fail
        with pytest.raises(TypeError):
            DataSchema()  # Missing required arguments

    def test_data_schema_optional_fields(self):
        """Test optional fields (item_list_key, value_list_key, group_key_out)."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
            group_key_out="output_group_id",
            item_list_key="custom_item_list",
            value_list_key="custom_value_list",
        )

        assert schema.group_key_out == "output_group_id"
        assert schema.item_list_key == "custom_item_list"
        assert schema.value_list_key == "custom_value_list"

    def test_data_schema_default_item_list_key(self):
        """Test that item_list_key defaults to 'item_list'."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )

        assert schema.item_list_key == "item_list"

    def test_data_schema_group_key_out_defaults_to_none(self):
        """Test that group_key_out defaults to None."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )

        assert schema.group_key_out is None

    def test_data_schema_value_list_key_defaults_to_none(self):
        """Test that value_list_key defaults to None."""
        schema = DataSchema(
            group_key="group_id",
            item_key="item_id",
            value_key="value",
        )

        assert schema.value_list_key is None
