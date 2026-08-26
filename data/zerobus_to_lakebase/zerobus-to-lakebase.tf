terraform {
  required_providers {
    databricks = {
      source = "databricks/databricks"
    }
  }
}

# Uses the same ~/.databrickscfg profile as the CLI.
provider "databricks" {
  profile = "YOUR PROFILE HERE"
}

variable "catalog" {
  default = "YOUR CATALOG HERE"
}

variable "schema" {
  default = "YOUR SCHEMA HERE"
}

#############
# Resources #
#############

# Zerobus only allows service principals to publish to it, so for testing purposes we create one for you.
resource "databricks_service_principal" "zerobus_ingest" {
  display_name = "zerobus-principal"
}

resource "databricks_sql_table" "zerobus_target" {
  catalog_name = var.catalog
  schema_name  = var.schema
  name         = "zerobus_target"
  table_type   = "MANAGED"

  column {
    name = "id"
    type = "string"
  }
  column {
    name = "value"
    type = "string"
  }

  properties = {
    "delta.enableChangeDataFeed"  = "true"
    "delta.enableDeletionVectors" = "true"
    "delta.enableRowTracking"     = "true"
  }
}

# Grant the service principal read/write access
resource "databricks_grant" "sp_source_table" {
  table      = "${var.catalog}.${var.schema}.${databricks_sql_table.zerobus_target.name}"
  principal  = databricks_service_principal.zerobus_ingest.application_id
  privileges = ["SELECT", "MODIFY"]
}

# Lakebase setup
resource "databricks_postgres_project" "this" {
  project_id = "zerobus-demo"
  spec = {
    pg_version   = 17
    display_name = "zerobus-demo"
    default_endpoint_settings = {
      autoscaling_limit_min_cu = 0.5
      autoscaling_limit_max_cu = 2.0
      suspend_timeout_duration = "300s"
    }
  }
}

resource "databricks_postgres_synced_table" "zerobus_target_synced" {
  synced_table_id = "${var.catalog}.${var.schema}.zerobus_test_synced"
  spec = {
    source_table_full_name             = "${var.catalog}.${var.schema}.${databricks_sql_table.zerobus_target.name}"
    primary_key_columns                = ["id"]
    scheduling_policy                  = "CONTINUOUS"
    postgres_database                  = "databricks_postgres"
    branch                             = "production"
    create_database_objects_if_missing = true
    new_pipeline_spec = {
      storage_catalog = var.catalog
      storage_schema  = var.schema
    }
  }
}
