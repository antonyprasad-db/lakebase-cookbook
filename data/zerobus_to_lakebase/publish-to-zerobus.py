import argparse
import json
import logging
import os
import sys
import uuid

from databricks.sdk.config import Config as DatabricksConfig
from zerobus.sdk.shared import RecordType, StreamConfigurationOptions, TableProperties
from zerobus.sdk.sync import ZerobusSdk

log = logging.getLogger(__name__)


def get_zerobus_endpoint(dbx_config: DatabricksConfig, region: str) -> str:
    """Derive the Zerobus endpoint from the workspace URL. These follow a standard naming pattern, so this replacement
    is very mechanical"""
    if dbx_config.is_aws:
        return f"https://{dbx_config.workspace_id}.zerobus.{region}.cloud.databricks.com"
    elif dbx_config.is_gcp:
        return f"https://{dbx_config.workspace_id}.zerobus.{region}.gcp.databricks.com"
    elif dbx_config.is_azure:
        return f"https://{dbx_config.workspace_id}.zerobus.{region}.azuredatabricks.net"
    else:
        raise ValueError(f"Workspace {dbx_config.workspace_id} ({dbx_config.host}) is not identified as an AWS, Azure, or GCP workspace")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple utility to publish arbitrary messages to Kafka")
    parser.add_argument("databricks_profile", type=str, help="Databricks CLI profile to use")
    parser.add_argument("cloud_region", type=str, help="Cloud region that your workspace belongs to")
    parser.add_argument("client_id", type=str, help="Service principal client ID.")
    parser.add_argument("fqtn", type=str, help="Fully qualified table came, catalog.schema.table")
    parser.add_argument("message", type=json.loads, help="JSON object to be sent over Kafka.")
    parser.add_argument("--number", "-n", type=int, default=1, help="Number of times to send the message. Useful when combined with --include-id")
    parser.add_argument("--include-id", action="store_true", help="If set, adds a top level 'id' field to the JSON message sent over Kafka.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    if "CLIENT_SECRET" not in os.environ:
        print("CLIENT_SECRET environment variable missing, this is required for Zerobus authentication", file=sys.stderr)
        exit(1)
    client_secret = os.environ["CLIENT_SECRET"]

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # Workspace config
    log.debug("Configuring Databricks with profile '%s'", args.databricks_profile)
    try:
        dbx_config = DatabricksConfig(profile=args.databricks_profile)
    except ValueError:
        log.exception("Encountered error loading config from profile, profile '%s' likely does not exist. Check stack trace for more info.", args.databricks_profile)
        exit(1)
    log.debug("Found workspace URL '%s' and ID '%s'", dbx_config.host, dbx_config.workspace_id)

    # Authentication and stream creation -- we're using the Databricks CLI's auth and stealing auth headers off it.
    auth_headers = dbx_config.authenticate()

    zerobus_endpoint = get_zerobus_endpoint(dbx_config, args.cloud_region)
    log.debug("Got zerobus endpoint '%s'", zerobus_endpoint)
    sdk = ZerobusSdk(zerobus_endpoint, dbx_config.host)

    stream = sdk.create_stream(
        client_id=args.client_id,
        client_secret=client_secret,
        table_properties=TableProperties(args.fqtn),
        options = StreamConfigurationOptions(record_type=RecordType.JSON),
    )
    log.debug("Started stream %s", stream.stream_id)

    log.info("Sending message '%s' (id not included) %d times", args.message, args.number)
    for _ in range(0, args.number):
        if args.include_id:
            args.message["id"] = str(uuid.uuid4())
            log.debug("Set message id field to '%s'", args.message["id"])

        message = json.dumps(args.message)
        ack = stream.ingest_record(message)

    log.info("Wrote %d messages to Zerobus; closing stream", args.number)
    stream.close()

