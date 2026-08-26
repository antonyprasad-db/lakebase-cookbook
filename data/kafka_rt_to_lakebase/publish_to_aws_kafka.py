import argparse
import json
import logging
import uuid

import boto3
from aws_msk_iam_sasl_signer import MSKAuthTokenProvider
from kafka import KafkaProducer
from kafka.net.sasl.oauth import AbstractTokenProvider

log = logging.getLogger(__name__)


class MSKTokenProvider(AbstractTokenProvider):
    def __init__(self, region: str, **config):
        super().__init__(**config)
        self.region = region

    def token(self):
        tok, _ = MSKAuthTokenProvider.generate_auth_token(self.region)
        return tok


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple utility to publish arbitrary messages to Kafka")
    parser.add_argument("cluster", type=str, help="AWS MSK cluster",)
    parser.add_argument("topic", type=str, help="Kafka topic to publish to")
    parser.add_argument("message", type=json.loads, help="JSON object to be sent over Kafka.")
    parser.add_argument("--number", "-n", type=int, default=1, help="Number of times to send the message. Useful when combined with --include-id")
    parser.add_argument("--include-id", action="store_true", help="If set, adds a top level 'id' field to the JSON message sent over Kafka.")
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("botocore").setLevel(logging.ERROR)
    logging.getLogger("kafka").setLevel(logging.ERROR)

    log.debug("Creating Kafka client using environment variable credentials and cluster %s", args.cluster)
    msk_client = boto3.client("kafka", region_name=args.region)
    bootstrap = msk_client.get_bootstrap_brokers(ClusterArn=args.cluster)["BootstrapBrokerStringPublicSaslIam"]
    log.debug("Got bootstrap providers: %s", bootstrap)
    producer = KafkaProducer(
        bootstrap_servers=bootstrap.split(","),
        security_protocol="SASL_SSL",
        sasl_mechanism="OAUTHBEARER",
        sasl_oauth_token_provider=MSKTokenProvider(args.region),
        value_serializer=lambda v: v.encode("utf-8"),
    )

    log.info("Sending message '%s' (id not included) %d times", args.message, args.number)
    for _ in range(0, args.number):
        if args.include_id:
            args.message["id"] = str(uuid.uuid4())
            log.debug("Set message id field to '%s'", args.message["id"])

        message = json.dumps(args.message)
        log.debug("Sending message on topic '%s': '%s'", args.topic, message)
        producer.send(args.topic, value=message)
        log.debug("Message sent on topic %s!", args.topic)


    log.info("Wrote %d messages to Kafka; flushing and closing", args.number)
    producer.flush()
    producer.close()

