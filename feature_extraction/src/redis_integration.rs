use redis::Commands;

use crate::feature_parser::{PayloadSet, ProtocolFeatureSet};

/// Connect to Redis. `host` is a `host:port` pair, e.g. `redis:6379`.
pub fn connect_to_redis(host: &str) -> redis::Connection {
    let url = format!("redis://{host}");
    let client = redis::Client::open(url.as_str())
        .unwrap_or_else(|e| panic!("Failed to create Redis client for {url:?}: {e}"));

    client
        .get_connection()
        .unwrap_or_else(|e| panic!("Failed to connect to Redis at {host}: {e}"))
}

pub fn push_payloadset_to_redis_queue(
    conn: &mut redis::Connection,
    flow_identifier: &str,
    payloadset: &PayloadSet,
) -> redis::RedisResult<()> {
    conn.lpush(flow_identifier, payloadset)
}

pub fn push_featureset_to_redis_queue(
    conn: &mut redis::Connection,
    flow_identifier: &str,
    feature_set: &ProtocolFeatureSet,
) -> redis::RedisResult<()> {
    let flow_feature_identifier = format!("{flow_identifier}features");
    conn.lpush(flow_feature_identifier, feature_set)
}
