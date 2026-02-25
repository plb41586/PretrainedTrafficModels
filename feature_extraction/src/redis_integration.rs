use redis::{Client, Commands};
use serde::de;
use crate::feature_parser::{PayloadSet};

pub fn connect_to_redis() -> redis::Connection {
    let client = redis::Client::open("redis://falkordb:6379")
    .expect("Failed to create Redis client");
    
    client
        .get_connection()
        .expect("Failed to connect to Redis")
}

pub fn push_payloadset_to_redis_queue(conn: &mut redis::Connection, flow_identifier: &str, payloadset: &PayloadSet) {
    let _: () = conn.lpush(flow_identifier, payloadset).expect("Failed to push to Redis queue");
}

