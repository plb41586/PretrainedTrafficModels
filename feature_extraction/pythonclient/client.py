import redis
import msgpack
import pandas as pd
import polars as pl
import time

r = redis.Redis(host='feature-extractor', port=6379, db=0)

def deserialize_to_dataframe(data):
    # Create an empty list to store the row data
    rows = []
    
    for record in data:  # In case we have multiple records
        # Unpack the list into corresponding variables
        (
            arp_set, arp_features, ip_set, ip_features, tcp_set, tcp_features, 
            udp_set, udp_features, icmp_set, icmp_features, transport_protocol, 
            application_protocol, dns_features, mqtt_features, modbus_tcp_features, http_features
        ) = record
        
        # Create a dictionary mapping the features to the respective columns
        row = {
            'frame.time': None,  # No time provided in the deserialized data
            'ip_set': ip_set,
            'ip.src_host': ip_features[0],
            'ip.dst_host': ip_features[1],
            'arp_set': arp_set,
            'arp.dst.proto_ipv4': arp_features[0],
            'arp.opcode': arp_features[2],
            'arp.hw.size': arp_features[3],
            'arp.src.proto_ipv4': arp_features[1],
            'icmp.checksum': icmp_features[0],
            'icmp.seq_le': icmp_features[1],
            'icmp.transmit_timestamp': icmp_features[2],
            'icmp.unused': icmp_features[3],
            'http.file_data': http_features[0],
            'http.content_length': http_features[1],
            'http.request.uri.query': http_features[2],
            'http.request.method': http_features[3],
            'http.referer': http_features[4],
            'http.request.full_uri': http_features[5],
            'http.request.version': http_features[6],
            'http.response': http_features[7],
            'tcp_set': tcp_set,
            'tcp.ack': None,  # Calculated from TCP flags (ack bit set)
            'tcp.ack_raw': tcp_features[0],
            'tcp.checksum': f'0x{tcp_features[1]:08x}',
            'tcp.connection.fin': tcp_features[2],
            'tcp.connection.rst': tcp_features[3],
            'tcp.connection.syn': tcp_features[4],
            'tcp.connection.synack': tcp_features[5],
            'tcp.dstport': tcp_features[6],
            'tcp.flags': tcp_features[7],
            'tcp.len': tcp_features[8],
            'tcp.options': ''.join(format(byte, '02x') for byte in tcp_features[9]),
            'tcp.payload': ''.join(format(byte, '02x') for byte in tcp_features[10]),
            'tcp.seq': tcp_features[11],
            'tcp.srcport': tcp_features[12],
            'udp_set': udp_set,
            'udp.port': udp_features[1],  # src port
            'udp.stream': udp_features[2],  # stream id
            'udp.time_delta': udp_features[3],
            'dns.qry.name': dns_features[0],
            'dns.qry.name.len': dns_features[1],
            'dns.qry.qu': dns_features[2],
            'dns.qry.type': dns_features[3],
            'dns.retransmission': dns_features[4],
            'dns.retransmit_request_in': dns_features[5],
            'mqtt.conack.flags': mqtt_features[0],
            'mqtt.conflag.cleansess': mqtt_features[1],
            'mqtt.conflags': mqtt_features[2],
            'mqtt.hdrflags': mqtt_features[3],
            'mqtt.len': mqtt_features[4],
            'mqtt.msg_decoded_as': mqtt_features[5],
            'mqtt.msg': mqtt_features[6],
            'mqtt.msgtype': mqtt_features[7],
            'mqtt.proto_len': mqtt_features[8],
            'mqtt.protoname': mqtt_features[9],
            'mqtt.topic': mqtt_features[10],
            'mqtt.topic_len': mqtt_features[11],
            'mqtt.ver': mqtt_features[12],
            'mbtcp.len': modbus_tcp_features[0],
            'mbtcp.trans_id': modbus_tcp_features[1],
            'mbtcp.unit_id': modbus_tcp_features[2]
        }
        
        # Append the row to the list of rows
        rows.append(row)
    
    # Create a DataFrame from the rows
    df = pd.DataFrame(rows)
    
    return df

# for i in range(100):
#     message = r.rpop("feature_queue", 1)
#     # print("raw: ", message)
#     # Deserialize the data using msgpack
#     deserialized_data = msgpack.unpackb(message[0], raw=False)

#     # Now `deserialized_data` should contain the original CombinedFeatures structure
#     # print("deserialized: ", deserialized_data)
    
#     new_row = deserialize_to_dataframe([deserialized_data])
#     # new_row = pl.from_pandas(new_row)

#     try:
#         # df.extend(new_row)
#         df = df.append(new_row)
#     except:
#         df = new_row
#     print("Received message: ", i)

# df.write_csv("/output/rec_features.csv")

# Initialize the DataFrame
df = None

# Timeout in seconds
TIMEOUT = 2
last_received_time = time.time()

# Infinite loop until no new data is received for TIMEOUT seconds
i = 0
batch_sz = 100
deserialized_messages = []
while True:
    # Try to fetch a message from the Redis queue
    messages = r.rpop("feature_queue", batch_sz)
    if messages is not None:
        for message in messages:
            if message:
                # Deserialize the data using msgpack
                deserialized_data = msgpack.unpackb(message, raw=False)
                deserialized_messages.append(deserialized_data)
                
                # # Deserialize to DataFrame and convert to Polars DataFrame
                # new_row = deserialize_to_dataframe([deserialized_data])
                # # new_row = pl.from_pandas(new_row)

                # Update the last received time
                last_received_time = time.time()
                i += 1
                print(f"Received message: {i}")

            # # Append the new row to the DataFrame
            # if df is None:
            #     df = pd.DataFrame(deserialized_messages)
            # else:
            #     df = pd.concat([df, pd.DataFrame(deserialized_messages)], ignore_index=True)

    else:
        # Check if TIMEOUT seconds have passed since the last received message
        if time.time() - last_received_time > TIMEOUT:
            print("No new messages received for 2 seconds. Exiting.")
            break

df = deserialize_to_dataframe(deserialized_messages)

FileName = "Water_Level"

# Write the DataFrame to CSV
if df is not None:
    df.to_csv(f"/output/{FileName}.csv")
    print(f"Data saved to /output/{FileName}.csv")
else:
    print("No data to save.")