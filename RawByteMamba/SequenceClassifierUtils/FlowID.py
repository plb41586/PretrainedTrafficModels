import polars as pl
import numpy as np
from typing import List
import ipaddress

def extract_indices_from_mask(data: np.ndarray, mask: np.ndarray):
    result = []
    for i in range(len(mask)):
        if mask[i]==1:
            result.append(data[i])
    return result

def add_ID_from_column(df: pl.DataFrame, columns: list[str], new_col_name: str) -> pl.DataFrame:
    return df.with_columns(
        pl.concat_list(columns).list.sort().hash().alias(new_col_name)
    )


def list_to_mac_address(byte_list: list[np.int64]) -> str:
    if len(byte_list) != 6:
        raise ValueError("Ethernet (MAC) addresses must contain exactly 6 bytes.")
    
    return ":".join(f"{byte:02X}" for byte in byte_list)

def list_to_ipv4_address(byte_list: List[int]) -> str:
    if len(byte_list) != 4:
        raise ValueError("IPv4 addresses must contain exactly 4 bytes.")
    
    if not all(0 <= byte <= 255 for byte in byte_list):
        raise ValueError("Each byte in an IPv4 address must be between 0 and 255.")
    
    return ".".join(str(byte) for byte in byte_list)


def bytes_to_ipv6_address(byte_list: List[int]) -> str:
    if len(byte_list) != 16:
        raise ValueError("IPv6 addresses must contain exactly 16 bytes.")
    
    if not all(0 <= byte <= 255 for byte in byte_list):
        raise ValueError("Each byte in an IPv6 address must be between 0 and 255.")
    
    # Convert bytes into 8 IPv6 segments (each segment is 2 bytes)
    segments = [(byte_list[i] << 8) | byte_list[i + 1] for i in range(0, 16, 2)]
    
    # Convert to IPv6 format
    ipv6_address = ipaddress.IPv6Address(":".join(f"{segment:X}" for segment in segments))
    
    return str(ipv6_address)  # Return properly formatted IPv6 string

def print_hex_dump(data: np.ndarray, bytes_per_line: int = 16):
    if not isinstance(data, np.ndarray) or data.dtype != np.uint8:
        raise ValueError("Input must be a NumPy array of dtype uint8.")
    
    for i in range(0, len(data), bytes_per_line):
        chunk = data[i:i + bytes_per_line]

        # Offset column
        offset = f"{i:08X}"
        
        # Hex bytes column
        hex_bytes = " ".join(f"{byte:02X}" for byte in chunk)
        hex_bytes = hex_bytes.ljust(bytes_per_line * 3)  # Padding for alignment
        
        # ASCII column
        ascii_rep = "".join(chr(byte) if 32 <= byte <= 126 else "." for byte in chunk)
        
        print(f"{offset}  {hex_bytes}  {ascii_rep}")

def get_addresses(row):
    packet_data = np.array(row[0])
    mask = np.array(row[1])
    extracted = extract_indices_from_mask(packet_data, mask)
    eth_sender = hash(np.array(extracted[:6]).tobytes())
    eth_receiver = hash(np.array(extracted[6:12]).tobytes())
    if len(extracted) == 44:
        ip_sender = hash(np.array(extracted[12:28]).tobytes())
        ip_receiver = hash(np.array(extracted[28:44]).tobytes())
    elif len(extracted) == 20:
        ip_sender = hash(np.array(extracted[12:16]).tobytes())
        ip_receiver = hash(np.array(extracted[16:20]).tobytes())
    else:
        address_list = [eth_sender, eth_receiver]
        address_list.sort()
        address_hash = hash(tuple(address_list))
        return address_hash
    address_list = [ip_sender, ip_receiver, eth_sender, eth_receiver]
    address_list.sort()
    address_hash = hash(tuple(address_list))
    return address_hash


def add_Flow_ID(data: pl.DataFrame) -> pl.DataFrame:
    return data.with_columns(pl.Series("FlowID", data.map_rows(get_addresses)))
