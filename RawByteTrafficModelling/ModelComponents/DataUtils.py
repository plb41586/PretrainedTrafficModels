import polars as pl
import numpy as np
import torch
import concurrent.futures
import math

class ID_Encoder:
    def __init__(self, SpecialIDs: dict, CLS_Placement: str):
        self.SpecialIDs = SpecialIDs
        self.CLS_Placement = CLS_Placement # "SOS" or "EOS"



    def _construct_input_ids_SOS(self, data: list) -> np.ndarray:
        """
        Construct the input IDs for the given data. 
        The CLS token is placed at the beginning of the sequence.
        
        Args:
            data (list): A list of NumPy arrays containing the 'data' values.
        
        Returns:
            np.ndarray: A NumPy array containing the input IDs.
        """
        input_ids = np.ones((len(data), 1520), dtype=np.int32)
        for i, d in enumerate(data):
            input_ids[i, :] = self.SpecialIDs["<pad>"]
            input_ids[i, 0] = self.SpecialIDs["<CLS>"]
            input_ids[i, 1:len(d)+1] = d
            input_ids[i, len(d)+1] = self.SpecialIDs["</s>"]
        return input_ids
    
    def _construct_input_ids_EOS(self, data: list) -> np.ndarray:
        """
        Construct the input IDs for the given data.
        The CLS token is placed at the end of the sequence before the padding.

        Args:
            data (list): A list of NumPy arrays containing the 'data' values.

        Returns:
            np.ndarray: A NumPy array containing the input IDs.
        """
        input_ids = np.ones((len(data), 1520), dtype=np.int32)
        for i, d in enumerate(data):
            input_ids[i, :] = self.SpecialIDs["<pad>"]
            input_ids[i, 0:len(d)] = d
            input_ids[i, len(d)] = self.SpecialIDs["<CLS>"]
        return input_ids

    def construct_input_ids(self, data: list) -> np.ndarray:
        """
        Construct the input IDs for the given data.
        The CLS token is placed according to the CLS_Placement attribute.

        Args:
            data (list): A list of NumPy arrays containing the 'data' values.

        Returns:
            np.ndarray: A NumPy array containing the input IDs.
        """
        if self.CLS_Placement == "SOS":
            return self._construct_input_ids_SOS(data)
        elif self.CLS_Placement == "EOS":
            return self._construct_input_ids_EOS(data)
        else:
            raise ValueError(f"Invalid CLS placement: {self.CLS_Placement}")

class TrainingDatasetHandler():
    def __init__(self, data: pl.DataFrame, seq_len: int, encoder: ID_Encoder):
        self.data = data
        self.attack_dfs = self.split_by_label(data)
        self.seq_len = seq_len # Number of packets in a sequence
        self.unique_labels = data["AttackLabel"].value_counts()
        self.InputIDEncoder = encoder

    def split_by_label(self, data: pl.DataFrame):
        data = data.with_row_index()
        #Split the training data by AttackLabel
        # Assuming your main dataframe is called 'df'
        # Get unique attack labels
        unique_labels = data['AttackLabel'].unique().to_list()

        # Create a dictionary to store the split dataframes
        attack_dfs = []

        # Split the dataframe for each unique attack label
        for label in unique_labels:
            attack_dfs.append(data.filter(pl.col('AttackLabel') == label))
        
        return attack_dfs

    def get_packet_sequence_from_df(self, df: pl.DataFrame, seq_len: int):
        """
        Get a sequence of packets from the given DataFrame.
        Start index is randomly selected in the range [0, len(df) - seq_len].
        
        Args:
            df (pl.DataFrame): The DataFrame containing the packet data.
            seq_len (int): The sequence length.
        
        Returns:
            pl.Dataframe: The packet sequence.
        """
        # Check if DataFrame has enough rows for the sequence length
        if len(df) < seq_len:
            # Extract the sequence unpadded sequence
            packet_sequence = df

        else:
            length = seq_len
            # Generate random start index
            max_start_idx = len(df) - length
            start_idx = np.random.randint(0, max_start_idx + 1)
            # Extract the sequence
            packet_sequence = df.slice(start_idx, length)
        
        return packet_sequence
    
    def even_sample(self, batch_size: int):
        """
        Evenly sample a batch of packet sequences from the attack DataFrames.
        Results in a batch with balanced labels.
        input:
            batch_size: The batch size.
        output:
            batch_data: A list of packet sequences in polars DataFrame format.
        """
        label_indices = np.random.choice(len(self.attack_dfs), batch_size)
        batch_data = []
        for i in label_indices:
            df = self.attack_dfs[i]
            batch_data.append(self.get_packet_sequence_from_df(df, self.seq_len))
        return batch_data

    def sample_flow_even_label(self, batch_size: int):
        """
        Evenly sample a batch of packet sequences from the attack DataFrames.
        Results in a batch with balanced labels.
        input:
            batch_size: The batch size.
        output:
            batch_data: A list of packet sequences in polars DataFrame format.
        """
        label_indices = np.random.choice(len(self.attack_dfs), batch_size)

        def process_index(i):
            df = self.attack_dfs[i]
            flow_pick = df.select(pl.col("FlowID").sample(n=1, with_replacement=False)).item()
            df = df.filter(pl.col("FlowID") == flow_pick)
            return self.get_packet_sequence_from_df(df, self.seq_len)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            batch_data = list(executor.map(process_index, label_indices))

        return batch_data

    def sample_epoch_packet_indices(self, batch_size: int):
        """
        Sample batches of packets from the training data randomly.
        Returns batches of indices to draw from training data to complete one epoch.
        Batches affected by class imbalance.
        input:
            batch_size: The batch size.
        output:
            batch_data: A list of batch indices in numpy array format intended to draw from training data.
        """
        # Get total number of samples
        num_samples = self.data.height
        
        # Generate and shuffle indices
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        # Split indices into batches
        batch_indices = [
            indices[i:i + batch_size] 
            for i in range(0, num_samples, batch_size)
        ]
        
        return batch_indices

    def get_pretraining_data(self, indices: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Retrieve the 'data' and 'mask' columns as NumPy arrays for the given indices.
        Apply mask to data.
        Return the masked bytes and the proto hierarchy values.

        Args:
            indices (np.ndarray): The indices to retrieve data for.
        Returns:
            list: A list of NumPy arrays containing the masked 'data' values.
            list: A list of proto hierarchy values.
        """
        selected_data = self.data["data"][indices].to_numpy()
        selected_masks = self.data["mask"][indices].to_numpy()
        selected_proto_hierarchy = self.data["proto_hierarchy"][indices].to_numpy()

        masked_bytes = self.apply_mask(selected_data, selected_masks)

        return masked_bytes, selected_proto_hierarchy

    def apply_mask(self, bytes, masks):
        """
        Apply the mask to the bytes.
        
        Args:
            bytes (list): A list of NumPy arrays containing the 'data' values.
            masks (list): A list of NumPy arrays containing the 'mask' values.
        
        Returns:
            list: A list of NumPy arrays containing the masked 'data' values.
        """
        # Invert array_two (swap 0s and 1s)
        masks_inverted = [1 - sub_array for sub_array in masks]

        # Perform element-wise multiplication
        masked_bytes = [a * b for a, b in zip(bytes, masks_inverted)]
        
        return masked_bytes

    def get_bytes_as_numpy(self, df: pl.DataFrame) -> tuple[np.ndarray, str]:
        """
        Retrieve the 'data' column as a NumPy array for the entire DataFrame.
        Retrieve the 'mask' column and apply it to data.
        And return the label assuming all rows have the same label.
        
        Args:
            df (pl.DataFrame): The Polars DataFrame containing the 'data' column.
        
        Returns:
            list: A list of NumPy arrays containing all 'data' values.
            label: The label of the data as string
        """
        selected = df["data"].to_numpy()
        masks = df["mask"].to_numpy()

        masked_bytes = self.apply_mask(selected, masks)

        label = df["AttackLabel"][0]
        return masked_bytes, label
    
    def draw_encoded_batch(self, batch_size: int) -> tuple[torch.Tensor, list[str]]:
        """
        Get a batch of packet sequences.
        
        Args:
            batch_size (int): The batch size.
        
        Returns:
            torch.Tensor: The batch of packet sequences.
        """
        batch_data = []
        batch_labels = []
        batch_dfs = self.even_sample(batch_size)
        for seq_df in batch_dfs:
            bytes, label = self.get_bytes_as_numpy(seq_df)
            InputIDs = self.InputIDEncoder.construct_input_ids(bytes)
            InputIDs = torch.tensor(InputIDs)
            batch_data.append(InputIDs)
            batch_labels.append(label)
        return torch.stack(batch_data), batch_labels

    def pad_sequence_IDs(self, sequence: np.ndarray) -> np.ndarray:
        """
        Pad the given sequence of packets with the full byte sequences consisting of the padding token.
        
        Args:
            sequence (np.ndarray): The sequence to pad.
        
        Returns:
            np.ndarray: The padded sequence.
        """
        sequence_length = sequence.shape[0]
        padding_length = (self.seq_len - sequence_length) + 1
        padded_packets = np.ones((padding_length, 1520), dtype=np.int32) * self.InputIDEncoder.SpecialIDs["<pad>"]
        padded_sequence = np.concatenate((sequence, padded_packets), axis=0)
        return padded_sequence, sequence_length

    def draw_encoded_flow_batch(self, batch_size: int) -> tuple[torch.Tensor, list[str]]:
        """
        Get a batch of packet sequences from flow within dataset.
        Labels are evenly sampled. Flows shorter
        
        Args:
            batch_size (int): The batch size.
        
        Returns:
            torch.Tensor: The batch of packet sequences.
            list[str]: The labels of the packet sequences.
            torch.Tensor: The sequence lengths of the packet sequences.
        """
        batch_data = []
        batch_labels = []
        sequence_lengths = np.zeros(batch_size, dtype=np.int32)
        batch_dfs = self.sample_flow_even_label(batch_size)
        for seq_index, seq_df in enumerate(batch_dfs):
            bytes, label = self.get_bytes_as_numpy(seq_df)
            InputIDs = self.InputIDEncoder.construct_input_ids(bytes)
            InputIDs, _sequence_length = self.pad_sequence_IDs(InputIDs) ## Sequence_length usefull if CLS Classification for sequences is implemented
            InputIDs = torch.tensor(InputIDs)
            batch_data.append(InputIDs)
            batch_labels.append(label)
            sequence_lengths[seq_index] = _sequence_length
        return torch.stack(batch_data), batch_labels, torch.tensor(sequence_lengths)

class ValidationDatasetHandler():
    def __init__(self, data: pl.DataFrame, seq_len: int, encoder: ID_Encoder, batch_size: int):
        self.data = data
        self.seq_len = seq_len # Number of packets in a sequence
        self.InputIDEncoder = encoder
        self.SeqDfs = []
        self.fill_seq_dfs()
        self.batch_size = batch_size
        self.batches = math.ceil(len(self.SeqDfs) / batch_size)
        self.current_batch = 0
        self.OnGoing = True

    def split_by_label(self, data: pl.DataFrame):
        data = data.with_row_index()
        #Split the training data by AttackLabel
        # Assuming your main dataframe is called 'df'
        # Get unique attack labels
        unique_labels = data['AttackLabel'].unique().to_list()

        # Create a dictionary to store the split dataframes
        attack_dfs = []

        # Split the dataframe for each unique attack label
        for label in unique_labels:
            attack_dfs.append(data.filter(pl.col('AttackLabel') == label))
        
        return attack_dfs

    def get_packet_sequence_from_df(self, df: pl.DataFrame, seq_len: int):
        """
        Get a sequence of packets from the given DataFrame.
        Start index is randomly selected in the range [0, len(df) - seq_len].
        
        Args:
            df (pl.DataFrame): The DataFrame containing the packet data.
            seq_len (int): The sequence length.
        
        Returns:
            pl.Dataframe: The packet sequence.
        """
        # Check if DataFrame has enough rows for the sequence length
        if len(df) < seq_len:
            # Extract the sequence unpadded sequence
            packet_sequence = df

        else:
            length = seq_len
            # Generate random start index
            max_start_idx = len(df) - length
            start_idx = np.random.randint(0, max_start_idx + 1)
            # Extract the sequence
            packet_sequence = df.slice(start_idx, length)
        
        return packet_sequence
    
    def even_sample(self, batch_size: int):
        """
        Evenly sample a batch of packet sequences from the attack DataFrames.
        Results in a batch with balanced labels.
        input:
            batch_size: The batch size.
        output:
            batch_data: A list of packet sequences in polars DataFrame format.
        """
        label_indices = np.random.choice(len(self.attack_dfs), batch_size)
        batch_data = []
        for i in label_indices:
            df = self.attack_dfs[i]
            batch_data.append(self.get_packet_sequence_from_df(df, self.seq_len))
        return batch_data

    def sample_flow_even_label(self, batch_size: int):
        """
        Evenly sample a batch of packet sequences from the attack DataFrames.
        Results in a batch with balanced labels.
        input:
            batch_size: The batch size.
        output:
            batch_data: A list of packet sequences in polars DataFrame format.
        """
        label_indices = np.random.choice(len(self.attack_dfs), batch_size)

        def process_index(i):
            df = self.attack_dfs[i]
            flow_pick = df.select(pl.col("FlowID").sample(n=1, with_replacement=False)).item()
            df = df.filter(pl.col("FlowID") == flow_pick)
            return self.get_packet_sequence_from_df(df, self.seq_len)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            batch_data = list(executor.map(process_index, label_indices))

        return batch_data

    def sample_epoch_packet_indices(self, batch_size: int):
        """
        Sample batches of packets from the training data randomly.
        Returns batches of indices to draw from training data to complete one epoch.
        Batches affected by class imbalance.
        input:
            batch_size: The batch size.
        output:
            batch_data: A list of batch indices in numpy array format intended to draw from training data.
        """
        # Get total number of samples
        num_samples = self.data.height
        
        # Generate and shuffle indices
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        # Split indices into batches
        batch_indices = [
            indices[i:i + batch_size] 
            for i in range(0, num_samples, batch_size)
        ]
        
        return batch_indices

    def get_pretraining_data(self, indices: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Retrieve the 'data' and 'mask' columns as NumPy arrays for the given indices.
        Apply mask to data.
        Return the masked bytes and the proto hierarchy values.

        Args:
            indices (np.ndarray): The indices to retrieve data for.
        Returns:
            list: A list of NumPy arrays containing the masked 'data' values.
            list: A list of proto hierarchy values.
        """
        selected_data = self.data["data"][indices].to_numpy()
        selected_masks = self.data["mask"][indices].to_numpy()
        selected_proto_hierarchy = self.data["proto_hierarchy"][indices].to_numpy()

        masked_bytes = self.apply_mask(selected_data, selected_masks)

        return masked_bytes, selected_proto_hierarchy

    def apply_mask(self, bytes, masks):
        """
        Apply the mask to the bytes.
        
        Args:
            bytes (list): A list of NumPy arrays containing the 'data' values.
            masks (list): A list of NumPy arrays containing the 'mask' values.
        
        Returns:
            list: A list of NumPy arrays containing the masked 'data' values.
        """
        # Invert array_two (swap 0s and 1s)
        masks_inverted = [1 - sub_array for sub_array in masks]

        # Perform element-wise multiplication
        masked_bytes = [a * b for a, b in zip(bytes, masks_inverted)]
        
        return masked_bytes

    def get_bytes_as_numpy(self, df: pl.DataFrame) -> tuple[np.ndarray, str]:
        """
        Retrieve the 'data' column as a NumPy array for the entire DataFrame.
        Retrieve the 'mask' column and apply it to data.
        And return the label assuming all rows have the same label.
        
        Args:
            df (pl.DataFrame): The Polars DataFrame containing the 'data' column.
        
        Returns:
            list: A list of NumPy arrays containing all 'data' values.
            label: The label of the data as string
        """
        selected = df["data"].to_numpy()
        masks = df["mask"].to_numpy()

        masked_bytes = self.apply_mask(selected, masks)

        label = df["AttackLabel"][0]
        return masked_bytes, label
    
    def draw_encoded_batch(self, batch_size: int) -> tuple[torch.Tensor, list[str]]:
        """
        Get a batch of packet sequences.
        
        Args:
            batch_size (int): The batch size.
        
        Returns:
            torch.Tensor: The batch of packet sequences.
        """
        batch_data = []
        batch_labels = []
        batch_dfs = self.even_sample(batch_size)
        for seq_df in batch_dfs:
            bytes, label = self.get_bytes_as_numpy(seq_df)
            InputIDs = self.InputIDEncoder.construct_input_ids(bytes)
            InputIDs = torch.tensor(InputIDs)
            batch_data.append(InputIDs)
            batch_labels.append(label)
        return torch.stack(batch_data), batch_labels

    def pad_sequence_IDs(self, sequence: np.ndarray) -> np.ndarray:
        """
        Pad the given sequence of packets with the full byte sequences consisting of the padding token.
        
        Args:
            sequence (np.ndarray): The sequence to pad.
        
        Returns:
            np.ndarray: The padded sequence.
        """
        sequence_length = sequence.shape[0]
        padding_length = (self.seq_len - sequence_length) + 1
        padded_packets = np.ones((padding_length, 1520), dtype=np.int32) * self.InputIDEncoder.SpecialIDs["<pad>"]
        padded_sequence = np.concatenate((sequence, padded_packets), axis=0)
        return padded_sequence, sequence_length

    def draw_validation_batch(self) -> tuple[torch.Tensor, list[str]]:
        batch_size = self.batch_size
        batch_data = []
        batch_labels = []
        batch_dfs = self.SeqDfs[self.current_batch * batch_size:(self.current_batch + 1) * batch_size]
        sequence_lengths = np.zeros(len(batch_dfs), dtype=np.int32)
        for seq_index, seq_df in enumerate(batch_dfs):
            bytes, label = self.get_bytes_as_numpy(seq_df)
            InputIDs = self.InputIDEncoder.construct_input_ids(bytes)
            InputIDs, _sequence_length = self.pad_sequence_IDs(InputIDs) ## Sequence_length usefull if CLS Classification for sequences is implemented
            InputIDs = torch.tensor(InputIDs)
            batch_data.append(InputIDs)
            batch_labels.append(label)
            sequence_lengths[seq_index] = _sequence_length
        self.current_batch += 1
        if self.current_batch >= self.batches:
            self.current_batch = 0
            self.OnGoing = False
        return torch.stack(batch_data), batch_labels, torch.tensor(sequence_lengths)

    def FlowDf2Seq(self, Flow_Df, packets_per_sequence):
        Flow_Df_len = Flow_Df.height
        num_sequences = math.floor(Flow_Df_len / packets_per_sequence)
        if num_sequences == 0:
            self.SeqDfs.append(Flow_Df)
        else:
            for i in range(num_sequences):
                start_idx = i * packets_per_sequence
                end_idx = start_idx + packets_per_sequence
                sequence = Flow_Df.slice(start_idx, packets_per_sequence)
                self.SeqDfs.append(sequence)

    def fill_seq_dfs(self):
        for attack in self.data["AttackLabel"].unique().to_list():
            print(f"Processing attack: {attack}")
            Attack_Df = self.data.filter(pl.col("AttackLabel") == attack)
            Flows = Attack_Df["FlowID"].unique().to_list()
            for Flow in Flows:
                Flow_Df = Attack_Df.filter(pl.col("FlowID") == Flow)
                self.FlowDf2Seq(Flow_Df, self.seq_len)


class PreTrainingDatasetHandler():
    def __init__(self, data: pl.DataFrame, seq_len: int, encoder: ID_Encoder):
        self.data = data
        self.seq_len = seq_len # Number of packets in a sequence
        self.InputIDEncoder = encoder
        self._flow_index = None # built lazily, see build_flow_index

    def build_flow_index(self) -> pl.DataFrame:
        """
        Map every flow_key to the row indices of its packets, in timestamp order.

        Built once and cached on the handler. This is deliberately *not* done in
        __init__: the packet-level scripts construct this handler with seq_len=1
        and never touch flows, so they must not pay for the sort.

        Returns:
            pl.DataFrame: columns `flow_key` (str) and `row_idx` (list[u32]),
                          one row per flow. Flow order is the sorted flow_key
                          order, and within a flow the indices are in
                          (timestamp_s, timestamp_us) order.
        """
        if self._flow_index is None:
            self._flow_index = (
                self.data
                .with_row_index("row_idx")
                .sort(["flow_key", "timestamp_s", "timestamp_us"])
                .group_by("flow_key", maintain_order=True)
                .agg(pl.col("row_idx"))
            )
        return self._flow_index

    def get_packet_sequence_from_df(self, df: pl.DataFrame, seq_len: int):
        """
        Get a sequence of packets from the given DataFrame.
        Start index is randomly selected in the range [0, len(df) - seq_len].
        
        Args:
            df (pl.DataFrame): The DataFrame containing the packet data.
            seq_len (int): The sequence length.
        
        Returns:
            pl.Dataframe: The packet sequence.
        """
        # Check if DataFrame has enough rows for the sequence length
        if len(df) < seq_len:
            # Extract the sequence unpadded sequence
            packet_sequence = df

        else:
            length = seq_len
            # Generate random start index
            max_start_idx = len(df) - length
            start_idx = np.random.randint(0, max_start_idx + 1)
            # Extract the sequence
            packet_sequence = df.slice(start_idx, length)
        
        return packet_sequence

    def sample_flow_batch(self, batch_size: int) -> list[pl.DataFrame]:
        """
        Sample `batch_size` flows and return one time-ordered packet window
        (up to self.seq_len packets) per flow. Used to build inputs for
        sequence-level (auto)encoder training, where a "sample" is a flow
        rather than an individual packet.

        Flows are drawn with replacement. Row lookup goes through the cached
        flow index (build_flow_index), so a batch costs one gather rather than
        `batch_size` full-table scans.

        input:
            batch_size: The batch size.
        output:
            batch_data: A list of packet sequences in polars DataFrame format,
                        each sorted by (timestamp_s, timestamp_us).
        """
        flow_index = self.build_flow_index()
        flow_picks = np.random.randint(0, flow_index.height, size=batch_size)
        row_idx_col = flow_index["row_idx"]

        batch_data = []
        for flow in flow_picks:
            rows = row_idx_col[int(flow)].to_numpy()
            length = rows.shape[0]
            if length > self.seq_len:
                start = np.random.randint(0, length - self.seq_len + 1)
                rows = rows[start:start + self.seq_len]
            batch_data.append(self.data[rows])

        return batch_data

    def sample_epoch_packet_indices(self, batch_size: int):
        """
        Sample batches of packets from the training data randomly.
        Returns batches of indices to draw from training data to complete one epoch.
        Batches affected by class imbalance.
        input:
            batch_size: The batch size.
        output:
            batch_data: A list of batch indices in numpy array format intended to draw from training data.
        """
        # Get total number of samples
        num_samples = self.data.height
        
        # Generate and shuffle indices
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        # Split indices into batches
        batch_indices = [
            indices[i:i + batch_size] 
            for i in range(0, num_samples, batch_size)
        ]
        
        return batch_indices
    
    def get_pretraining_data(self, indices: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Retrieve the 'data' and 'mask' columns as NumPy arrays for the given indices.
        Apply mask to data.
        Return the masked bytes and the proto hierarchy values.

        Args:
            indices (np.ndarray): The indices to retrieve data for.
        Returns:
            list: A list of NumPy arrays containing the masked 'data' values.
            list: A list of proto hierarchy values.
        """
        selected_data = self.data["data"][indices].to_numpy()
        selected_masks = self.data["mask"][indices].to_numpy()
        selected_proto_hierarchy = self.data["proto_hierarchy"][indices].to_numpy()

        masked_bytes = self.apply_mask(selected_data, selected_masks)

        return masked_bytes, selected_proto_hierarchy

    def apply_mask(self, bytes, masks):
        """
        Apply the mask to the bytes.
        
        Args:
            bytes (list): A list of NumPy ndarray containing the 'data' values as byte objects.
            masks (list): A list of NumPy ndarrays containing the 'mask' values as byte objects.
        
        Returns:
            list: A list of NumPy arrays containing the masked 'data' values.
        """
        masked_bytes = []
        for data, mask in zip(bytes, masks):
            data_array = np.frombuffer(data, dtype=np.uint8)
            mask_array = np.frombuffer(mask, dtype=np.uint8)
            data_array = data_array.astype(np.int32)
            mask_array = mask_array.astype(np.int32)
            masked_data = data_array * (1 - mask_array) + mask_array * self.InputIDEncoder.SpecialIDs["<EndPointMasking>"]
            masked_bytes.append(masked_data)
        
        return masked_bytes

    def get_bytes_as_numpy(self, df: pl.DataFrame) -> tuple[np.ndarray, str]:
        """
        Retrieve the 'data' column as a NumPy array for the entire DataFrame.
        Retrieve the 'mask' column and apply it to data.
        And return the flow_key, since all rows of a sequence share one flow.

        Args:
            df (pl.DataFrame): The Polars DataFrame containing the 'data' column.

        Returns:
            list: A list of NumPy arrays containing all 'data' values.
            label: The flow_key of the sequence, as a string
        """
        selected = df["data"].to_numpy()
        masks = df["mask"].to_numpy()

        masked_bytes = self.apply_mask(selected, masks)

        label = df["flow_key"][0]
        return masked_bytes, label

    def pad_sequence_IDs(self, sequence: np.ndarray) -> np.ndarray:
        """
        Pad the given sequence of packets with the full byte sequences consisting of the padding token.

        Args:
            sequence (np.ndarray): The sequence to pad.

        Returns:
            np.ndarray: The padded sequence.
        """
        sequence_length = sequence.shape[0]
        padding_length = (self.seq_len - sequence_length) + 1
        padded_packets = np.ones((padding_length, 1520), dtype=np.int32) * self.InputIDEncoder.SpecialIDs["<pad>"]
        padded_sequence = np.concatenate((sequence, padded_packets), axis=0)
        return padded_sequence, sequence_length

    def draw_sequence_batch(self, batch_size: int) -> tuple[torch.Tensor, list[str], torch.Tensor]:
        """
        Get a batch of packet sequences, one flow per sample, for sequence-level
        (auto)encoder training. Flows longer than self.seq_len are randomly
        windowed; every sequence is padded out to self.seq_len + 1 packet slots
        (the extra slot is where Sequence_Encoder later writes its own CLS).

        Args:
            batch_size (int): The batch size.

        Returns:
            torch.Tensor: (batch_size, self.seq_len + 1, 1520) token ids.
            list[str]: The flow_key of each sampled sequence.
            torch.Tensor: (batch_size,) number of real (non-padding) packets.
        """
        batch_data = []
        batch_labels = []
        sequence_lengths = np.zeros(batch_size, dtype=np.int32)
        batch_dfs = self.sample_flow_batch(batch_size)
        for seq_index, seq_df in enumerate(batch_dfs):
            bytes_, label = self.get_bytes_as_numpy(seq_df)
            InputIDs = self.InputIDEncoder.construct_input_ids(bytes_)
            InputIDs, sequence_length = self.pad_sequence_IDs(InputIDs)
            InputIDs = torch.tensor(InputIDs)
            batch_data.append(InputIDs)
            batch_labels.append(label)
            sequence_lengths[seq_index] = sequence_length
        return torch.stack(batch_data), batch_labels, torch.tensor(sequence_lengths)


def checkpoint_fingerprint(path: str) -> str:
    """
    sha256 of a checkpoint file, used to tie a latent cache to the exact packet
    encoder that produced it.

    Comparing paths is not enough: retraining the packet-level model usually
    writes back to the same filename, and a cache built from the old weights
    would then be silently reused with the new ones.
    """
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_latent_cache(cache_dir: str, packet_ae_ckpt: str = None) -> tuple[torch.Tensor, pl.DataFrame, dict]:
    """
    Load a packet-latent cache written by PreTraining/CachePacketLatents.py.

    Args:
        cache_dir:      directory holding meta.json, flow_offsets.parquet and shard_*.npy.
        packet_ae_ckpt: if given, verify the cache was built from this exact
                        checkpoint and raise if not.

    Returns:
        torch.Tensor: (N, D) float16 latents, in cache row order.
        pl.DataFrame: flow_key / start / length.
        dict:         the cache's meta.json.
    """
    import json
    import os

    meta_path = os.path.join(cache_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No latent cache at {cache_dir}. Build it first with "
            f"`python -m RawByteTrafficModelling.PreTraining.CachePacketLatents` "
            f"(set SPLIT_FILE/CACHE_DIR at the top of that script)."
        )
    with open(meta_path) as f:
        meta = json.load(f)

    if packet_ae_ckpt is not None:
        expected = meta.get("packet_ae_sha256")
        actual = checkpoint_fingerprint(packet_ae_ckpt)
        if expected is None:
            raise ValueError(
                f"{cache_dir} predates checkpoint fingerprinting and cannot be "
                f"verified against {packet_ae_ckpt}. Rebuild it with CachePacketLatents."
            )
        if expected != actual:
            raise ValueError(
                f"{cache_dir} was built from a different packet encoder "
                f"(cache {expected[:12]}, {packet_ae_ckpt} is {actual[:12]}). "
                f"The latents are stale -- rebuild with CachePacketLatents."
            )

    shards = [np.load(os.path.join(cache_dir, f"shard_{i:04d}.npy"))
              for i in range(meta["num_shards"])]
    latents = torch.from_numpy(np.concatenate(shards, axis=0))
    if latents.shape[0] != meta["num_rows"]:
        raise ValueError(f"{cache_dir}: shards hold {latents.shape[0]} rows, "
                         f"meta.json says {meta['num_rows']}")

    flow_offsets = pl.read_parquet(os.path.join(cache_dir, "flow_offsets.parquet"))
    return latents, flow_offsets, meta


class CachedLatentSequenceHandler():
    """
    Sequence-level batching straight out of a cached packet-latent array.

    The packet encoder is frozen during sequence-level training, so its output
    for a given packet never changes. CachePacketLatents.py encodes every packet
    of a split once and writes the latents in flow-grouped, timestamp-sorted
    order; each flow is therefore a *contiguous* slice of that array and a
    training batch is a gather rather than 65 x batch_size packet forwards.

    Shapes match what PreTrainingDatasetHandler.pad_sequence_IDs produces on the
    token path: P = packets_per_sequence slots, with seq_lens in [1, P-1] real
    packets, which is what Sequence_Encoder's scatter-at-seq_len requires.
    """

    def __init__(self, latents: torch.Tensor, flow_offsets: pl.DataFrame,
                 packets_per_sequence: int):
        """
        Args:
            latents:      (N, D) packet latents, in the cache's row order.
            flow_offsets: columns `flow_key`, `start`, `length` -- one row per
                          flow, `start` indexing into `latents`.
            packets_per_sequence: P, including the slot the seq-CLS overwrites.
        """
        self.latents = latents
        self.flow_keys = flow_offsets["flow_key"].to_list()
        self.starts = flow_offsets["start"].to_numpy().astype(np.int64)
        self.lengths = flow_offsets["length"].to_numpy().astype(np.int64)
        self.num_packets = packets_per_sequence
        self.seq_len = packets_per_sequence - 1   # max real packets per sequence
        self.latent_dim = latents.shape[1]

    def epoch_flow_batches(self, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
        """
        One epoch's worth of flow indices, shuffled, without replacement.

        The flow-level analogue of sample_epoch_packet_indices -- every flow is
        visited exactly once per epoch (the token-path draw_sequence_batch
        samples with replacement and has no epoch notion).
        """
        flows = rng.permutation(len(self.starts))
        return [flows[i:i + batch_size] for i in range(0, len(flows), batch_size)]

    def _gather(self, starts: np.ndarray, takes: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """(starts, takes) row slices -> ((B, P, D) float32 latents, (B,) lengths)."""
        B = starts.shape[0]
        src = np.concatenate([np.arange(s, s + t) for s, t in zip(starts, takes)])
        dest_b = np.repeat(np.arange(B), takes)
        dest_p = np.concatenate([np.arange(t) for t in takes])

        out = torch.zeros(B, self.num_packets, self.latent_dim, dtype=torch.float32)
        out[torch.from_numpy(dest_b).long(), torch.from_numpy(dest_p).long()] = \
            self.latents[torch.from_numpy(src).long()].float()
        return out, torch.from_numpy(takes.copy()).long()

    def draw_latent_batch(self, flow_ids: np.ndarray,
                          rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Training batch: one randomly positioned window per flow.

        Windowing mirrors get_packet_sequence_from_df -- flows longer than
        seq_len get a random contiguous window, shorter flows are taken whole
        and zero-padded. Padding is zeros because DynamicCLSPooling returns a
        zero vector for an all-<pad> packet, so the token path pads with zeros
        too; anything else would show the sequence backbone a different padding
        distribution than it was smoke-tested on.
        """
        lengths = self.lengths[flow_ids]
        takes = np.minimum(lengths, self.seq_len)
        slack = lengths - takes
        offsets = (rng.random(len(flow_ids)) * (slack + 1)).astype(np.int64)
        offsets = np.minimum(offsets, slack)
        return self._gather(self.starts[flow_ids] + offsets, takes)

    def enumerate_windows(self) -> np.ndarray:
        """
        Deterministic, non-overlapping windows over every flow.

        Same chopping as ValidationDatasetHandler.FlowDf2Seq: a flow shorter
        than seq_len yields one short window, otherwise floor(L / seq_len) full
        windows and the remainder is dropped.

        Returns:
            np.ndarray: (W, 2) array of (start_row, length) pairs.
        """
        windows = []
        for start, length in zip(self.starts, self.lengths):
            num_sequences = length // self.seq_len
            if num_sequences == 0:
                windows.append((start, length))
            else:
                for i in range(num_sequences):
                    windows.append((start + i * self.seq_len, self.seq_len))
        return np.array(windows, dtype=np.int64)

    def latent_batch_from_windows(self, windows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """(W, 2) (start, length) rows -> ((W, P, D) latents, (W,) lengths)."""
        return self._gather(windows[:, 0], windows[:, 1])


import time
# Test the class
if __name__ == '__main__':
    import os
    # Load training data
    data_dir = "Model_Trainings/RawByteInput/data/WithFlowID"
    training_data_file = os.path.join(data_dir, "train_flow_mask.ipc")
    training_data = pl.read_ipc(training_data_file)
    # Initialize the training dataset handler
    ID_Encoder = ID_Encoder(SpecialIDs = {"<pad>": 256, "</s>": 257, "<CLS>": 258, "<mask>": 259}, CLS_Placement="EOS")
    handler = TrainingDatasetHandler(training_data, 16, ID_Encoder)
    batch_size = 512

    # Measure runtime of the original method
    start_time = time.time()
    BatchIDs, BatchLabels, seq_lens = handler.draw_encoded_flow_batch(batch_size)
    end_time = time.time()
    print(f"Original method runtime: {end_time - start_time:.6f} seconds")

    print(BatchIDs.shape)
    print(BatchIDs[0])