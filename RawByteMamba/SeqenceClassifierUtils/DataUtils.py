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