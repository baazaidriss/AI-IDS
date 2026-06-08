import joblib
import pandas as pd

# Load saved model and encoder
model = joblib.load("rf_ids_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# Example network flow
sample = {
    "destination_port": 80,
    "flow_duration": 1000,
    "total_fwd_packets": 10,
    "total_length_of_fwd_packets": 500,
    "fwd_packet_length_max": 100,
    "fwd_packet_length_min": 20,
    "fwd_packet_length_mean": 50,
    "fwd_packet_length_std": 10,
    "bwd_packet_length_max": 100,
    "bwd_packet_length_min": 20,
    "bwd_packet_length_mean": 50,
    "flow_bytes_per_s": 1000,
    "flow_packets_per_s": 50,
    "flow_iat_mean": 10,
    "flow_iat_std": 5,
    "flow_iat_max": 20,
    "flow_iat_min": 1,
    "fwd_iat_mean": 10,
    "fwd_iat_std": 5,
    "fwd_iat_min": 1,
    "bwd_iat_total": 100,
    "bwd_iat_mean": 10,
    "bwd_iat_std": 5,
    "bwd_iat_max": 20,
    "bwd_iat_min": 1,
    "fwd_header_length": 40,
    "bwd_header_length": 40,
    "bwd_packets_per_s": 25,
    "min_packet_length": 20,
    "max_packet_length": 100,
    "packet_length_mean": 50,
    "packet_length_variance": 100,
    "fin_flag_count": 0,
    "psh_flag_count": 0,
    "ack_flag_count": 1,
    "init_win_bytes_forward": 1024,
    "init_win_bytes_backward": 1024,
    "act_data_pkt_fwd": 5,
    "min_seg_size_forward": 20,
    "active_mean": 100,
    "active_max": 200,
    "active_min": 50,
    "idle_mean": 500
}

# Convert to DataFrame
df = pd.DataFrame([sample])

# Make prediction
prediction = model.predict(df)

# Convert numeric prediction to attack name
attack_name = encoder.inverse_transform(prediction)

print("=" * 50)
print("AI IDS PREDICTION")
print("=" * 50)
print("Predicted Class:", attack_name[0])
print("=" * 50)