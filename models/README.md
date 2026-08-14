# Live audio model files

Place these two files directly in this folder:

- `audio_encoder_best.pt` — the unimodal `AudioAttentionEncoder` checkpoint.
- `audio_feature_extractor.pt` — a TorchScript audio feature extractor.

The feature extractor receives a float waveform shaped `[1, samples]` at 16 kHz
and must return embeddings shaped `[1, time, audio_input_dim]`. The default
`audio_input_dim` is 1024.

The encoder checkpoint may be either a raw state dictionary or a dictionary
containing `model_state_dict`. It may optionally include `audio_input_dim`,
`audio_hidden`, and `dropout` metadata.
