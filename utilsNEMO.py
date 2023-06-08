from pydub import AudioSegment, utils
import os
import math
import json
import noisereduce

def extract_audio_from_mp4(mp4_path, output_folder, reduce_noise=False):
    audio = AudioSegment.from_file(mp4_path)

    duration = audio.duration_seconds
    chunk_duration = 15  # Duration of each chunk in seconds
    num_chunks = math.ceil(duration / chunk_duration)

    chunks_metadata = []

    for i in range(num_chunks):
        chunk_start = i * chunk_duration * 1000  # Convert to milliseconds
        chunk_end = min((i + 1) * chunk_duration * 1000, len(audio))
        chunk = audio[chunk_start:chunk_end]

        chunk = chunk.set_frame_rate(16000)  # Set the sampling rate to 16000
        chunk = chunk.set_channels(1)  # Set mono channel
        chunk = chunk.set_sample_width(2)  # Set sample width to 2 bytes (16-bit)

        if reduce_noise:
          # ----- EXPERIMENT: NOISE REDUCTION -----
          # Convert chunk to numpy array
          chunk_array = np.array(chunk.get_array_of_samples())

          # Apply denoising using noisereduce
          reduced_noise = noisereduce.reduce_noise(y=chunk_array, sr=16000)

          # Convert back to AudioSegment
          denoised_chunk = AudioSegment(
              data=reduced_noise.tobytes(),
              sample_width=2,
              frame_rate=16000,
              channels=1
          )

          chunk = denoised_chunk


          # ----- END EXPERIMENT -----


        wav_filename = f"chunk_{i+1}.wav"
        wav_path = os.path.join(output_folder, wav_filename)
        # Export chunk as WAV
        chunk.export(os.path.abspath(wav_path), format="wav")

        chunk_duration = chunk.duration_seconds
        chunk_metadata = {
            "audio_filepath": os.path.abspath(wav_path),
            "duration": chunk_duration
        }
        chunks_metadata.append(chunk_metadata)

        print("Chunk", i+1, "extracted and saved as:", wav_path)

    json_filename = "metadata.json"
    json_path = os.path.join(output_folder, json_filename)
    with open(json_path, "w") as json_file:
        for metadata in chunks_metadata:
            json_file.write(json.dumps(metadata) + '\n')

    print("Metadata saved as:", json_path)

