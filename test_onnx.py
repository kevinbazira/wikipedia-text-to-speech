import time
import soundfile as sf
import onnxruntime as ort

# 🛑 1. THE ONNX THREAD-THRASHING FIX (Monkeypatch)
# Intercept ONNX Runtime initialization to strictly force 1 thread
original_init = ort.InferenceSession.__init__

def patched_init(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
    if sess_options is None:
        sess_options = ort.SessionOptions()
    
    # Force 1 thread for mathematical operations
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    
    # Call the original initialization with our strict options
    original_init(self, path_or_bytes, sess_options, providers, provider_options, **kwargs)

# Apply the patch!
ort.InferenceSession.__init__ = patched_init


print("1. Importing libraries...")
try:
    from kokoro_onnx import Kokoro
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

print("2. Loading ONNX model into memory...")
try:
    start_load = time.time()
    kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
    print(f"✅ Model loaded in {time.time() - start_load:.2f} seconds.")
except Exception as e:
    print(f"❌ Failed to load model. Error: {e}")
    exit(1)

text = "Wikipedia is a free online encyclopedia, created and edited by volunteers."
print(f"3. Generating audio for: '{text}'")

try:
    start_gen = time.time()
    samples, sample_rate = kokoro.create(text, voice="af_heart", speed=1.0, lang="en-us")
    
    print("4. Saving audio...")
    sf.write("onnx_test.wav", samples, sample_rate)
    
    elapsed = time.time() - start_gen
    print(f"✅ Success! Audio saved to onnx_test.wav in {elapsed:.2f} seconds.")
except Exception as e:
    print(f"❌ Failed during generation: {e}")