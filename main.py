import io, json, os, subprocess, argparse, sys, re
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import urlretrieve
import wave

# --- CONSTANTS ---
VOICES_FILE = 'voices.json'
CONFIG_FILE = 'config.json'
PIPER_DIR = os.path.abspath("./piper")
PIPER_BINARY = os.path.join(PIPER_DIR, "piper")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DOWNLOAD_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"

# --- GLOBAL STATE ---
# We load this once at startup
CUSTOM_REPLACEMENTS = []

# --- UTILS ---
def load_config():
    """Loads custom replacement rules from config.json"""
    global CUSTOM_REPLACEMENTS
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                conf = json.load(f)
                CUSTOM_REPLACEMENTS = conf.get('replacements', [])
                print(f"--- Loaded {len(CUSTOM_REPLACEMENTS)} custom replacement rules ---")
        except Exception as e:
            print(f"⚠️ Error loading config.json: {e}")

def normalize_text(text):
    """
    Sanitizes novel text:
    1. Applies custom regex rules from config.json
    2. Squashes repeated characters (Generic Safety)
    """
    # 1. Custom Replacements (Priority)
    for rule in CUSTOM_REPLACEMENTS:
        try:
            # Case-insensitive replacement
            text = re.sub(rule['pattern'], rule['replacement'], text, flags=re.IGNORECASE)
        except re.error as e:
            print(f"⚠️ Invalid Regex '{rule['pattern']}': {e}")

    # 2. Generic Safety (Squash repeats)
    # Collapse 3+ repeats of any character into just 3 "Keeee" -> "Keee"
    text = re.sub(r'(.)\1{2,}', r'\1\1\1', text)
    
    # Collapse 3+ repeats of punctuation "!!!!" -> "!"
    text = re.sub(r'([!?.])\1{2,}', r'\1', text)
    
    return text

def load_voices_registry():
    if not os.path.exists(VOICES_FILE):
        print(f"❌ Error: {VOICES_FILE} not found.")
        print("👉 Download it: wget -O voices.json https://huggingface.co/rhasspy/piper-voices/raw/main/voices.json")
        sys.exit(1)
    with open(VOICES_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def list_voices(registry, lang_filter=None):
    print(f"\n{'VOICE KEY':<35} | {'LANGUAGE':<15} | {'QUALITY':<8}")
    print("-" * 65)
    for key, info in registry.items():
        lang_code = info['language']['code']
        if lang_filter and lang_filter not in lang_code:
            continue
        print(f"{key:<35} | {info['language']['name_english']:<15} | {info['quality']:<8}")
    print("-" * 65)

def get_model_paths(model_key, registry):
    if model_key not in registry:
        print(f"❌ Error: Model '{model_key}' not found in voices.json")
        sys.exit(1)
    
    data = registry[model_key]
    files = data['files']
    
    onnx_rel_path = next((k for k in files.keys() if k.endswith('.onnx')), None)
    config_rel_path = next((k for k in files.keys() if k.endswith('.onnx.json')), None)
    
    if not onnx_rel_path or not config_rel_path:
        print("❌ Error: Malformed registry entry")
        sys.exit(1)

    os.makedirs(MODEL_DIR, exist_ok=True)
    local_onnx = os.path.join(MODEL_DIR, os.path.basename(onnx_rel_path))
    local_config = os.path.join(MODEL_DIR, os.path.basename(config_rel_path))

    return {
        "key": model_key,
        "onnx_url": DOWNLOAD_BASE + onnx_rel_path,
        "config_url": DOWNLOAD_BASE + config_rel_path,
        "local_onnx": local_onnx,
        "local_config": local_config
    }

def ensure_model(paths):
    for url, local_path in [(paths['onnx_url'], paths['local_onnx']), (paths['config_url'], paths['local_config'])]:
        if not os.path.exists(local_path) or os.path.getsize(local_path) < 1000:
            print(f"⬇️ Downloading: {os.path.basename(local_path)}...")
            try:
                urlretrieve(url, local_path)
            except Exception as e:
                print(f"❌ Download failed: {e}")
                sys.exit(1)
    print(f"✅ Voice Ready: {paths['key']}")

def get_sample_rate(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)["audio"]["sample_rate"]
    except:
        return 22050

# --- SERVER ---
class PiperFactory:
    def __init__(self, onnx_path, sample_rate, cpu_core):
        self.onnx_path = onnx_path
        self.sample_rate = sample_rate
        self.cpu_core = cpu_core

    def handler(self):
        factory_ref = self
        
        class CustomHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                try:
                    length = int(self.headers.get('Content-Length', 0))
                    data = json.loads(self.rfile.read(length))
                    
                    # 1. Clean Text
                    raw_text = data.get("text", "").strip()
                    if not raw_text: return self.send_error(400, "Empty text")
                    
                    text = normalize_text(raw_text)

                    # 2. Calculate Speed
                    user_speed = float(data.get("speed", 1.0))
                    length_scale = 1.0 / max(0.1, user_speed)
                    length_scale = max(0.5, min(length_scale, 3.0))

                    print(f"📥 Processing {len(text)} chars | Speed: {user_speed}x")

                    cmd = [
                        PIPER_BINARY, 
                        "--model", factory_ref.onnx_path, 
                        "--output_file", "-", 
                        "--output_raw",
                        "--length_scale", str(length_scale)
                    ]
                    
                    if factory_ref.cpu_core is not None:
                        cmd = ["taskset", "-c", str(factory_ref.cpu_core)] + cmd

                    env = os.environ.copy()
                    env["LD_LIBRARY_PATH"] = f"{PIPER_DIR}:{env.get('LD_LIBRARY_PATH', '')}"
                    env["ESPEAK_DATA_PATH"] = os.path.join(PIPER_DIR, "espeak-ng-data")

                    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
                    raw_pcm, stderr = process.communicate(input=text.encode('utf-8'))

                    if process.returncode != 0:
                        print(f"❌ Piper Error: {stderr.decode()}")
                        return self.send_error(500, "Piper failed")

                    wav_buffer = io.BytesIO()
                    with wave.open(wav_buffer, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(factory_ref.sample_rate)
                        wf.writeframes(raw_pcm)

                    self.send_response(200)
                    self.send_header('Content-Type', 'audio/wav')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(wav_buffer.getvalue())

                except Exception as e:
                    print(f"❌ Error: {e}")
                    self.send_error(500)
        
        return CustomHandler

# --- MAIN ENTRY ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Piper TTS Universal Server")
    parser.add_argument("-m", "--model", help="Voice key")
    parser.add_argument("--list", action="store_true", help="List available voices")
    parser.add_argument("--lang", help="Filter --list by language")
    parser.add_argument("-p", "--port", type=int, default=5000)
    parser.add_argument("-c", "--cpu", type=int, default=None)
    args = parser.parse_args()

    registry = load_voices_registry()

    if args.list:
        list_voices(registry, args.lang)
        sys.exit(0)

    if not args.model:
        args.model = "en_US-amy-low"
        print(f"ℹ️ No model specified, defaulting to: {args.model}")

    if not os.path.exists(PIPER_BINARY):
        print(f"❌ Critical: Piper binary not found at {PIPER_BINARY}")
        sys.exit(1)

    # 1. Load Config
    load_config()

    # 2. Prepare Model
    paths = get_model_paths(args.model, registry)
    ensure_model(paths)
    sample_rate = get_sample_rate(paths['local_config'])

    # 3. Start Server
    print(f"--- 🚀 Server ({args.model}) running on port {args.port} ---")
    factory = PiperFactory(paths['local_onnx'], sample_rate, args.cpu)
    HTTPServer(('127.0.0.1', args.port), factory.handler()).serve_forever()