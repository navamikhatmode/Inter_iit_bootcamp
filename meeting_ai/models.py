"""Thin wrappers around the Hugging Face models. Heavy imports happen lazily so the rest of the package
(and the unit tests) work without torch / transformers installed."""
import gc

from . import config as cfg
from .textutils import collapse_repetitions


class LLM:
    """Chat model with greedy decoding, optionally 4-bit (NF4) quantised."""

    def __init__(self, name, device_index=0, quant4=True):
        self.name, self.device_index, self.quant4 = name, device_index, quant4
        self.model = self.tokenizer = None

    @property
    def loaded(self):
        return self.model is not None

    def load(self):
        if self.loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        kwargs = {"torch_dtype": torch.float16, "device_map": {"": self.device_index}}
        if self.quant4:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(self.name, **kwargs).eval()

    def unload(self):
        import torch
        self.model = self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

    def generate(self, system, user, max_new_tokens=1024):
        import torch
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


class Transcriber:
    """Whisper speech-to-text with 30 s sliding windows and optional domain-term prompting."""

    def __init__(self, name, device_index=0, batch_size=4):
        self.name, self.device_index, self.batch_size = name, device_index, batch_size
        self.pipe = None

    @property
    def loaded(self):
        return self.pipe is not None

    def load(self):
        if self.loaded:
            return
        import torch
        from transformers import pipeline
        self.pipe = pipeline("automatic-speech-recognition", model=self.name,
                             torch_dtype=torch.float16, device=self.device_index)

    def unload(self):
        import torch
        self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()

    def transcribe(self, wav_path, hint_terms=""):
        gen = {"language": "english", "task": "transcribe"}
        kw = dict(return_timestamps=True, chunk_length_s=30, stride_length_s=5, batch_size=self.batch_size)
        result = None
        if hint_terms.strip():
            try:  # a glossary prompt nudges Whisper towards the right spelling of domain terms
                ids = self.pipe.tokenizer.get_prompt_ids("Glossary: " + hint_terms.strip(), return_tensors="pt")
                result = self.pipe(wav_path, generate_kwargs={**gen, "prompt_ids": ids.to(f"cuda:{self.device_index}")}, **kw)
            except Exception:  # noqa: BLE001 - prompting is best-effort, fall back to plain decoding
                result = None
        if result is None:
            result = self.pipe(wav_path, generate_kwargs=gen, **kw)
        return collapse_repetitions(result["text"].strip())
