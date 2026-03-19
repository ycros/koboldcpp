import ctypes
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import koboldcpp as kcpp


class FakeHandle:
    def __init__(self):
        self.last_inputs = None

    def generate(self, inputs):
        self.last_inputs = inputs
        out = kcpp.generation_outputs()
        out.status = 1
        out.stopreason = 1
        out.prompt_tokens = 2
        out.completion_tokens = 3
        out.text = b"ok"
        out.error_message = b""
        return out


def decode_c_string(value):
    if not value:
        return ""
    return ctypes.string_at(value).decode("utf-8")


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def main():
    fake_handle = FakeHandle()
    kcpp.handle = fake_handle
    kcpp.args = types.SimpleNamespace(defaultgenamt=64, allowcustomsamplers=True, genlimit=0)
    kcpp.currentusergenkey = ""
    kcpp.totalgens = 0
    kcpp.pendingabortkey = ""
    kcpp.chatcompl_adapter = None
    kcpp.maxctx = 8192
    kcpp.showsamplerwarning = False

    res = kcpp.generate({
        "prompt": "hello",
        "custom_sampler": "function sample(s) { return s.pick(); }",
        "custom_sampler_params": {"alpha": 1.25},
    })
    assert_true(res["status"] == 1, "mock generation succeeds with custom sampler payload")
    assert_true([fake_handle.last_inputs.sampler_order[i] for i in range(fake_handle.last_inputs.sampler_len)] == [6, 0, 1, 3, 4, 2, 7, 5],
                "default sampler_order injects custom sampler before temp")
    assert_true(decode_c_string(fake_handle.last_inputs.custom_sampler) == "function sample(s) { return s.pick(); }",
                "custom sampler source is marshalled across ctypes")
    assert_true(json.loads(decode_c_string(fake_handle.last_inputs.custom_sampler_params)) == {"alpha": 1.25},
                "custom sampler params are marshalled as JSON")
    assert_true(fake_handle.last_inputs.custom_sampler_debug is False,
                "custom sampler debugging stays opt-in on the base API")

    res = kcpp.generate({
        "prompt": "hello",
        "custom_sampler": "function sample(s) { return s.pick(); }",
        "custom_sampler_params": "plain-string",
    })
    assert_true(res["status"] == 1, "mock generation succeeds with plain string params")
    assert_true(json.loads(decode_c_string(fake_handle.last_inputs.custom_sampler_params)) == "plain-string",
                "plain string params are wrapped as JSON strings")

    res = kcpp.generate({
        "prompt": "hello",
        "custom_sampler": "function sample(s) { return s.pick(); }",
        "custom_sampler_params": '{"mode":"raw-json"}',
    })
    assert_true(res["status"] == 1, "mock generation succeeds with raw JSON text params")
    assert_true(json.loads(decode_c_string(fake_handle.last_inputs.custom_sampler_params)) == {"mode": "raw-json"},
                "pre-serialized JSON params are not double-encoded")

    res = kcpp.generate({
        "prompt": "hello",
        "custom_sampler": "function sample(s) { return s.pick(); }",
        "sampler_order": [6, 0, 1, 3, 4, 2, 5],
    })
    assert_true(res["status"] == 0 and res["stopreason"] == -2,
                "missing custom sampler slot fails early")
    assert_true("slot 7" in res["error"], "missing custom sampler slot returns a clear error")

    kcpp.args = types.SimpleNamespace(defaultgenamt=64, allowcustomsamplers=False, genlimit=0)
    res = kcpp.generate({
        "prompt": "hello",
        "custom_sampler": "function sample(s) { return s.pick(); }",
    })
    assert_true(res["status"] == 0 and res["stopreason"] == -2,
                "disabled custom samplers are rejected")
    assert_true("--allowcustomsamplers" in res["error"], "disabled custom sampler error explains the launch flag")

    print("all custom sampler API regression tests passed")


if __name__ == "__main__":
    main()
