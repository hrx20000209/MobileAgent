# Load model directly
from transformers import AutoProcessor, AutoModelForVision2Seq

processor = AutoProcessor.from_pretrained("ByteDance-Seed/UI-TARS-1.5-7B")
model = AutoModelForVision2Seq.from_pretrained("ByteDance-Seed/UI-TARS-1.5-7B")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What do you see in the screen?"}
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
