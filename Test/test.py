# Load model directly
from transformers import AutoProcessor, AutoModelForVision2Seq

processor = AutoProcessor.from_pretrained("ByteDance-Seed/UI-TARS-1.5-7B")
model = AutoModelForVision2Seq.from_pretrained("ByteDance-Seed/UI-TARS-1.5-7B")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image",
             "url": "https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.consumerreports.org%2Felectronics-computers%2Fcell-phones%2Fapple-iphone-11%2Fm399694%2F&psig=AOvVaw0-JM5SJKKSi7wM2nOmEgLq&ust=1755791212859000&source=images&cd=vfe&opi=89978449&ved=0CBUQjRxqFwoTCJjBsY_emY8DFQAAAAAdAAAAABAE"},
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
