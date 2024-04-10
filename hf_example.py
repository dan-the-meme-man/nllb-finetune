from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-600M", src_lang="spa_Latn"
)
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

es_batch = [
    "Solo dura una semana.",
    "¡Era más que asqueroso!"
]

other_batch = [
    "Mä simanakiw",
    "¡Sinti axtañapuninwa!"
]

max_length = 256

batch = tokenizer(
    text = es_batch,
    return_tensors = 'pt',
    padding = 'longest',
    truncation = True,
    max_length = max_length
)

outputs = model.generate(
    **batch,
    forced_bos_token_id=tokenizer.lang_code_to_id["ayr_Latn"],
    max_length=max_length
)
decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

print(batch)
print('---')
print(outputs)
print('---')
print(decoded)