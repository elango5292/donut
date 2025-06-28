import io
import torch.quantization
import re
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from transformers import DonutProcessor, VisionEncoderDecoderModel
from typing import List

# Initialize FastAPI app
app = FastAPI()

# Load model and processor
print("Loading DocVQA model and processor...")
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa", cache_dir="/app/cache")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa", cache_dir="/app/cache")
# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
print(f"Model moved to {device.upper()} and dynamically quantized.")
print(f"Decoder max position embeddings: {model.decoder.config.max_position_embeddings}")

@app.post("/answer")
async def answer_questions(image: UploadFile = File(...), questions: List[str] = Form(...)):
    try:
        # Read and process the image
        image_data = await image.read()
        print(f"Received image data of type: {type(image_data)} and size: {len(image_data)} bytes")
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Process each question and collect answers
        answers = []
        for question in questions:
            task_prompt = f"<s_question>{question}</s_question><s_answer>"
            decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
            pixel_values = processor(image, return_tensors="pt").pixel_values
            
            # Run inference
            outputs = model.generate(
                pixel_values.to(device),
                decoder_input_ids=decoder_input_ids.to(device),
                max_length=model.decoder.config.max_position_embeddings,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
            
            # Decode and clean the output
            sequence = processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
            cleaned_answer = re.sub(r"<s_question>.*</s_question><s_answer>", "", sequence).strip()
            answers.append(cleaned_answer)
        
        return {"answers": answers}
    except Exception as e:
        return {"error": str(e)}