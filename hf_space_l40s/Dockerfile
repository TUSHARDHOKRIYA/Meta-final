FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir transformers>=4.46.0 trl==0.14.0 peft>=0.13.0 accelerate>=1.0.0 \
    datasets bitsandbytes matplotlib scipy huggingface_hub pydantic openai
COPY train_multitask_part1.py /app/
COPY train_multitask_part2.py /app/
COPY run_training.py /app/
ENV PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["python", "-u", "run_training.py"]
