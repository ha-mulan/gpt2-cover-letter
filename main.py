from transformers import AutoModelWithLMHead, AutoTokenizer, top_k_top_p_filtering
import torch
from flask import Flask, request, Response
from torch.nn import functional as F
from queue import Queue, Empty
import time
import threading

# Server & Handling Setting
app = Flask(__name__)

requests_queue = Queue()
BATCH_SIZE = 1
CHECK_INTERVAL = 0.1

tokenizer = AutoTokenizer.from_pretrained("jonasmue/cover-letter-gpt2")
model = AutoModelWithLMHead.from_pretrained("jonasmue/cover-letter-gpt2", return_dict=True)


# Queue 핸들링
def handle_requests_by_batch():
    while True:
        requests_batch = []
        while not (len(requests_batch) >= BATCH_SIZE):
            try:
                requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

            for requests in requests_batch:
                requests['output'] = run(requests['input'][0])


# 쓰레드
threading.Thread(target=handle_requests_by_batch).start()


# Sketch Start
def run(sequence):
    input_ids = tokenizer.encode(sequence, return_tensors="pt")
    # get logits of last hidden state
    next_token_logits = model(input_ids).logits[:, -1, :]
    # filter
    filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)
    # sample
    probs = F.softmax(filtered_next_token_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=5)

    result = dict()
    for idx, token in enumerate(next_token.tolist()[0]):
        result[idx] = tokenizer.decode(token)
    return result


@app.route("/gpt2-word", methods=['GET'])
def gpt2():
    # 큐에 쌓여있을 경우,
    if requests_queue.qsize() > BATCH_SIZE:
        return jsonify({'error': 'TooManyReqeusts'}), 429

    # 웹페이지로부터 이미지와 스타일 정보를 얻어옴.
    try:
        text = request.args['text']
        print(text)

    except Exception:
        print("Empty Text")
        return Response("fail", status=400)

    # Queue - put data
    req = {
        'input': [text]
    }
    requests_queue.put(req)

    # Queue - wait & check
    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)

    return req['output']


# Health Check
@app.route("/healthz", methods=["GET"])
def healthCheck():
    return "", 200


if __name__ == "__main__":
    from waitress import serve
    serve(app, host='0.0.0.0', port=80)