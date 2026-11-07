import json
import os
import re
import time
from pathlib import Path
from tqdm import tqdm
import requests
from dotenv import load_dotenv


load_dotenv()


DIFY_BASE_URL = "http://localhost"
DIFY_API_URL = f"{DIFY_BASE_URL}/v1/chat-messages"

# 請改成 Windows 本機實際的 ORQA 測試資料路徑
TEST_DATA_PATH = Path(r"./dataset/ORQA_test.jsonl")

# 測試結果儲存路徑
OUTPUT_DIR = Path("./results/dify_rag_1020")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_FILE = (OUTPUT_DIR / "predictions.json")
INVALID_FILE = (OUTPUT_DIR / "invalid_answers.json")
WRONG_FILE = (OUTPUT_DIR / "wrong_predictions.json")

# 測試 ? 題
TEST_LIMIT = 1468

# API Key 從環境變數取得，不直接寫在程式中
DIFY_API_KEY = os.getenv("DIFY_API_KEY")

# API 等待時間上限
REQUEST_TIMEOUT = 180

# 將模型輸出的選項字母轉換成整數
ANSWER_MAPPING = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
}


def load_jsonl(file_path):

    dataset = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                dataset.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"第 {line_number} 行不是有效的 JSON：{exc}"
                ) from exc

    return dataset

def read_jsonl_records(file_path):
    """讀取既有的 JSONL 紀錄；檔案不存在時回傳空 list。"""

    if not file_path.exists():
        return []

    records = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{file_path} 第 {line_number} 行不是有效的 JSON：{exc}"
                ) from exc

    return records


def append_jsonl(file_path, record):
    """將單筆紀錄追加到 JSONL，並立即寫入磁碟。"""

    with open(file_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()


def write_jsonl(file_path, records):
    """以指定的 records 覆寫 JSONL 檔案。"""

    with open(file_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_query(datapoint):
    """
    將 Context、Question、Options 組合成一段文字，
    傳入 Dify Chatflow 的 query。
    """
    context = datapoint["CONTEXT"]
    question = datapoint["QUESTION"]
    options = datapoint["OPTIONS"]

    if isinstance(options, dict):
        option_lines = [
            f"{letter}. {options[letter]}"
            for letter in ["A", "B", "C", "D"]
        ]
    else:
        if len(options) != 4:
            raise ValueError(f"預期有 4 個選項，實際取得 {len(options)} 個。")

        option_lines = [
            f"{letter}. {option}"
            for letter, option in zip(
                ["A", "B", "C", "D"], options
            )
        ]

    query = (
        #"Read the following ORQA problem carefully.\n\n"
        "=== CONTEXT ===\n"
        f"{context}\n\n"
        "=== QUESTION ===\n"
        f"{question}\n\n"
        "=== OPTIONS ===\n"
        + "\n".join(option_lines)
    )

    return query


def call_dify(query, user_id):
    """
    呼叫已發布的 Dify Chatflow。

    回傳 Dify API 的 JSON response。
    """
    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": {},
        "query": query,
        "response_mode": "blocking",
        "user": user_id,
    }

    response = requests.post(
        DIFY_API_URL,
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        try:
            error_detail = response.json()
        except ValueError:
            error_detail = response.text
        raise requests.exceptions.HTTPError(
            f"Dify API 回傳 HTTP {response.status_code}：{error_detail}"
        ) from exc
    
    result = response.json()

    # Dify 即使回傳 HTTP 200，也可能在 JSON 中包含錯誤
    if result.get("status") == "failed":
        raise RuntimeError(
            result.get("message", "Dify workflow execution failed.")
        )

    answer = result["answer"]
    if not isinstance(answer, str):
        raise ValueError("Dify API 回傳的 answer 欄位不是有效的文字。")


    return result, answer

def extract_between(text, start_marker, end_marker):
    """擷取兩個標記之間的文字。"""
    pattern = (
        re.escape(start_marker)
        + r"(.*?)"
        + re.escape(end_marker)
    )

    match = re.search(pattern, text, flags=re.DOTALL)

    if not match:
        raise ValueError(
            f"找不到輸出標記：{start_marker} / {end_marker}"
        )

    return match.group(1).strip()

def parse_retrieval_chunks(answer):
    """
    從 Dify 的直接回覆中擷取 retrieval chunks。

    回傳格式：
    {
        "retrieval_chunk_1": {
            "title": "...",
            "content": "..."
        },
        "retrieval_chunk_2": {...},
        "retrieval_chunk_3": {...}
    }

    如果某個 chunk 不存在，該欄位會是 None。
    """
    retrieval_text = extract_between(
        answer,
        "=== RETRIEVAL_KNOWLEDGE_START ===",
        "=== RETRIEVAL_KNOWLEDGE_END ===",
    )

    chunks = {
        f"retrieval_chunk_{i}": None
        for i in range(1, 4)
    }

    # 依照 CHUNK 1、CHUNK 2、CHUNK 3 分段
    chunk_pattern = re.compile(
        r"===\s*CHUNK\s+(\d+)\s*===\s*"
        r"(.*?)"
        r"(?=\n\s*===\s*CHUNK\s+\d+\s*===|\Z)",
        flags=re.DOTALL,
    )

    for match in chunk_pattern.finditer(retrieval_text):
        chunk_number = int(match.group(1))

        # 目前只保存前三個 chunk
        if chunk_number not in (1, 2, 3):
            continue

        chunk_text = match.group(2).strip()

        title_match = re.search(
            r"(?m)^\s*Title:\s*(.*?)\s*$",
            chunk_text,
        )

        content_match = re.search(
            r"Content:\s*\n(.*)",
            chunk_text,
            flags=re.DOTALL,
        )

        title = title_match.group(1).strip() if title_match else None
        content = content_match.group(1).strip() if content_match else None

        chunks[f"retrieval_chunk_{chunk_number}"] = {
            "title": title,
            "content": content,
        }

    return chunks


def parse_dify_answer(answer):
    """
    從 Dify 的 answer 中分別擷取：
    1. analysis：完整的分析內容
    2. raw_response：LLM response 的原始輸出
    3. prediction：將 A/B/C/D 轉換為 0/1/2/3
    4. retrieval_chunk_1 ~ retrieval_chunk_3
    """
    raw_response = extract_between(
        answer,
        "=== FINAL_ANSWER_START ===",
        "=== FINAL_ANSWER_END ===",
    )

    analysis = extract_between(
        answer,
        "=== ANALYSIS_START ===",
        "=== ANALYSIS_END ===",
    )
    # 嚴格要求模型只輸出一個選項字母，不從長文字中猜答案
    normalized_response = raw_response.strip().upper()

    prediction = ANSWER_MAPPING.get(normalized_response, -1)
    retrieval_chunks = parse_retrieval_chunks(answer)

    return {
        "raw_response": raw_response,
        "analysis": analysis,
        "prediction": prediction,
        **retrieval_chunks,
    }

def test_one_question(datapoint, fallback_index):
    """呼叫 Dify 並建立一筆完整的預測紀錄。"""

    index = datapoint.get("index", fallback_index)

    record = {
        "index": index,
        "context": datapoint.get("CONTEXT"),
        "question": datapoint.get("QUESTION"),
        "options": datapoint.get("OPTIONS"),
        "analysis": None,
        "raw_response": None,
        "prediction": -1,
        "target_answer": datapoint.get("TARGET_ANSWER"),
        "retrieval_chunk_1": None,
        "retrieval_chunk_2": None,
        "retrieval_chunk_3": None,
        "status": "pending",
        "error": None,
        "dify_answer": None,
        "dify_task_id": None,
        "elapsed_seconds": None,
    }

    start_time = time.perf_counter()

    try:
        query = build_query(datapoint)

        api_result, dify_answer = call_dify(
            query=query,
            user_id=f"orqa-dify-{index}",
        )

        # 保留 Dify 直接回覆的完整文字，方便除錯
        record["dify_answer"] = dify_answer
        record["dify_task_id"] = api_result.get("task_id")

        parsed = parse_dify_answer(dify_answer)

        record["analysis"] = parsed["analysis"]
        record["raw_response"] = parsed["raw_response"]
        record["prediction"] = parsed["prediction"]

        record["retrieval_chunk_1"] = parsed["retrieval_chunk_1"]
        record["retrieval_chunk_2"] = parsed["retrieval_chunk_2"]
        record["retrieval_chunk_3"] = parsed["retrieval_chunk_3"]

        if record["prediction"] == -1:
            raise ValueError(
                "無法將 raw_response 轉換成 0/1/2/3："
                f"{record['raw_response']!r}"
            )

        record["status"] = "success"

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"無法連線至 Dify：{DIFY_API_URL}\n"
            "請確認 Docker 中的 Dify 已啟動，且 localhost 網址與埠號正確。"
        ) from exc

    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"呼叫 Dify API 超過 {REQUEST_TIMEOUT} 秒仍未完成。\n"
            "請檢查 Dify 工作流、模型服務及網路連線。"
        ) from exc
    
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"

    finally:
        record["elapsed_seconds"] = round(
            time.perf_counter() - start_time,
            3,
        )

    return record

def evaluate_results(records):
    """只評估成功且標準答案可轉換的紀錄。"""

    valid_records = []
    invalid_targets = 0

    for record in records:
        if record.get("status") != "success":
            continue

        target = record.get("target_answer")

        if target is None:
            invalid_targets += 1
            continue

        valid_records.append((record, target))

    if not valid_records:
        print("沒有可用的預測紀錄，無法計算 accuracy。")
        return

    correct = sum(
        record["prediction"] == target
        for record, target in valid_records
    )

    accuracy = correct / len(valid_records)

    print("\n========== Evaluation ==========")
    print(f"可評估題數：{len(valid_records)}")
    print(f"答對題數：{correct}")
    print(f"Accuracy：{accuracy:.4f}")
    print(f"無法轉換標準答案的紀錄：{invalid_targets}")
    print("================================\n")

def save_wrong_predictions(records):
    """將預測錯誤的題目寫入 wrong_predictions.jsonl。"""

    wrong_records = []

    for record in records:
        if record.get("status") != "success":
            continue

        target = record.get("target_answer")
        if target is None:
            continue

        if record.get("prediction") != target:
            wrong_records.append({
                "index": record.get("index"),
                "question": record.get("question"),
                "context": record.get("context"),
                "options": record.get("options"),
                "prediction": record.get("prediction"),
                "raw_response": record.get("raw_response"),
                "target_answer": record.get("target_answer"),
                "analysis": record.get("analysis"),
                "retrieval_chunk_1": record.get("retrieval_chunk_1"),
                "retrieval_chunk_2": record.get("retrieval_chunk_2"),
                "retrieval_chunk_3": record.get("retrieval_chunk_3"),
            })

    write_jsonl(WRONG_FILE, wrong_records)
    print(f"錯誤題目數：{len(wrong_records)}")
    print(f"錯誤題目紀錄：{WRONG_FILE.resolve()}")


def main():
    if not DIFY_API_KEY:
        raise RuntimeError("找不到 DIFY_API_KEY 環境變數。")

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"找不到 ORQA 資料集：{TEST_DATA_PATH}\n")

    dataset = load_jsonl(TEST_DATA_PATH)

    if not dataset:
        raise ValueError("ORQA 資料集沒有可測試的題目。")

    test_dataset = (
        dataset if TEST_LIMIT is None
        else dataset[:TEST_LIMIT]
    )

    # 讀取既有紀錄，供 Resume 與最終評估使用
    existing_records = read_jsonl_records(PREDICTION_FILE)

    # 只跳過已成功完成的題目；失敗題目會在下次重新嘗試
    completed_indices = {
        record["index"]
        for record in existing_records
        if record.get("status") == "success"
    }

    print(f"資料集題數：{len(dataset)}")
    print(f"本次測試範圍：{len(test_dataset)} 題")
    print(f"輸入資料：{TEST_DATA_PATH.resolve()}")
    print(f"預測結果：{PREDICTION_FILE.resolve()}")
    print(f"已完成題數：{len(completed_indices)}")
    print("-" * 60)

    success_count = 0
    error_count = 0

    for fallback_index, datapoint in enumerate(
        tqdm(test_dataset, desc="Dify RAG", unit="question")
    ):
        index = datapoint.get("index", fallback_index)

        if index in completed_indices:
            continue

        try:
            record = test_one_question(
                datapoint=datapoint,
                fallback_index=fallback_index,
            )
        except RuntimeError as exc:
            print("DIFY API ERROR:", exc)
            raise SystemExit(1)

        # 每完成一題就追加至 JSONL
        append_jsonl(PREDICTION_FILE, record)

        if record["status"] == "success":
            success_count += 1
            """
            print(
                f"Index={record['index']} | "
                f"Raw={record['raw_response']!r} | "
                f"Prediction={record['prediction']} | "
                f"Target={record['target_answer']} | "
                f"Time={record['elapsed_seconds']}s"
            )
            """
        else:
            error_count += 1

            # 錯誤紀錄獨立儲存，方便檢查與重跑
            append_jsonl(INVALID_FILE, record)

            #print(
            #    f"Index={record['index']} | "
            #    f"ERROR: {record['error']}"
            #)

        time.sleep(1)

    # 同一題可能有多筆紀錄（例如之前失敗、後來重試成功）。
    # 以最後一筆紀錄為準。
    all_records = read_jsonl_records(PREDICTION_FILE)
    latest_by_index = {}

    for record in all_records:
        latest_by_index[record["index"]] = record

    final_records = [
        latest_by_index[datapoint.get("index", idx)]
        for idx, datapoint in enumerate(test_dataset)
        if datapoint.get("index", idx) in latest_by_index
    ]

    save_wrong_predictions(final_records)
    evaluate_results(final_records)

    print(f"本次新增成功：{success_count}")
    print(f"本次新增失敗：{error_count}")
    print(f"預測結果：{PREDICTION_FILE.resolve()}")
    print(f"失敗紀錄：{INVALID_FILE.resolve()}")



if __name__ == "__main__":
    main()