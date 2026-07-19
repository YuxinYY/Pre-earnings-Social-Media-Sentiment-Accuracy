import zstandard as zstd
import json
import io
import os
import time


def open_text_stream(input_path):
    """
    自动检测文件类型：
    - zstd 压缩文件（魔术头 28 b5 2f fd）用 ZstdDecompressor 流式解压
    - 普通文本/ndjson 文件直接打开
    """
    with open(input_path, 'rb') as fh:
        magic = fh.read(4)

    is_zstd = magic == b'\x28\xb5\x2f\xfd'

    if is_zstd:
        print(f"[Detect] {input_path} is zstd compressed")
        fh = open(input_path, 'rb')
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        stream_reader = dctx.stream_reader(fh)
        return io.TextIOWrapper(stream_reader, encoding='utf-8'), fh
    else:
        print(f"[Detect] {input_path} is plain text / ndjson")
        fh = open(input_path, 'r', encoding='utf-8')
        return fh, fh


def stream_filter(input_path, output_path, start_ts, end_ts, progress_interval=1_000_000):
    # 1. 检查输入文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            f"Please place the dataset at this path or update the path in the script."
        )

    # 2. 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    input_size_gb = os.path.getsize(input_path) / (1024 ** 3)
    print(f"[Start] Filtering {input_path}")
    print(f"[Start] Input size: {input_size_gb:.2f} GB")
    print(f"[Start] Time range: {start_ts} to {end_ts}")
    print(f"[Start] Output will be written to: {output_path}")

    total_lines = 0
    written_lines = 0
    start_time = time.time()

    text_stream, file_handle = open_text_stream(input_path)

    with text_stream:
        with open(output_path, 'w', encoding='utf-8') as out:
            try:
                for line in text_stream:
                    total_lines += 1

                    # 每处理 progress_interval 行打印一次进度
                    if total_lines % progress_interval == 0:
                        elapsed = time.time() - start_time
                        print(
                            f"[Progress] Processed {total_lines:,} lines, "
                            f"written {written_lines:,} matching records "
                            f"({elapsed:.1f}s elapsed)"
                        )

                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    created = obj.get('created_utc')
                    if created and start_ts <= int(created) <= end_ts:
                        out.write(line)
                        written_lines += 1

            except KeyboardInterrupt:
                print("\n[Interrupted] Stopping early due to user interrupt.")

    # 关闭底层文件句柄
    file_handle.close()

    elapsed = time.time() - start_time
    print(f"[Done] Processed {total_lines:,} lines in total")
    print(f"[Done] Wrote {written_lines:,} matching records to {output_path}")
    print(f"[Done] Time elapsed: {elapsed:.1f}s")

    if written_lines == 0:
        print("[Warning] No records matched the time range. Check your start_ts/end_ts.")

    print("-" * 60)


if __name__ == "__main__":
    # 2021-01-01 00:00:00 UTC 到 2021-12-31 23:59:59 UTC
    start_ts = 1609459200
    end_ts = 1640995199

    # 处理 comments（zstd 压缩，8 GB）
    # stream_filter(
    #     'data_processing/reddit/subreddits25/wallstreetbets_comments.zst',
    #     'data_processing/reddit/wallstreetbets_comments_2021.ndjson',
    #     start_ts=start_ts,
    #     end_ts=end_ts
    # )

    # 处理 submissions（纯文本 ndjson，36 GB）
    stream_filter(
        'data_processing/reddit/subreddits25/wallstreetbets_submissions.zst',
        'data_processing/reddit/wallstreetbets_submissions_2021.ndjson',
        start_ts=start_ts,
        end_ts=end_ts
    )
