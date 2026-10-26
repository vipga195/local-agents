# local-agent

Agent cá nhân chạy cục bộ: não là Ollama (`qwen2.5:7b`), bộ nhớ là SQLite, công cụ gồm file (chỉ đọc) và tùy chọn MCP `drawthings`.

## Cấu trúc

- `agent.py` — vòng lặp chính (`run`) và CLI.
- `brain.py` — client Ollama `/api/chat` có tool calling (timeout mặc định 300s).
- `memory.py` — SQLite: profile, bài học (lessons, FTS), lịch sử run/step.
- `tools.py` — `ToolBox`: `search_lessons`, `add_lesson`, `list_dir`, `read_file`, `finish`, `add_node`, `link_nodes`, `find_nodes`, `get_node`, `add_adr` + tool MCP cho phép (`txt2img`, `send_telegram_image`, `ask_choice`).
- `mcp_client.py` — client MCP stdio tối thiểu.
- `lessons.py` — CLI duyệt bài học.
- `distill.py` — duyệt node và chưng cất node đã duyệt ra markdown.
- `fake_mcp_server.py`, `test_*.py` — test (`python3 -m unittest discover`).

## Chạy

```bash
python3 agent.py "mục tiêu" [--db PATH] [--root DIR] [--max-steps 8] [--with-drawthings]
```

`--with-drawthings` khởi động `drawthings_mcp.py` (đổi lệnh bằng `AGENT_DRAWTHINGS_CMD`). Chạy loại này sẽ sinh ảnh thật và gửi tới Telegram.

Biến môi trường: `OLLAMA_HOST`, `AGENT_MODEL`, `AGENT_DB` (mặc định `~/.local-agent/memory.db`).

## Bài học

Bài học do agent ghi luôn ở trạng thái `proposed` (chưa đáng tin). Duyệt bằng:

```bash
python3 lessons.py list --status proposed
python3 lessons.py approve <id>
python3 lessons.py reject <id>
```

## Kho node và chưng cất

Tri thức mới (kịch bản, nhân vật, prompt ảnh) lưu thành node + cạnh trong SQLite (`nodes`, `edges`, có FTS). Agent dùng `find_nodes` trước khi tạo, `get_node` để đọc, `add_node`/`link_nodes` để ghi. Node agent tạo luôn ở trạng thái `proposed`.

```bash
python3 distill.py list --status proposed
python3 distill.py approve <id>      # hoặc reject <id>
python3 distill.py export --out DIR [--include-proposed] [--slim]
```

- `export` ghi `DIR/<type>/<slug>.md` (frontmatter có các cạnh, mục `## Data` chứa prompt) và `DIR/index.md`; từ chối ghi vào thư mục có thành phần tên `knowledge-base`.
- `--slim`: node `accepted` đã xuất được rút gọn trong SQLite (xóa `data`, ghi `distilled_path`, giữ tên/summary/cạnh) và `<db>.bak` được sao lưu trước. `get_node` đọc nội dung từ file chưng cất; `add_node` lên node đã rút gọn bị từ chối.
- Hook tự động: cuối mỗi phiên `agent.py` gọi chưng cất + rút gọn cho node `accepted` vào `~/.local-agent/distilled` (`--distill-dir` hoặc `AGENT_DISTILL_DIR`; tắt bằng `--no-distill`). Lỗi chưng cất chỉ cảnh báo, không đổi mã thoát.
- ADR: `add_adr` tạo node `adr`; chỉ xuất `adr/<ngày>-<slug>.md` (template KB) khi `accepted` và `git_evidence.py` thấy commit chạm `paths` đã push (PR lấy từ tiêu đề commit, không nhập tay).
- Độc lập với graph builder của KB; đưa file vào KB do người xem diff rồi tự commit.

## Trạng thái

Xong:
- Mốc 1: chạy thật 3/3.
- Mốc 2: chạy thật với `--with-drawthings`: sinh 2 ảnh, gửi Telegram, `ask_choice`, ghi bài học `proposed` (2026-10-26).

Còn lại / hạn chế:
- Tìm node đã rút gọn chỉ theo tên/summary (prompt đã dời sang file). Model hay đặt tên cạnh không thống nhất (`appears_in` / `has_character`).
- Model 7B hay đoán sai `model`/`sampler_name` và gọi nhiều tool trong một lượt; hiện chỉ giảm bằng `DRAWTHINGS_PROMPT`. Sửa chắc hơn: chỉ thực thi tool call đầu tiên mỗi lượt, hoặc từ chối `ask_choice` khi chưa gửi ảnh.
- Bài học tự sinh có thể khái quát quá mức từ một lần chọn; cần duyệt tay.
- Chưa `git init`, chưa commit.
