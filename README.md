# local-agent

Agent cá nhân chạy cục bộ: não mặc định là Ollama (`qwen2.5:7b`; đổi qua slot sang OpenAI/Anthropic), bộ nhớ là SQLite, công cụ gồm file (chỉ đọc), tùy chọn MCP `drawthings` và tùy chọn `ask_claude`.

## Cấu trúc

- `agent.py` — vòng lặp chính (`run`) và CLI.
- `brain.py` — client có tool calling: `OllamaBrain` (`/api/chat`), `OpenAIBrain`, `AnthropicBrain` (timeout mặc định 300s).
- `config.py` — đọc `config.toml` (slot provider, `limits`), tạo brain theo slot.
- `rules.py` — nạp file luật vào system prompt.
- `claude_cli.py` — cầu nối chỉ đọc tới `claude -p` (đăng nhập subscription).
- `git_evidence.py` — kiểm commit đã push cho ADR.
- `memory.py` — SQLite: profile, bài học (lessons, FTS), lịch sử run/step.
- `tools.py` — `ToolBox`: `search_lessons`, `add_lesson`, `list_dir`, `read_file`, `finish`, `add_node`, `link_nodes`, `find_nodes`, `get_node`, `add_adr`, `ask_claude` (chỉ khi `--with-claude`) + tool MCP cho phép (`txt2img`, `send_telegram_image`, `ask_choice`).
- `mcp_client.py` — client MCP stdio tối thiểu.
- `lessons.py` — CLI duyệt bài học.
- `distill.py` — duyệt node và chưng cất node đã duyệt ra markdown.
- `fake_mcp_server.py`, `test_*.py` — test (`python3 -m unittest discover`).

## Chạy

```bash
python3 agent.py "mục tiêu" [--db PATH] [--root DIR] [--max-steps 8] [--with-drawthings] [--with-claude] [--no-distill] [--distill-dir DIR]
```

`--with-drawthings` khởi động `drawthings_mcp.py` (đổi lệnh bằng `AGENT_DRAWTHINGS_CMD`). Chạy loại này sẽ sinh ảnh thật và gửi tới Telegram.

`--with-claude` bật `ask_claude`: gọi `claude -p` chỉ đọc (`Read,Grep,Glob`), tối đa 3 lần/phiên, bỏ `ANTHROPIC_API_KEY` khỏi môi trường để dùng đăng nhập subscription; kết quả coi là dữ liệu không tin cậy.

Biến môi trường: `OLLAMA_HOST`, `AGENT_MODEL`, `AGENT_DB` (mặc định `~/.local-agent/memory.db`), `AGENT_CONFIG`, `AGENT_DISTILL_DIR`, `AGENT_DRAWTHINGS_CMD`.

## Cấu hình (`~/.local-agent/config.toml`)

```toml
[slots.main]
provider = "ollama"            # ollama | openai | anthropic (claude là bí danh)
model = "qwen2.5:7b"
# host = "..."
# api_key_env = "ANTHROPIC_API_KEY"   # tên biến môi trường; `api_key` trực tiếp bị từ chối

[limits]
warn_tokens = 100000           # số nguyên dương
# max_tokens = 120000          # tùy chọn: chặn cứng, phải >= warn_tokens
# min_context = 32768          # tùy chọn: mọi slot đang dùng phải khai báo `context` >= mức này
```

- Không có file → mặc định Ollama. File lỗi TOML/không đọc được/slot hoặc khóa sai → in `error:` và thoát mã 2.
- `main.fallback = "<tên slot>"` (slot phải tồn tại, khác `main`): main lỗi lần 1 thì thử lại bước đó, lỗi lần 2 liên tiếp thì chuyển sang slot fallback cho phần còn lại của phiên, ghi lý do vào `runs.route`. Không khai báo thì lỗi brain kết thúc phiên như cũ.
- Slot `aux` (nếu khai báo): sau phiên, aux đặt tiêu đề ngắn cho run (`runs.title`), best-effort, lỗi bị bỏ qua.
- `warn_tokens`: chỉ cảnh báo (stderr) một lần khi ước lượng token tích lũy vượt ngưỡng (ước lượng theo 3 ký tự/token, không phải số thật); `max_tokens` (nếu đặt) chặn cứng: chạm ngưỡng thì không gọi model nữa, phiên kết thúc `error: context limit reached`. `context` (khai báo ở slot) do bạn tự ghi, agent không đo; `max_tokens` không được lớn hơn `context`.

## Luật người dùng

`rules.py` nạp `~/.local-agent/AGENT.md` và `<root>/AGENT.md`; nếu không có cả hai thì dùng `AGENTS.md` (ollama/openai) hoặc `CLAUDE.md` (anthropic/claude) ở `--root`. File trong `--root` phải nằm trong root. Cắt ở 6000 ký tự. Luật không ghi đè giới hạn an toàn hay yêu cầu xác nhận hành động bên ngoài.

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
- `aux` mới dùng để đặt tên run (chưa nén ngữ cảnh); fallback/aux chưa chạy thật với model.
- Chưa đối chiếu ước lượng token với số thật; chưa kiểm `temperature` cho Anthropic; chưa đo tỉ lệ model 7B gọi tool sai.
- Chưa test đủ đường thoát lỗi config; chưa cài `gitleaks`.
