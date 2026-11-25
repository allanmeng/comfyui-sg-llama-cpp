中文 | [English](README_EN.md)

# comfyui-sg-llama-cpp

在 ComfyUI 里跑 GGUF 语言模型与视觉模型。针对 **Intel Arc / Iris Xe（SYCL）** 调优，并内置一条
**批量打标流水线**，直接产出 LoRA 数据集。

![节点预览](assets/node_preview.png)

这个插件干两件事：

| 用途 | 接线 | 典型场景 |
|---|---|---|
| 单图推理 | Model Loader → Options → Engine | 描述图片、改写提示词、就一张图提问 |
| 批量打标 | Image Folder Loader → Engine → Caption Saver | 每张图一份 `.txt`，喂给 LoRA 训练 |

两者共用同一个引擎、同一份 GGUF，**都不依赖任何第三方节点**。

---

## 目录

- [环境要求](#环境要求)
- [安装](#安装)
- [快速上手](#快速上手)
- [节点参考](#节点参考)
- [模型目录](#模型目录)
- [批量打标的实践细节](#批量打标的实践细节)
- [B580 实测数据](#b580-实测数据)
- [故障排查](#故障排查)
- [本分支相对上游改了什么](#本分支相对上游改了什么)
- [许可与链接](#许可与链接)

---

## 环境要求

| 组件 | 要求 | 说明 |
|---|---|---|
| ComfyUI | 包含 `comfy_api.latest` 的版本 | 本插件使用 V3 schema（`define_schema`），不是旧的 `INPUT_TYPES` |
| llama-cpp-python | **>= 0.3.39** | 0.3.39+ 是硬性要求（MTMD 重写）；0.4.x 额外解锁 `load_mode`、KV 量化和 `reasoning_budget` |
| 显卡后端 | Intel Arc 用 SYCL wheel | JamePeng 的 release **不含** SYCL 构建，见下文 |

插件**从函数签名而不是版本号**判断已安装的 `llama-cpp-python` 支持什么，不认识的参数会被丢弃。
所以 0.3.x 和 0.4.x 共用一份代码，用不上的控件只是不起作用，不会报错。

## 安装

### 1. 装 wheel

**Intel GPU（SYCL）** —— 预编译 Windows wheel：

```
https://github.com/allanmeng/llama-cpp-python-sycl-windows
```

**其他后端（CUDA / CPU / Vulkan）：**

```
https://github.com/JamePeng/llama-cpp-python/releases
```

### 2. 克隆插件

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/allanmeng/comfyui-sg-llama-cpp
```

### 3. 重启 ComfyUI

节点出现在 `LlamaCPP` 分类下（打标那一对在 `LlamaCPP/Dataset`）。

### 从 llama-cpp-python 0.3.38 或更早版本升级

**不要**用 `pip install --upgrade`。旧版本残留在 `lib/` 里的 DLL 会和新版本冲突：

```bat
pip uninstall llama-cpp-python -y
pip install llama_cpp_python-<版本>-<你的构建>.whl
```

> 在 Windows 上，0.3.39+ 的 MTMD 重写把视觉处理搬进了 `GenericMTMDChatHandler`，并用
> `mmproj_path` 取代了 `clip_model_path`。不先卸载、直接覆盖安装到 0.3.38 之上，是
> “节点能加载但视觉输出全是乱码”最常见的原因。

### 4. 可选配置

在 `__init__.py` 同级放一个 `config.json`，可以追加模型目录、过滤无关 GGUF —— 见
[模型目录](#模型目录)。不放也能正常工作。

---

## 快速上手

### 描述单张图片

```
Llama CPP Model Loader ── MODEL ────┐
   model_name      = <你的模型>.gguf
   mmproj_model_name = <你的 mmproj>.gguf   ← 要读图必须接
                                     │
Llama CPP Options ────── OPTIONS ────┤
   n_ctx = 8192                      │
                                     ▼
                              Llama CPP Engine ── RESPONSE ──► (Show Text / 预览)
                                 images       = Load Image
                                 prompt       = "详细描述这张图片。"
                                 system_prompt = ""
```

仓库里的 `demo_workflow_llama-cpp-python_0.3.39+.json` 就是这张图。它用了
[ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts) 的
`ShowText|pysssss`，想直接加载就一起装，并把里面的 `Load Image` 换成你自己的图。

**真正影响结果的四个设置**

| 设置 | 位置 | 原因 |
|---|---|---|
| `mmproj_model_name` | Model Loader | 不接就只跑纯文本，`images` 会被忽略 |
| `n_ctx = 8192` | Options | 一张大图的视觉 token 就有 4000+，旧的 `2048` 默认值装不下 |
| `max_tokens = 2048` | Engine | 详细描述轻松超过 500 token，推理模型还会先花预算思考 |
| `think_display = hide` | Engine | 剥掉 `<think>` 块，让 `RESPONSE` 是答案而不是自说自话 |

`chat_format` 保持默认即可。视觉模型的处理方式是根据 `mmproj` 文件自动选的，这个下拉框只对
模板特殊的纯文本模型有意义。

### 批量打标一个目录

```
Llama CPP Image Folder Loader ── images ──┐
   folder_path = 图片目录                   │
   max_side    = 1024                      ├──► Llama CPP Engine ── RESPONSE ──► Llama CPP Caption Saver
   max_images  = 0  (0 = 全部)             │       （每张图执行一次，            （每张图写一份 .txt，
                              └─ paths  ───┘        各产出一条标注）              默认写在原图旁边）
```

正式跑之前先设 `max_images = 3` 验证一遍。推荐参数见
[批量打标的实践细节](#批量打标的实践细节)。

---

## 节点参考

### Llama CPP Model Loader

`LlamaCPP` · 输出 `MODEL`

| 输入 | 必填 | 默认 | 说明 |
|---|:--:|---|---|
| `model_name` | 是 | — | [扫描](#模型目录)到的所有 `.gguf`，排除文件名含 `mmproj` 的 |
| `chat_format` | 否 | `llama-2` | 只对纯文本模型生效；视觉模型的 handler 由 mmproj 决定 |
| `mmproj_model_name` | 否 | `None` | 文件名含 `mmproj` 的文件。选 `None` = 纯文本 |

### Llama CPP Options

`LlamaCPP` · 输出 `OPTIONS`

**基础**

| 输入 | 默认 | 说明 |
|---|---|---|
| `n_gpu_layers` | `-1` | `-1` = 全部层放显卡 |
| `n_ctx` | `8192` | 大图或长提示词要往上加。`8192` 是实测可用下限 |
| `n_threads` / `n_threads_batch` | `-1` | `-1` = 自动 |
| `n_batch` / `n_ubatch` | `2048` / `512` | `n_batch` 加大能加快 prefill，代价是内存 |
| `main_gpu` | `0` | 设备序号 |
| `offload_kqv` | 开 | K/Q/V 留在显卡 |
| `numa` | 关 | NUMA 亲和（多路主机） |
| `verbose` | 关 | 会把每个张量、每个 token 都打到控制台。开启时会回显最终生效的调优参数 |

**视觉**

| 输入 | 默认 | 说明 |
|---|---|---|
| `vision_use_gpu` | 开 | 视觉编码器放显卡 |
| `vision_image_min_tokens` | `1024` | Qwen-VL 的最低要求。`-1` = 用 handler 默认 |
| `vision_image_max_tokens` | `-1` | `-1` = 用 handler 默认。必须是 `-1` 或 `>= min_tokens`；数值矛盾时丢弃并告警，不会中断整批 |

**混合 / 循环模型（Qwen3.5 之类）**

| 输入 | 默认 | 说明 |
|---|---|---|
| `ctx_checkpoints` | `-1` | `-1` = 用 llama-cpp-python 默认值（16）。**不要用 `0`** —— 会走一条有 bug 的快路径，大图直接崩 |
| `checkpoint_interval` | `4096` | checkpoint 的 token 间隔 |
| `checkpoint_on_device` | 关 | checkpoint 负载放显存（每个约 50 MiB）。实测不影响吞吐，关着就行 |

**加载策略**（仅 `llama-cpp-python` 0.4.x；0.3.x 自动忽略）

| 输入 | 默认 | 说明 |
|---|---|---|
| `use_mmap` / `use_mlock` / `use_direct_io` | 开 / 关 / 关 | **[0.4.x 已废弃]** 0.4.x 的加载路径里已经没有它们。0.3.x 仍然生效。0.4.x 请用 `load_mode` |
| `load_mode` | `auto` | `auto` / `none` / `mmap` / `mlock` / `mmap_mlock` / `direct_io` |
| `lazy_mode` | `auto` | `auto` / `off` / `on`，依赖 mmap |
| `flash_attn_type` | `auto` | 主模型的 Flash Attention：`auto` / `disabled` / `enabled` |
| `vision_flash_attn` | `auto` | 视觉编码器内的 Flash Attention：`auto` / `enabled` / `disabled` |
| `type_k` / `type_v` | `auto` | KV cache 数据类型：`auto` / `f16` / `bf16` / `q8_0` / `q5_1` / `q5_0` / `q4_1` / `q4_0` / `iq4_nl`。`auto` = f16。**量化 V 必须开 Flash Attention**，与 `flash_attn_type=disabled` 的组合会在加载前被拒绝。开之前先看[实测结论](#b580-实测数据) |

> 0.4.x 的这些输入是**故意追加在参数列表末尾**的。ComfyUI 按位置存储工作流的控件值，
> 插在中间会让升级前保存的每个工作流的值整体错位。

### Llama CPP Engine

`LlamaCPP` · 输出 `RESPONSE`

**生成**

| 输入 | 默认 | 说明 |
|---|---|---|
| `model` / `prompt` | — | 必填 |
| `images` | — | 单张或一批 |
| `options` | — | 来自 Options 节点 |
| `system_prompt` | 空 | |
| `response_format` | `text` | `text` 或 `json_object`。选 `json_object` 会展开 `json_schema` 字段，把输出约束成合法 JSON |
| `max_tokens` | `2048` | 答案被截断就往上加 |
| `memory_cleanup` | `full_cleanup` | `close` / `backend_free` / `full_cleanup` / `persistent`，见[显存策略](#显存策略) |
| `think_display` | `hide` | `hide` 只返回最后一个 `</think>` 之后的正文；`show` 保留整个思考块。能处理畸形标签（缺失开标签、重复闭标签、悬空开标签） |
| `reasoning_budget` | `-1` | 仅 0.4.x。限制第一个 `<think>` 块的 token 数：`-1` = 库默认，`0` = 立刻收尾，`N` = 最多 N 个思考 token |

**采样**

| 输入 | 默认 | | 输入 | 默认 |
|---|---|---|---|---|
| `temperature` | `0.1` | | `repeat_penalty` | `1.1` |
| `top_p` | `0.9` | | `present_penalty` | `0.0` |
| `top_k` | `20` | | `frequency_penalty` | `0.0` |
| `min_p` | `0.05` | | `seed` | `1`（带生成后控制） |

### Llama CPP Memory Cleanup

`LlamaCPP` · 输出 `PASSTHROUGH`（原样透传输入）

| 输入 | 默认 | 说明 |
|---|---|---|
| `memory_cleanup` | `close` | `close` / `backend_free` / `full_cleanup` / `persistent` |
| `passthrough` | — | 可选，用来把本节点串进已有链条而不打断连线 |

### Llama CPP Image Folder Loader

`LlamaCPP/Dataset` · 输出 `images`（列表）、`paths`（列表）、`count`

输出的是**列表**，一个文件一项。这是关键：下游没有 `INPUT_IS_LIST` 的节点因此会对每张图各执行
一次。反过来，如果目录加载器输出的是**张量批**，N 张图会被塞进同一条请求，第一次就因为上下文
不足而失败。

| 输入 | 默认 | 说明 |
|---|---|---|
| `folder_path` | 空 | 绝对路径原样使用；相对路径解析到 `ComfyUI/input` 下 |
| `extensions` | `png,jpg,jpeg,webp,bmp` | 逗号分隔 |
| `recursive` | 开 | 扫子目录 |
| `sort_order` | `natural` | `natural`（2 排在 10 前）/ `name` / `modified` / `none` |
| `max_images` | `0` | `0` = 全部。先设小值验证工作流 |
| `start_index` | `0` | 排序后跳过前 N 张，用于中断后续跑 |
| `max_side` | `0` | 长边超过 N 就等比缩小（`0` = 保持原尺寸）。这是降低视觉 token 数和 base64 体积最省事的手段，打标场景 `1024` 是个好默认值 |

图片会按 EXIF 转正，透明像素合成到白色（不是黑色）。无法解码的文件跳过并告警，不会中断整批。

### Llama CPP Caption Saver

`LlamaCPP/Dataset` · 输出 `saved`、`skipped`

每份标注写一个 `.txt`，默认落在**原图同目录、同文件名**（`abc.png` → `abc.txt`），这正是 kohya
系训练器期望的配对方式。图片不会被重新编码，也不会被改名。

| 输入 | 默认 | 说明 |
|---|---|---|
| `text` | — | 引擎的 `RESPONSE` |
| `paths` | — | 加载器的 `paths`。不接则改为在 `output_dir` 里写编号文件（`caption_00000.txt` …） |
| `output_dir` | 空 | 空 = 写在每张原图旁边；相对路径解析到 `ComfyUI/input` 下 |
| `skip_on_error` | 开 | 丢弃内容为引擎报错串（`Error generating response: ...`）的标注。不开的话，某张图失败时错误信息会被当成标注写进数据集 |
| `skip_empty` | 开 | 丢弃空标注 |

文件以 UTF-8、`\n` 换行、无 BOM 写入，每次写入都会回显，便于跑完复盘：

```
[llama_cpp dataset] caption saved: ...\ComfyUI_1057334871997372_.txt (223 chars)
[llama_cpp dataset] captions: 3 saved, 0 skipped
```

---

## 模型目录

以下目录中的 `.gguf` 会被**递归**扫描：

| 来源 | 路径 | 自动 |
|---|---|:--:|
| ComfyUI 文本编码器 | `ComfyUI/models/text_encoders/` 和 `ComfyUI/models/clip/` | 是 |
| 专用 LLM 目录 | `ComfyUI/models/LLM/` | 是 |
| 自定义目录 | `config.json` 里写的任意路径 | 否 |

扫到的文件再按文件名分进两个下拉框：

- **模型下拉框** —— 文件名不含 `mmproj` 的所有 `.gguf`
- **mmproj 下拉框** —— 文件名含 `mmproj` 的所有 `.gguf`

`mmproj` 文件放在哪个子目录都行，只看文件名里有没有这个关键词。

### 推荐布局

```
ComfyUI/models/LLM/
├── Qwen3.5-4B-Q4_K_M.gguf
├── Qwen3.5-4B-mmproj-BF16.gguf
├── Qwen2.5-VL-3B-instruct-q4_k_m.gguf
└── mmproj/
    └── Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf
```

### config.json

可选。没有它时只扫 ComfyUI 自己的模型目录。

```json
{
  "model_folders": [
    "D:\\AI\\LLM\\models",
    "/home/user/models"
  ],
  "exclude_patterns": [
    "wan22",
    "krea2",
    "qwen_image"
  ]
}
```

- 绝对路径和相对路径都行；不存在的路径会被静默过滤。
- `model_folders` 是在自动目录**之外追加**，不是替换。
- `exclude_patterns` 是一组**大小写不敏感的子串**，匹配模型相对于其搜索根目录的路径。命中的文件
  会同时从**模型和 mmproj 两个下拉框**里消失。典型用法：`model_folders` 指向了一个和扩散模型、
  视频模型共用的目录 —— llama.cpp 跑不了那些 GGUF，它们只会把列表拉长。

示例见 `config.example.json`。

---

## 批量打标的实践细节

### 显存策略

| `memory_cleanup` | 行为 | 批量打标 |
|---|---|---|
| `full_cleanup` | 每次生成后释放模型，下次重新加载 | **默认值。** 4B 级模型重载约 2 秒，远比其他方案便宜 |
| `persistent` | 模型常驻 | 在混合视觉模型上实测**慢约 26 倍**。混合（循环）架构下的前缀/KV 复用开销远大于它省下的加载时间。打标不要用 |
| `close` / `backend_free` | 更轻的清理 | 单次跑可以，批量不建议 |

加载器是 `not_idempotent` 的，所以往目录里加了图不用重启 ComfyUI。

### 单图耗时

每次生成都会打印一行，**与 `verbose` 无关**：

```
[llama_cpp] image timing | load 2.20s | generate 5.72s | total 7.93s | prompt 252 tok, out 60 tok (10.5 tok/s)
```

`verbose` 不适合用来测批量：llama.cpp 每张图都会打几百行 `find_slot` 和张量日志，而 ComfyUI
本身只对整轮给一句 `Prompt executed in X seconds`。这一行才是让「N 张图 × 单图成本」可见的东西。

### 标注卫生

- 一段话、一行、UTF-8、无 BOM。注意 `str.strip()` **不会**去掉 U+FEFF，所以只要 BOM 出现在响应里
  就会被原样写盘 —— tokenizer 抱怨第 0 位多了个 token 时，先查文件头字节。
- 标注的语言和句式，必须和你推理时写的提示词一致。训练用标签串、推理用长句，是 LoRA 训不出来
  最常见的原因之一。
- 看「写全了哪些维度」而不是「写了多少字」：每个**变量**属性（服装、背景、景别）都要写出来，
  否则模型只能把它绑到触发词上。

### 第一次跑的建议参数

| 设置 | 值 | 原因 |
|---|---|---|
| `max_images` | `3` | 先验证接线、读三条标注，再花时间跑整个目录 |
| `max_side` | `1024` | 2000 px 的图视觉 token 约是 1000 px 的 4 倍，标注质量并没有提升 |
| `n_ctx` | `8192` | 给一张大图的 prefill 加答案留余量 |
| `max_tokens` | `512`–`2048` | 太小会把答案截在半句；而被截断的思考块比短答案更糟，所以同时设 `reasoning_budget` |
| `reasoning_budget` | `256`–`512` | 防止思考吃掉整个预算。必须**明显小于** `max_tokens`，否则强制收尾的 `</think>` 永远到不了 |

如果图是刚生成的、想按生成顺序打标，把 `sort_order` 设成 `modified`。

---

## B580 实测数据

环境：Arc B580（12 GB，SYCL）、llama-cpp-python 0.4.3、Qwen3.5-4B-Uncensored Q4_K_M、
1792×2304 图片（4032 视觉 token）、`n_ctx=8192`。单机数据，看方向不要当规格。**没有在 Arc B70
或非 Intel 后端上测过。**

**0.4.x 的调优开关不带来提速。** `load_mode`、`lazy_mode`、`flash_attn_type`、
`vision_flash_attn` 对 `clip_encode`、prefill 和 decode 的影响都在运行间噪声之内（`clip_encode`
波动 ≤1.4%，每种组合都是约 11.0 s）。默认值保持 `auto`。暴露它们是为了让工作流**能固定**某种
行为，不是因为它们更快。

**KV cache 量化有效，但在小型混合模型上是净亏。**

| `type_k` / `type_v` | Decode | 相对 f16 |
|---|---|---|
| `auto` / `auto`（f16） | **85.59 t/s** | — |
| `q8_0` / `auto`（只量化 K） | 82.22 t/s | −3.9% |
| `q8_0` / `q8_0` | 79.22 t/s | −7.4% |
| `q4_0` / `q4_0` | 78.70 t/s | −8.0% |

只量化 K 的代价几乎正好是量化 K+V 的一半 —— 如果 decode 循环每个 token 都要反量化整段历史，
这就是应有的剂量关系，说明设置确实传到了 llama.cpp。它在这里仍然亏，是因为该模型的 32 层里只有
8 层带 KV（其余是循环层），f16 的 KV cache 总共才约 256 MiB；省 114 MiB 在 12 GB 卡上看不出来，
却要付出 7% 的 decode 速度。**保持 `auto`**，除非你跑大尺寸稠密 VLM 或极长上下文——那种情况下
cache 能到 GB 级。

两个容易误解的点：

- 日志里的 `SYCL0 compute buffer size` 是**计算图的临时缓冲**，不是 KV cache。KV 有独立分配，
  只有 `verbose=true` 时的 `llama_kv_cache: size = ... (type): ... MiB` 才会报告它。改 KV 类型
  永远不会让 compute buffer 那个数字变小。
- 类型的 block size 必须能整除 head 维度（32 能整除 Qwen3.5 的 256）。不支持的组合由 llama.cpp
  自己报错，不归插件管。

**重调过的默认值**（2.3.0）只对新建节点生效，已有工作流保留自己存的值。

| 节点 | 参数 | 旧 | 新 |
|---|---|---|---|
| Options | `n_ctx` | `2048` | `8192` |
| Engine | `memory_cleanup` | `close` | `full_cleanup` |
| Engine | `max_tokens` | `512` | `2048` |
| Engine | `temperature` / `top_p` / `top_k` | `0.2` / `0.95` / `40` | `0.1` / `0.9` / `20` |
| Engine | `think_display` | `show` | `hide` |

每一项的完整来龙去脉见 [CHANGELOG.md](CHANGELOG.md)。

---

## 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| 大图第一次 decode 就报 `failed to create memory slot for batch` / 崩溃 | `n_ctx` 装不下视觉 prefill（4000+ token）加第一个 decode 槽 | 把 `n_ctx` 提到 `8192` 或更高 |
| 混合模型（Qwen3.5）大图访问违规崩溃 | `ctx_checkpoints = 0` 会走 llama-cpp-python 里一条有 bug 的快路径 | 改回 `-1` |
| 改 `use_mmap` / `use_mlock` / `use_direct_io` 毫无反应 | 0.4.x 的加载路径里没有它们了 | 改用 `load_mode` |
| `RESPONSE` 是一段被截断的思考，没有答案 | `max_tokens` 全花在 `<think>` 里，`</think>` 从未生成，`hide` 无从剥离 | 提高 `max_tokens`，并把 `reasoning_budget` 设得明显更小 |
| 每个 `.txt` 里都是 `Error generating response: ...` | 引擎把报错串当正文返回，保存节点照写了 | 打开 `skip_on_error`（默认已开） |
| 加载时出现 `control-looking token: <id> '</s>' was not control-type; this is probably a bug in the model` | GGUF 的 `token_type` 元数据把 `</s>` 标成了普通 token，llama.cpp 覆盖类型并把它登记为 EOS | 无害。每次加载模型打印一次 —— 所以 3 张图配 `full_cleanup` 会打 3 次。换成官方转换脚本产出的 GGUF 通常就静默了 |
| `V cache quantization requires flash_attn` | 量化了 `type_v` 又同时 `flash_attn_type=disabled` | 开 Flash Attention，或把 `type_v` 留 `auto` |
| 下拉框里一个模型都没有 | 目录没被扫到，或全被 `exclude_patterns` 过滤了 | 见[模型目录](#模型目录)；用 `verbose` 看实际扫描了哪些根目录 |
| 升级后视觉输出全是乱码 | 新 wheel 覆盖装到旧版之上，`lib/` 里残留旧 DLL | 先卸载再装，见[安装](#安装) |

---

## 本分支相对上游改了什么

| 方面 | 上游 | 本分支 |
|---|---|---|
| llama-cpp-python | 0.3.2x 时代的视觉 handler | **0.3.39+ MTMD**：`GenericMTMDChatHandler`，`mmproj_path` 直接传给 `Llama()` |
| 参数传递 | 固定 kwarg 列表 | 按已安装的 `Llama()` 签名做白名单，0.3.x 与 0.4.x 共用一条代码路径 |
| 0.4.x 开关 | 未暴露 | `load_mode`、`lazy_mode`、`flash_attn_type`、`vision_flash_attn`、`type_k` / `type_v` |
| 思考输出 | 原样返回 | `think_display`（`hide` / `show`），能处理畸形标签；0.4.x 上另有 `reasoning_budget` |
| 默认值 | `n_ctx 2048`、`memory_cleanup close`、`max_tokens 512` | 在 B580 上重调为 `8192` / `full_cleanup` / `2048` |
| 视觉 token 边界 | 参数矛盾直接中断整批 | 丢弃非法边界并告警 |
| 模型目录 | 仅 `model_folders` | 增加 `exclude_patterns` |
| 测速 | 只能靠 `verbose` | 单图计时行，与 `verbose` 无关 |
| 批量打标 | 无 | `LlamaCPP/Dataset`：目录加载器 + 标注保存器 |

这个插件 fork 自 [sebagallo/comfyui-sg-llama-cpp](https://github.com/sebagallo/comfyui-sg-llama-cpp)，
上游适用的修复会持续跟进合并。欢迎反馈 issue 和改进 SYCL 路径的 PR。

---

## 许可与链接

采用 **GNU AGPLv3**，见 [LICENSE](LICENSE)。作为 AGPLv3 项目的 fork，本分支同样保持 AGPLv3。

| | |
|---|---|
| 本分支（SYCL 优化） | https://github.com/allanmeng/comfyui-sg-llama-cpp |
| Windows SYCL wheel | https://github.com/allanmeng/llama-cpp-python-sycl-windows |
| 其他 wheel（CUDA / CPU） | https://github.com/JamePeng/llama-cpp-python/releases |
| 上游插件 | https://github.com/sebagallo/comfyui-sg-llama-cpp |
| 变更记录 | [CHANGELOG.md](CHANGELOG.md) |
