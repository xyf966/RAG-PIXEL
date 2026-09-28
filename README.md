# PixelRAG Studio

接手项目时建议按以下顺序阅读：

1. [`项目交接文档.md`](项目交接文档.md)：安装、使用、外部依赖、验证记录、已知限制和接手验收清单；
2. [`RAG系统架构与运行逻辑.md`](RAG系统架构与运行逻辑.md)：模块划分、数据契约、索引、检索、回答和引用校验逻辑；
3. 本 README：快速安装、命令行入口和主要功能说明。

## 从源码直接安装和启动

不需要先打包 EXE。Windows 用户可以下载源码，在项目自己的 `.venv` 虚拟环境中直接运行 `pixelrag_studio.py`。推荐使用这种方式进行交接、开发和内部试用。

### 运行条件

- Windows 10 或 Windows 11 64 位；
- Python 3.12 64 位，安装时勾选 Python Launcher 和 Tcl/Tk；
- 建议预留至少 15 GB 磁盘空间，用于 Python 依赖、模型和项目索引；
- 能访问 Python 包源；
- 首次使用默认远程模型名时，需要下载约 4 GB 的 Qwen3-VL-Embedding-2B；
- 扫描 PDF 的 OCR 需要 Java 11 或更高版本；
- Word、PowerPoint 和 Excel 的原生页面渲染及视觉对象导出需要安装相应的 Microsoft Office 桌面应用；
- 生成回答需要可访问阿里云百炼，并准备有效的 API Key。

### 最简单的安装方式

1. 从 GitHub 下载并解压项目，或克隆正式代码分支。
2. 双击 `setup.cmd`。脚本会创建 `.venv` 并安装 `requirements.txt` 中的依赖。
3. 安装完成后双击 `start-studio.cmd`。

第一次安装需要下载数 GB 依赖，耗时取决于网络。安装中断后可以再次运行 `setup.cmd`，脚本会复用已经创建的 `.venv`。

使用 Git 克隆默认 `main` 分支：

```powershell
git clone https://github.com/xyf966/RAG-PIXEL.git
cd RAG-PIXEL
.\setup.ps1
.\start-studio.cmd
```

### 使用 PowerShell 安装

```powershell
.\setup.ps1
.\.venv\Scripts\python.exe verify_environment.py
.\.venv\Scripts\python.exe pixelrag_studio.py
```

PowerShell 的执行策略阻止直接运行脚本时，可以使用 `setup.cmd`，或仅对当前进程执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

企业电脑已经安装 Python 3.12、但没有注册 `py` 或 `python` 命令时，可以显式指定解释器：

```powershell
.\setup.ps1 -PythonExecutable "C:\路径\到\python.exe"
```

### 环境自检

```powershell
.\.venv\Scripts\python.exe verify_environment.py
```

自检会验证 Python 版本、Tkinter、Docling、OpenDataLoader PDF、PixelRAG、PyTorch、Transformers、FAISS、Office COM Python 支持等依赖。Java 和 Microsoft Office 缺失默认显示警告，因为程序仍可处理部分格式；如需把它们作为强制条件，可运行：

```powershell
.\.venv\Scripts\python.exe verify_environment.py --require-java --require-office
```

### 运行测试

```powershell
.\.venv\Scripts\python.exe -m compileall -q hybrid_input pixelrag_studio.py verify_environment.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

### 源码交付说明

- `.venv`、模型、项目数据和新生成的测试产物均由 `.gitignore` 排除；接手人通过 `setup.cmd` 自行建立环境。仓库目前仍保留一批早期验收报告和图片，后续应确认无保留价值后停止跟踪。
- 不要把 API Key 写入源码或提交到 GitHub。可以在程序界面输入，或使用 `DASHSCOPE_API_KEY`。
- 不要把包含业务文件、索引和日志的 `PixelRAG-Studio-Data` 直接作为公开仓库内容上传。
- GUI、Hybrid 输入层、Office worker 及仓库内 Office 验收/诊断脚本均使用当前项目的 `.venv`，不再搜索或依赖开发者电脑上的外部 `.conda-env`。
- `HYBRID_DOCLING_PYTHON` 只用于明确覆盖 worker 解释器；未设置时使用当前项目 Python。
- `.build-env` 仅属于可选的历史 EXE 打包流程，不是源码安装、启动、解析、建库或测试的依赖。

## Hybrid 输入层

项目现已包含独立的 `hybrid_input` 输入包。它只负责把不同格式转换成统一的
`HybridDocument`。识别阶段不建立索引，也不调用 Pixel：

```text
原文件 → 格式检测 → 可替换解析器 → text/table/visual + 位置 → HybridDocument
```

当前支持 PDF、DOC/DOCX、PPT/PPTX、XLS/XLSX、TXT、Markdown、HTML 和常见图片。
PDF 默认使用 OpenDataLoader PDF 识别标题、段落、列表、表格、图片和阅读顺序，失败时自动回退 PyMuPDF；
DOC/DOCX、PPT/PPTX 和 XLS 默认通过独立 Docling worker 解析；XLSX 默认使用 openpyxl 原生读取单元格、公式、合并范围、图片和图表锚点，失败时回退 Docling；图片处理使用
识别契约为 schema 1.1，视觉对象统一使用 `kind=visual`，并通过 `visual_type` 区分
`image/chart/icon/diagram`。即使 ingest 命令传入 `--vision-processor pixelrag`，识别结果的
`vision_results` 仍为空；视觉向量只在索引构建阶段生成。
OpenDataLoader 使用本机现有 Java 11；可通过 `HYBRID_PDF_JAVA` 替换 Java 运行时。
Office 识别阶段只枚举原生视觉对象和位置，不导出图片。索引构建时才使用 Word
`CopyAsPicture`、PowerPoint `Shape.Export`、Excel `Chart.Export/Shape.CopyPicture` 物化这些区域。
PDF 通过 OpenDataLoader Hybrid 和矢量区域检测识别图片、图表、图标及示意图；区域裁图同样延迟到索引阶段。
透明背景合成、纯白/纯黑过滤、尺寸检查和去重均属于 Pixel 前的索引质量门。

每一步都可以独立调用：

```powershell
.\.venv\Scripts\python.exe -m hybrid_input capabilities
.\.venv\Scripts\python.exe -m hybrid_input detect 文档.docx
.\.venv\Scripts\python.exe -m hybrid_input parse 文档.docx --output 解析结果
.\.venv\Scripts\python.exe -m hybrid_input render 文档.docx --output 渲染结果
.\.venv\Scripts\python.exe -m hybrid_input ingest 文档.docx --output artifacts
.\.venv\Scripts\python.exe -m hybrid_input ingest 文档.docx --output artifacts --vision-processor pixelrag --vision-device cpu
```

PixelRAG 视觉处理器直接复用官方 `pixelrag_embed.embed_cpu` 后端。模型选择顺序为：
`--vision-model`、环境变量 `HYBRID_PIXELRAG_MODEL`、项目内已下载的 `model-cache/Qwen3-VL-Embedding-2B`，
最后才是 Hugging Face 模型名 `Qwen/Qwen3-VL-Embedding-2B`。首次使用远程模型名会下载约 4GB 权重。
同一次索引构建使用一个常驻 `PixelRAGVisionProcessor`，模型首次需要视觉向量时加载，随后跨文档复用，
构建完成后统一释放，不会为每份文档重复加载约 4GB 权重。

Docling/Office worker 默认使用当前项目的 `.venv`。如确有隔离执行需求，可通过 `HYBRID_DOCLING_PYTHON` 显式指定其他兼容的 Python 3.12 环境。
模块之间仅交换 JSON、图片文件和版本化数据契约；更换解析器或渲染器不要求修改
其他步骤。

### 在 VS Code 中运行

打开项目文件夹后，VS Code 会自动使用 `.venv` 解释器。进入左侧“运行和调试”，
可直接选择：

- `Studio：启动桌面应用`
- `Hybrid：查看可用能力`
- `Hybrid：识别当前文件`
- `Hybrid：仅解析当前文件`
- `Hybrid：渲染当前 Office 文件`
- `Hybrid：完整处理当前文件`

处理某份文档时，先在 VS Code 编辑器中打开或选中该文件，再运行对应配置。
结果统一写入工作区的 `.hybrid-output` 目录。测试可以直接使用 VS Code 的测试面板，
或运行默认测试任务“检查：语法与最小测试”。

PixelRAG Studio 当前提供面向 Windows 的本地混合文档输入、索引构建、检索检查与证据约束回答：

1. HybridPipeline 只识别文字、表格、视觉对象及来源位置；
2. 文字按标题、段落、位置和 token 上限进行结构感知分块；
3. 表格按表头、行组和列组生成独立结构化块；
4. Office/PDF 视觉区域在索引阶段才被物化并通过质量门；
5. 三类块统一由 Qwen3-VL-Embedding-2B 生成 2048 维归一化向量；
6. 文字、表格和视觉分别写入 schema 2.0 FAISS 索引快照；
7. 检索检查器对三个通道召回、融合与去重，并展示原始证据和来源位置；
8. 回答层通过阿里云百炼 API 生成结构化主张和逐证据原文摘录，并对每条主张执行强制引用校验。

检索层先在本地确定性拆分常见的中文复合问题，再可由百炼补充英文翻译和更细的“对象 + 属性 + 条件”查询，总查询数最多 8 条。远程查询增强失败时，本地子查询仍会继续检索，不会退回只搜整句。原问题、英文翻译和子查询分别执行向量召回、BM25 关键词召回和完整短语匹配，再通过加权 RRF 排序并交错合并，避免一个对象的重复结果挤掉其他对象。默认向量、关键词和精确匹配权重分别为 `1.0`、`1.5` 和 `2.0`，视觉通道使用 `1.10` 的校准权重。纯向量候选仍需通过原始相似度 `0.40` 的资格门，明确的关键词或短语命中可以进入融合候选。候选去重后会扩展相邻文字，并按页、幻灯片或工作表以及 `bbox` 坐标组装为回答层可引用的证据。

Pixel 图像接口的逻辑输入只来自 `kind=visual`；内部物化出的临时 `kind=image` 不属于识别契约。文字和表格使用同一模型的文本接口，进入 Pixel 图像接口的数量必须为 0。没有视觉对象的文档仍可建立文字和表格索引。

索引采用不可变快照。新快照全部写入并校验成功后，才通过 `CURRENT` 指针发布；构建失败不会覆盖上一个有效快照：

```text
index\
├── CURRENT
└── snapshots\<build-id>\
    ├── manifest.json
    ├── text.faiss
    ├── text-metadata.jsonl
    ├── table.faiss
    ├── table-metadata.jsonl
    ├── visual.faiss
    ├── visual-metadata.jsonl
    ├── visual-assets\
    └── build-report.json
```

旧的 `hybrid-index.json / semantic-index.json / image-index.faiss` 属于 schema 1.x，不能由 schema 2.0 读取，需要重新构建。

## 使用桌面界面

完成源码安装后双击 `start-studio.cmd`，或运行 `.\.venv\Scripts\python.exe pixelrag_studio.py`。如果后续取得单独发布的完整 EXE 目录，也可以双击其中的 `PixelRAG-Studio.exe`。进入界面后：

1. 创建项目；
2. 添加 PDF、Office 文档、图片或支持的文本型文档；
3. 点击“开始构建”；
4. 等待日志显示 schema 2.0 索引快照完成；
5. 打开“检索检查器”，输入问题并选择 Top K；
6. 点击“搜索”，在左侧查看排名和分数；右侧可切换“证据视图”和“原页视图”。PDF、Word、PowerPoint、Excel 会显示证据所在完整页面并标注核心/关联区域，图片按单页原图显示；
7. 打开“证据回答”，输入百炼 API Key，点击“连接测试”并选择回答模型；
8. 点击“基于当前检索结果生成回答”，查看带 `[E001]` 引用的回答与来源列表。

检索检查器仍可独立使用。回答层默认连接百炼北京地域的 OpenAI 兼容地址 `https://dashscope.aliyuncs.com/compatible-mode/v1` 并使用 `qwen-plus`。在界面输入 API Key，或通过 `DASHSCOPE_API_KEY` 环境变量提供；Key 不会写入回答诊断日志。可通过 `PIXELRAG_BAILIAN_BASE_URL` 与 `PIXELRAG_BAILIAN_MODEL` 覆盖默认地址和模型。使用新加坡地域时，请将 API 地址改为对应地域的兼容模式地址，并使用同地域的 API Key。

回答层不会直接信任模型生成的引用。它先把检索证据规范化为稳定的 `E001` 编号，再由 `LLMEvidenceSelector` 对最多 48 条候选执行语义筛选和覆盖排序。简单问题默认使用 6 条/1 万字符预算，复合、比较和列表问题可自适应扩到 10 条/1.6 万字符，并按文档交错取证，避免单一文档占满预算。长上下文只按实际会发送的最多 4,000 字符计入预算，因此深层关联表格不会被前面的长文本或图片说明挤掉。对于“某主体的某分类有什么”一类精确表格问题，会优先保留所有命中分类的核心块，并只向回答模型提供对应分类行，避免同一表格内相邻类别串入答案。每条事实主张必须引用预算内证据，并为每个引用返回能在对应证据中逐字定位的连续原文；伪造引用、漏引、无原文锚点、手写引用标记，或把“证据未提及某信息”写成带引用事实，都会触发一次修复，修复失败时安全停止。

复合问题只要至少一个子问题有直接证据，就回答有依据的部分，把缺失对象或条件写入 `limitations`，并返回 `partial_answer`；只有全部子问题都无直接证据时才拒答。

百炼请求使用 OpenAI 兼容的 `/chat/completions` 接口。Qwen3.7/3.8 的受支持 Max、Flash、Plus 型号使用严格 JSON Schema；旧型号保留 JSON Object 兼容模式。正常回答阶段通常包含一次证据筛选调用和一次结构化回答调用；引用校验在本地确定性执行，首次校验失败时最多再调用一次模型修复。回答请求默认关闭 Qwen 思考模式；断连、超时、429 和 5xx 最多自动重试两次，鉴权和参数错误不会重试。

回答诊断日志默认采用 metadata-only：保留证据 ID、分数、来源、坐标、预算和状态，不重复保存整段文档内容、邻接块、引用原文或完整模型原始响应。历史日志不会自动删除。

“向支持视觉的百炼模型发送图片”默认关闭。打开后只会发送已通过检索资格门、本地证据预算和路径检查的视觉资产；所选百炼模型必须自身支持视觉输入。

程序不依赖 Codex，也不依赖原来的 Conda 环境。源码方式通过 `setup.cmd` 把 PixelRAG、Torch、Transformers、FAISS 等 Python 依赖安装到项目 `.venv`；`.venv` 不上传 GitHub。只有另行交付的完整 EXE 目录才会携带自己的 Python 运行时和打包依赖，不能只复制单个 EXE。

## 模型与离线运行

Qwen3-VL-Embedding-2B 权重体积较大，不嵌入 EXE。第一次构建索引时，Transformers 会下载模型到：

```text
PixelRAG-Studio-Data\models
```

模型下载完成后，本地解析、建库和检索可以离线运行。也可以在界面的“视觉 Embedding 模型”中填写已经下载好的本地模型目录。回答生成仍需连接百炼。
应用使用 Windows 系统证书存储建立 HTTPS 连接，兼容由企业证书代理管理的电脑。

回答生成模型由阿里云百炼托管，不包含在 PixelRAG Studio 安装目录内。`Qwen3-VL-Embedding-2B` 仍是本地向量模型，不能替代百炼中的生成模型。

## 数据位置

默认情况下，项目、索引、日志和模型均保存在程序旁边：

```text
PixelRAG-Studio-Data\
├── models\
└── projects\
```

将整个 `PixelRAG-Studio` 文件夹及 `PixelRAG-Studio-Data` 一起复制，即可迁移到另一台兼容的 Windows 电脑。

## 性能提示

- 当前应用默认使用 CPU，首次建库和启动服务可能需要较长时间；
- 建议先用 1 份、1–5 页的 PDF 体验；
- 模型运行需要较大的内存；
- 后续可以加入 CUDA 设备选择以提高速度。

## 支持格式

PDF、DOC/DOCX、PPT/PPTX、XLS/XLSX、PNG、JPG/JPEG、WebP、Markdown、TXT、HTML。
