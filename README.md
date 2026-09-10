# PixelRAG Studio

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
.build-env\Scripts\python.exe -m hybrid_input capabilities
.build-env\Scripts\python.exe -m hybrid_input detect 文档.docx
.build-env\Scripts\python.exe -m hybrid_input parse 文档.docx --output 解析结果
.build-env\Scripts\python.exe -m hybrid_input render 文档.docx --output 渲染结果
.build-env\Scripts\python.exe -m hybrid_input ingest 文档.docx --output artifacts
.build-env\Scripts\python.exe -m hybrid_input ingest 文档.docx --output artifacts --vision-processor pixelrag --vision-device cpu
```

PixelRAG 视觉处理器直接复用官方 `pixelrag_embed.embed_cpu` 后端。模型选择顺序为：
`--vision-model`、环境变量 `HYBRID_PIXELRAG_MODEL`、项目内已下载的 `model-cache/Qwen3-VL-Embedding-2B`，
最后才是 Hugging Face 模型名 `Qwen/Qwen3-VL-Embedding-2B`。首次使用远程模型名会下载约 4GB 权重。
同一次索引构建使用一个常驻 `PixelRAGVisionProcessor`，模型首次需要视觉向量时加载，随后跨文档复用，
构建完成后统一释放，不会为每份文档重复加载约 4GB 权重。

可通过 `HYBRID_DOCLING_PYTHON` 指定 Docling/Office worker 使用的 Python 环境。
模块之间仅交换 JSON、图片文件和版本化数据契约；更换解析器或渲染器不要求修改
其他步骤。

### 在 VS Code 中运行

打开项目文件夹后，VS Code 会自动使用 `.build-env` 解释器。进入左侧“运行和调试”，
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
8. 回答层通过阿里云百炼 API 生成结构化主张，并对每条主张执行强制引用校验。

检索层同时执行向量召回、BM25 关键词召回和完整短语匹配，再通过加权 RRF 融合排序；默认向量、关键词和精确匹配权重分别为 `1.0`、`1.5` 和 `2.0`，视觉通道使用 `1.10` 的校准权重。纯向量候选仍需通过原始相似度 `0.40` 的资格门，明确的关键词或短语命中可以进入融合候选。候选去重后会扩展相邻文字，并按页、幻灯片或工作表以及 `bbox` 坐标组装为回答层可引用的证据。

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

## 独立使用

双击 `PixelRAG-Studio.exe`，然后：

1. 创建项目；
2. 添加 PDF、Office 文档、图片或支持的文本型文档；
3. 点击“开始构建”；
4. 等待日志显示 schema 2.0 索引快照完成；
5. 打开“检索检查器”，输入问题并选择 Top K；
6. 点击“搜索”，在左侧查看排名和分数，在右侧查看文字、表格或图片证据；
7. 打开“证据回答”，输入百炼 API Key，点击“连接测试”并选择回答模型；
8. 点击“基于当前检索结果生成回答”，查看带 `[E001]` 引用的回答与来源列表。

检索检查器仍可独立使用。回答层默认连接百炼北京地域的 OpenAI 兼容地址 `https://dashscope.aliyuncs.com/compatible-mode/v1` 并使用 `qwen-plus`。在界面输入 API Key，或通过 `DASHSCOPE_API_KEY` 环境变量提供；Key 不会写入回答诊断日志。可通过 `PIXELRAG_BAILIAN_BASE_URL` 与 `PIXELRAG_BAILIAN_MODEL` 覆盖默认地址和模型。使用新加坡地域时，请将 API 地址改为对应地域的兼容模式地址，并使用同地域的 API Key。

回答层不会直接信任模型生成的引用。它先把检索证据规范化为稳定的 `E001` 编号并进行预算控制，再让百炼模型输出结构化主张。对于“某主体的某分类有什么”一类精确表格问题，会优先保留所有命中分类的核心块，并只向回答模型提供对应分类行，避免同一表格内相邻类别串入答案。每条事实主张必须引用已筛选的直接证据；伪造引用、漏引或手写引用标记会触发一次修复，修复失败时安全停止。证据不足时明确拒答。

如果首轮筛选只有 `context` 而没有 `direct/conflict`，回答层会对相关上下文执行一次支持性复核，区分“包含可写入答案的局部事实”和“仅为标题、单位或背景”。如果仍无直接证据，可再对排名靠前的文字和表格做一次语义复核；只有模型同时返回原证据中可逐字核验的 `support_quote` 才能恢复为 `direct`，否则在生成前拒答。

百炼请求使用 OpenAI 兼容的 `/chat/completions` 接口和 JSON Object 输出模式，具体 JSON Schema 同时写入提示词。生成后只执行本地确定性校验，包括 JSON 契约、引用编号和证据绑定；不再调用第二个 LLM 对答案进行语义或数值审计。回答请求默认关闭 Qwen 思考模式，避免模型长时间无输出而被网络代理中断。

“向支持视觉的百炼模型发送图片”默认关闭。打开后只会发送已通过检索资格门、证据筛选和路径检查的视觉资产；所选百炼模型必须自身支持视觉输入。

程序不依赖 Codex，也不依赖原来的 Conda 环境。运行时、PixelRAG、Torch、Transformers、FAISS 和服务组件均包含在应用目录中。
PDF 渲染所需的 Poppler 命令也随应用一起提供。

## 模型与离线运行

Qwen3-VL-Embedding-2B 权重体积较大，不嵌入 EXE。第一次构建索引时，Transformers 会下载模型到：

```text
PixelRAG-Studio-Data\models
```

模型下载完成后，应用可以离线运行。也可以在界面的“视觉 Embedding 模型”中填写已经下载好的本地模型目录。
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
