# PixelRAG Studio

## Hybrid 输入层

项目现已包含独立的 `hybrid_input` 输入包。它只负责把不同格式转换成统一的
`HybridDocument`，不负责建立检索索引：

```text
原文件 → 格式检测 → 可替换解析器 → 图片标准化 → 可替换视觉处理器 → HybridDocument
```

当前支持 PDF、DOC/DOCX、PPT/PPTX、XLS/XLSX、TXT、Markdown、HTML 和常见图片。
PDF 默认使用 PyMuPDF；Office 默认通过独立 Docling worker 解析；图片处理使用
Pillow；视觉向量化阶段暂为 `deferred`，等待后续单独接入 PixelRAG。
Office 文件会在独立布局补全步骤中调用 Microsoft Office 导出临时 PDF，将 Docling
逻辑位置与真实页码、归一化 bbox 对齐；该步骤失败时不会破坏原始语义解析结果。
Excel 原生图表由独立的 Excel COM 渲染器直接导出 PNG，并保留工作表名、单元格范围、
Excel 点坐标和逻辑 bbox；该渲染器可单独替换，不影响解析、图片处理或后续视觉模型。

每一步都可以独立调用：

```powershell
.build-env\Scripts\python.exe -m hybrid_input capabilities
.build-env\Scripts\python.exe -m hybrid_input detect 文档.docx
.build-env\Scripts\python.exe -m hybrid_input parse 文档.docx --output 解析结果
.build-env\Scripts\python.exe -m hybrid_input render 文档.docx --output 渲染结果
.build-env\Scripts\python.exe -m hybrid_input ingest 文档.docx --output artifacts
```

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

PixelRAG Studio 是面向 Windows 的本地视觉文档检索应用。它封装 PixelRAG 0.4.0 官方完整流水线：

1. Pixelshot 将 PDF、图片和文档渲染为视觉页面；
2. 页面切分为 1024px 视觉块；
3. Qwen3-VL-Embedding-2B 生成视觉向量；
4. FAISS 建立本地索引；
5. 输入文字问题，直接检索相关页面视觉区域。

## 独立使用

双击 `PixelRAG-Studio.exe`，然后：

1. 创建项目；
2. 添加 PDF 或图片；
3. 点击“开始构建”；
4. 等待日志显示索引完成；
5. 点击“启动搜索服务”；
6. 等待模型加载完成后输入问题并搜索。

程序不依赖 Codex，也不依赖原来的 Conda 环境。运行时、PixelRAG、Torch、Transformers、FAISS 和服务组件均包含在应用目录中。
PDF 渲染所需的 Poppler 命令也随应用一起提供。

## 模型与离线运行

Qwen3-VL-Embedding-2B 权重体积较大，不嵌入 EXE。第一次构建索引时，Transformers 会下载模型到：

```text
PixelRAG-Studio-Data\models
```

模型下载完成后，应用可以离线运行。也可以在界面的“视觉 Embedding 模型”中填写已经下载好的本地模型目录。
应用使用 Windows 系统证书存储建立 HTTPS 连接，兼容由企业证书代理管理的电脑。

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

PDF、PNG、JPG/JPEG、Markdown、TXT、HTML。
