# PixelRAG 混合文档输入验收

- 模型：本地 `Qwen3-VL-Embedding-2B`
- 设备：`cpu`
- 文档数：4
- 图片块：14，进入 PixelRAG：14
- 文字块：64，进入 PixelRAG：0
- 表格块：12，进入 PixelRAG：0
- 非图片误入数：0

| 文档 | 类型 | 文字 | 表格 | 图片 | Pixel 结果 | 非图片误入 | 向量维度 | 状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Word 测试样本 | docx | 13 | 1 | 2 | 2 | 0 | 2048 | PASS |
| PowerPoint 测试样本 | pptx | 13 | 1 | 2 | 2 | 0 | 2048 | PASS |
| Excel 测试样本 | xlsx | 3 | 3 | 2 | 2 | 0 | 2048 | PASS |
| PDF 测试样本 | pdf | 35 | 7 | 8 | 8 | 0 | 2048 | PASS |

## 验收结论

通过条件：每个有效图片块恰好对应一个 PixelRAG 结果；文字块和表格块均无 PixelRAG 结果；所有结果 provider 为 `pixelrag`、status 为 `complete`，且向量维度一致。

**PASS**
