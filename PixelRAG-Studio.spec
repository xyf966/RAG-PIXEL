# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = [('C:/Users/XEE8SZH/Desktop/多模态输入/vendor/poppler/pdftoppm.exe', 'poppler'), ('C:/Users/XEE8SZH/Desktop/多模态输入/vendor/poppler/pdfinfo.exe', 'poppler')]
hiddenimports = ['PIL.ImageTk', 'pypdfium2', 'pixelrag_embed.embed_cpu', 'pixelrag_embed.chunk', 'pixelrag_embed.index', 'pixelrag_index.sources.local', 'transformers.models.qwen3_vl.configuration_qwen3_vl', 'transformers.models.qwen3_vl.modeling_qwen3_vl', 'transformers.models.qwen3_vl.processing_qwen3_vl', 'transformers.models.qwen3_vl.image_processing_qwen3_vl', 'transformers.models.qwen3_vl.video_processing_qwen3_vl']
tmp_ret = collect_all('pixelrag')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pixelrag_render')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pixelrag_embed')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pixelrag_index')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pixelrag_serve')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('faiss')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('truststore')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pdf2image')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['C:/Users/XEE8SZH/Desktop/多模态输入/pixelrag_studio.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PixelRAG-Studio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PixelRAG-Studio',
)
