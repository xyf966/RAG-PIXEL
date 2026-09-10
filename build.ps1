$ErrorActionPreference = 'Stop'

$Python = Join-Path $PSScriptRoot '.build-env\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Build environment not found. Create .build-env and install pixelrag[all] plus pyinstaller first.'
}

$Dist = Join-Path $PSScriptRoot 'dist'
$Build = Join-Path $PSScriptRoot 'build'
$Output = Join-Path $Dist 'PixelRAG-Studio'
$ProjectData = Join-Path $Output 'PixelRAG-Studio-Data'
$BuildBackupRoot = Join-Path $PSScriptRoot '.codex-tmp\build-data-backup'
$ProjectDataBackup = Join-Path $BuildBackupRoot 'PixelRAG-Studio-Data'
$VendorPoppler = Join-Path $PSScriptRoot 'vendor\poppler'
$PdfToPpm = Join-Path $VendorPoppler 'pdftoppm.exe'
$PdfInfo = Join-Path $VendorPoppler 'pdfinfo.exe'

if (Test-Path -LiteralPath $ProjectData) {
    New-Item -ItemType Directory -Force -Path $BuildBackupRoot | Out-Null
    if (Test-Path -LiteralPath $ProjectDataBackup) {
        Remove-Item -LiteralPath $ProjectDataBackup -Recurse -Force
    }
    Move-Item -LiteralPath $ProjectData -Destination $ProjectDataBackup
}

try {
    & $Python -m PyInstaller `
        --noconfirm `
        --clean `
        --onedir `
        --windowed `
        --name 'PixelRAG-Studio' `
        --distpath $Dist `
        --workpath $Build `
        --specpath $PSScriptRoot `
        --collect-all pixelrag `
        --collect-all pixelrag_render `
        --collect-all pixelrag_embed `
        --collect-all pixelrag_index `
        --collect-all pixelrag_serve `
        --collect-all faiss `
        --collect-all truststore `
        --collect-all pdf2image `
        --collect-submodules hybrid_input `
        --add-binary "${PdfToPpm};poppler" `
        --add-binary "${PdfInfo};poppler" `
        --hidden-import PIL.ImageTk `
        --hidden-import pypdfium2 `
        --hidden-import pixelrag_embed.embed_cpu `
        --hidden-import pixelrag_embed.chunk `
        --hidden-import pixelrag_embed.index `
        --hidden-import pixelrag_index.sources.local `
        --hidden-import transformers.models.qwen3_vl.configuration_qwen3_vl `
        --hidden-import transformers.models.qwen3_vl.modeling_qwen3_vl `
        --hidden-import transformers.models.qwen3_vl.processing_qwen3_vl `
        --hidden-import transformers.models.qwen3_vl.image_processing_qwen3_vl `
        --hidden-import transformers.models.qwen3_vl.video_processing_qwen3_vl `
        --add-data "${PSScriptRoot}\hybrid_input;hybrid_input" `
        (Join-Path $PSScriptRoot 'pixelrag_studio.py')

    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }
}
finally {
    if (Test-Path -LiteralPath $ProjectDataBackup) {
        New-Item -ItemType Directory -Force -Path $Output | Out-Null
        Move-Item -LiteralPath $ProjectDataBackup -Destination $ProjectData
    }
}

Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'README.md') -Destination $Output -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'THIRD_PARTY_NOTICES.md') -Destination $Output -Force

$ModelSource = Join-Path $PSScriptRoot 'model-cache\Qwen3-VL-Embedding-2B'
if (Test-Path -LiteralPath (Join-Path $ModelSource 'model.safetensors')) {
    $ModelTarget = Join-Path $Output 'models\Qwen3-VL-Embedding-2B'
    New-Item -ItemType Directory -Force -Path $ModelTarget | Out-Null
    Copy-Item -Path (Join-Path $ModelSource '*') -Destination $ModelTarget -Recurse -Force
}

Write-Host "Built: $Output\PixelRAG-Studio.exe"
