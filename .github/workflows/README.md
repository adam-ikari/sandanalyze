# SandAnalyze CI/CD Build Configuration

This directory contains CI/CD configurations for building and packaging SandAnalyze across multiple platforms.

## Supported Platforms

- **Linux** (Ubuntu) - x86_64
- **macOS** - arm64 (Apple Silicon)
- **Windows** - x86_64

## GitHub Actions Workflow

The `.github/workflows/build.yml` workflow:

1. **Test Phase**: Runs pytest on all three platforms (Ubuntu, Windows, macOS)
2. **Build Phase**: Uses PyInstaller to create standalone executables
3. **Package Phase**: Creates platform-specific archives (.tar.gz for Linux/macOS, .zip for Windows)
4. **Release Phase**: Automatically creates GitHub releases with artifacts when a version tag is pushed

### Trigger Conditions

- Push to `main` or `master` branch
- Pull requests to `main` or `master`
- Tags starting with `v` (e.g., `v1.0.0`)
- Manual trigger via `workflow_dispatch`

## Local Build Scripts

### Linux/macOS

```bash
./scripts/build.sh
```

Output: `dist/sandanalyze-<platform>-<arch>.tar.gz`

### Windows

```cmd
scripts\build.bat
```

Output: `dist\sandanalyze-windows-x86_64.zip`

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- PyInstaller (installed automatically by scripts)

## Build Outputs

| Platform | Format | Filename Pattern |
|----------|--------|------------------|
| Linux    | tar.gz | `sandanalyze-linux-x86_64.tar.gz` |
| macOS    | tar.gz | `sandanalyze-macos-arm64.tar.gz` |
| Windows  | zip    | `sandanalyze-windows-x86_64.zip` |
