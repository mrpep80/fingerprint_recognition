# Bundled NBIS runtimes

The desktop application can ship NBIS without requiring Homebrew, CMake, or a system installation.

Expected runtime layout:

```text
runtimes/
├── darwin-arm64/bin/{mindtct,bozorth3}
├── darwin-x64/bin/{mindtct,bozorth3}
├── linux-x64/bin/{mindtct,bozorth3}
└── windows-x64/bin/{mindtct.exe,bozorth3.exe}
```

Only binaries built from a pinned, verified NBIS release should be placed here. The runtime resolver selects the directory matching the host OS/CPU. During development, a system `mindtct`/`bozorth3` in `PATH` is accepted as a fallback.

Do not commit locally built debug binaries. The release packaging process should populate the platform directory from a reproducible build and record the NBIS version/checksums in the release manifest.
