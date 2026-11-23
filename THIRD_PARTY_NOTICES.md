# Third-Party Notices

This repository vendors the following open-source libraries under `ui/vendor/` for the self-contained Gaudium Web UI. Each library's full license text is kept in its respective subdirectory.

| Component | Path | License |
|---|---|---|
| Preact | `ui/vendor/preact/` | MIT (`ui/vendor/preact/LICENSE`) |
| htm | `ui/vendor/htm/` | Apache-2.0 (`ui/vendor/htm/LICENSE`) |
| Marked | `ui/vendor/marked/` | MIT (`ui/vendor/marked/LICENSE.md`) |
| DOMPurify | `ui/vendor/dompurify/` | Apache-2.0 / MPL-2.0 (`ui/vendor/dompurify/LICENSE`) |
| KaTeX | `ui/vendor/katex/` | MIT (`ui/vendor/katex/LICENSE`) |
| highlight.js | `ui/vendor/hljs/` | BSD-3-Clause (`ui/vendor/hljs/LICENSE`) |
| PDF.js | `ui/vendor/pdfjs/` | Apache-2.0 (`ui/vendor/pdfjs/LICENSE`) |
| JetBrains Mono | `ui/vendor/fonts/` | SIL Open Font License 1.1 (`ui/vendor/fonts/LICENSE-JetBrainsMono`) |

## Intel Gaudi SDK interface header (compiled into the glue libraries)

The graph-compiler glue libraries built from `kernels/glue/` (`libgaudium_tpc.so`,
`libgaudium_ga.so`, `libgaudium_st.so`) are compiled against `tpc_kernel_lib_interface.h`, the
header of the Intel Gaudi SDK that defines the interface between the graph compiler and a
kernel library. The header is not part of this repository; it is read from the installed SDK
when the kernels are built. It is under the BSD 2-Clause licence below, which binary
distributions of those libraries (the release archives) reproduce here:

```text
Copyright (c) 2022 Habana Labs.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

*   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
*   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

No file of the Intel Gaudi SDK is distributed with Gaudium: the driver, the firmware, SynapseAI
and `tpc-clang` are installed from Intel under Intel's own licence. The kernel ELFs in a
release archive are what `tpc-clang` produced from the sources in `kernels/tpc/`; where a
kernel calls one of the compiler's built-in special functions (such as `v_exp_f32` or
`v_rsqrt_f32`), the compiler puts its code for that function into the ELF.
