/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

namespace clang {
class FunctionDecl;
} // namespace clang

namespace ascify {
namespace detail {

// This is an analyzer-to-emitter contract, not a lowering IR.  It records the
// identity and origin of a kernel only after the existing family analyzer has
// completed its full semantic proof.
enum class StaticKernelSemanticFamily : unsigned char {
  Unknown,
  Softmax,
  RmsNorm,
  LayerNorm,
};

enum class StaticKernelSourceProvenance : unsigned char {
  Unknown,
  MainFile,
  UserHeader,
  SystemHeader,
};

struct StaticKernelSourceRange {
  unsigned beginRawEncoding = 0;
  unsigned endRawEncoding = 0;

  bool valid() const {
    return beginRawEncoding != 0 && endRawEncoding != 0;
  }
};

struct StaticKernelProof {
  const clang::FunctionDecl *canonicalDecl = nullptr;
  const clang::FunctionDecl *definitionDecl = nullptr;
  StaticKernelSourceRange definitionRange;
  StaticKernelSourceProvenance provenance =
      StaticKernelSourceProvenance::Unknown;
  StaticKernelSemanticFamily family =
      StaticKernelSemanticFamily::Unknown;

  bool proven() const {
    return canonicalDecl != nullptr && definitionDecl != nullptr &&
           definitionRange.valid() &&
           provenance != StaticKernelSourceProvenance::Unknown &&
           family != StaticKernelSemanticFamily::Unknown;
  }
};

} // namespace detail
} // namespace ascify
