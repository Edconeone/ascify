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

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <string>
#include <utility>
#include "AscifyAction.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#if LLVM_VERSION_MAJOR < 17
#include "clang/Basic/TargetInfo.h"
#endif
#include "LLVMCompat.h"
#include "ArgParse.h"
#include "LocalHeader.h"
#include "ImplicitCudaHeaders.h"
#include "CUDA2DPP.h"
#include "StringUtils.h"

using namespace ascify;

const std::string sDPP = "DPP";
const std::string s_string_literal = "[string literal]";
// Matchers' names
const StringRef sCudaLaunchKernel = "cudaLaunchKernel";
const StringRef sCudaGlobalScalarDoubleParam = "cudaGlobalScalarDoubleParam";
const StringRef sCudaDefaultDim3 = "cudaDefaultDim3";
const StringRef sProvenGlobalAtomicCall = "provenGlobalAtomicCall";
const StringRef sCanonicalWarpAddReduction = "canonicalWarpAddReduction";
const StringRef sCanonicalBinaryReducer = "canonicalBinaryReducer";

namespace {

bool isRecognizedNvidiaSampleHelperCuda(llvm::StringRef path) {
  if (path.empty() || llvm::sys::path::filename(path) != "helper_cuda.h")
    return false;
  const auto bufferOrError = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrError)
    return false;
  const llvm::StringRef contents = (*bufferOrError)->getBuffer();
  // These markers jointly identify NVIDIA cuda-samples' helper_cuda surface
  // and the exact two macro contracts admitted below. A basename match alone
  // is deliberately insufficient: a project-local same-name header must not
  // acquire sample-helper semantics.
  return contents.contains("#ifndef COMMON_HELPER_CUDA_H_") &&
         contents.contains("#define COMMON_HELPER_CUDA_H_") &&
         contents.contains(
             "#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)") &&
         contents.contains(
             "#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)") &&
         contents.contains("template <typename T>\nvoid check(T result") &&
         contents.contains("inline void __getLastCudaError(");
}

bool locationComesFromRecognizedNvidiaSampleHelper(
    clang::SourceManager &sourceManager, clang::SourceLocation location) {
  const clang::SourceLocation spelling =
      sourceManager.getSpellingLoc(location);
  if (spelling.isInvalid())
    return false;
  return isRecognizedNvidiaSampleHelperCuda(
      sourceManager.getFilename(spelling));
}

bool locationComesFromAscifyCudaCompat(
    clang::SourceManager &sourceManager, clang::SourceLocation location) {
  const clang::SourceLocation spelling =
      sourceManager.getSpellingLoc(location);
  if (spelling.isInvalid())
    return false;
  const llvm::StringRef path = sourceManager.getFilename(spelling);
  if (path.empty() ||
      llvm::sys::path::filename(path) != "ascify_cuda_compat.hpp")
    return false;
  const auto bufferOrError = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrError)
    return false;
  const llvm::StringRef contents = (*bufferOrError)->getBuffer();
  return contents.contains("#ifndef ASCIFY_ASCIFY_CUDA_COMPAT_HPP") &&
         contents.contains(
             "#define ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS") &&
         contents.contains(
             "#define ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR") &&
         contents.contains("inline void sampleCheckCudaErrors(") &&
         contents.contains("inline void sampleGetLastCudaError(");
}

bool activeNvidiaSampleHelperMacroBodyMatches(
    llvm::StringRef name, const clang::MacroInfo &macroInfo,
    clang::Preprocessor &preprocessor) {
  if (!macroInfo.isFunctionLike() || macroInfo.getNumParams() != 1)
    return false;
  const clang::IdentifierInfo *parameter = *macroInfo.param_begin();
  if (parameter == nullptr)
    return false;
  const std::string parameterName = parameter->getName().str();
  std::vector<std::string> actual;
  actual.reserve(macroInfo.getNumTokens());
  for (const clang::Token &token : macroInfo.tokens())
    actual.push_back(preprocessor.getSpelling(token));

  std::vector<std::string> expected;
  if (name == "checkCudaErrors") {
    expected = {"check", "(", "(", parameterName, ")", ",", "#",
                parameterName, ",", "__FILE__", ",", "__LINE__", ")"};
  } else if (name == "getLastCudaError") {
    expected = {"__getLastCudaError", "(", parameterName, ",",
                "__FILE__", ",", "__LINE__", ")"};
  } else {
    return false;
  }
  return actual == expected;
}

bool isAdmittedCudaRuntimeStatusCall(
    clang::SourceManager &sourceManager, const clang::Expr *expression) {
  if (expression == nullptr)
    return false;
  expression = expression->IgnoreParenImpCasts();
  const auto *call = llvm::dyn_cast<clang::CallExpr>(expression);
  if (call == nullptr)
    return false;
  const clang::FunctionDecl *callee = call->getDirectCallee();
  if (callee == nullptr || callee->getIdentifier() == nullptr)
    return false;
  const llvm::StringRef name = callee->getName();
  // This is a return-domain allowlist, not a general CUDA API allowlist.
  // Every admitted name is implemented by ascify_cuda_compat.hpp with an
  // aclError result. Occupancy, Driver APIs, library statuses, and arbitrary
  // integer-valued calls stay outside the set even when wrapped by NVIDIA's
  // generic checkCudaErrors macro.
  static const char *const admittedNames[] = {
      "cudaMalloc", "cudaFree", "cudaMemcpy", "cudaMemcpyAsync",
      "cudaMemset", "cudaMemsetAsync", "cudaMallocHost",
      "cudaFreeHost", "cudaGetDevice", "cudaSetDevice",
      "cudaGetDeviceCount", "cudaDeviceGetAttribute",
      "cudaDeviceSynchronize", "cudaDeviceReset",
      "cudaFuncGetAttributes", "cudaFuncSetAttribute",
      "cudaStreamCreate", "cudaStreamCreateWithFlags",
      "cudaStreamDestroy", "cudaStreamSynchronize",
      "cudaEventCreate", "cudaEventDestroy", "cudaEventRecord",
      "cudaEventSynchronize", "cudaEventElapsedTime",
      "cudaGetLastError", "cudaPeekAtLastError",
  };
  if (std::find_if(std::begin(admittedNames), std::end(admittedNames),
                   [&](const char *admitted) { return name == admitted; }) ==
      std::end(admittedNames))
    return false;
  const auto mapped = CUDA_RENAMES_MAP().find(name);
  if (mapped == CUDA_RENAMES_MAP().end() ||
      mapped->second.apiType != API_RUNTIME ||
      !mapped->second.dppName.starts_with("ascify::"))
    return false;
  const clang::SourceLocation declarationLocation =
      sourceManager.getExpansionLoc(callee->getLocation());
  return declarationLocation.isValid() &&
         !sourceManager.isWrittenInMainFile(declarationLocation) &&
         sourceManager.isInSystemHeader(declarationLocation) &&
         !locationComesFromRecognizedNvidiaSampleHelper(
             sourceManager, declarationLocation);
}

const char *const RowwiseSimdReservedMacros[] = {
    "ascify",
    "target",
    "dav_c310",
    "rowwise_simd_v1",
    "SimdTryResult",
    "HybridTryResult",
    "NotHandled",
    "RowwiseSimdFacadeV1",
    "RowwiseHybridFacadeV1",
    "HybridRecipeDescriptor",
    "HybridStageDescriptor",
    "DispatchRegisteredHybrid",
    "TrySoftmaxSimd",
    "TryRmsNormSimd",
    "TryLayerNormSimd",
    "TrySoftmaxHybrid",
    "TryRmsNormHybrid",
    "TryLayerNormHybrid",
    "ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_RECIPES_HPP_",
    "ASCIFY_TARGET_DAV_C310_ROWWISE_HYBRID_REGISTRY_V1_HPP_",
    "ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_SELECTORS_V1_HPP_",
    "ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_ABI_H_",
    "ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_ABI_TYPES_READY",
    "ASCIFY_DAV_C310_ROWWISE_SIMD_ABI_VERSION",
    "ASCIFY_ASCIFY_CUDA_COMPAT_HPP",
    "ASCIFY_ALIGN",
    "ASCIFY_FORCEINLINE",
    "ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS",
    "ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR",
    "ascify950_softmax_reg_recompute_launch_v1",
    "ascify950_rmsnorm_reg_cached_launch_v1",
    "ascify950_rmsnorm_reg_plain_rowbatch_launch_v1",
    "ascify950_layernorm_reg_cached_launch_v1",
    "handled",
    "status",
    "alignas",
    "alignof",
    "asm",
    "auto",
    "bool",
    "break",
    "case",
    "catch",
    "char",
    "char16_t",
    "char32_t",
    "class",
    "const",
    "constexpr",
    "const_cast",
    "continue",
    "decltype",
    "default",
    "delete",
    "do",
    "double",
    "dynamic_cast",
    "else",
    "enum",
    "explicit",
    "export",
    "extern",
    "false",
    "float",
    "for",
    "friend",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "mutable",
    "namespace",
    "new",
    "noexcept",
    "public",
    "nullptr",
    "operator",
    "private",
    "protected",
    "register",
    "reinterpret_cast",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "static_assert",
    "static_cast",
    "struct",
    "switch",
    "template",
    "this",
    "thread_local",
    "throw",
    "true",
    "try",
    "typedef",
    "typeid",
    "typename",
    "union",
    "unsigned",
    "using",
    "virtual",
    "void",
    "volatile",
    "wchar_t",
    "while",
    "exp",
    "expf",
    "__expf",
    "rsqrt",
    "rsqrtf",
    "__frsqrt_rn",
    "__fdividef",
    "ascify_target_direct_load_tag",
    "ascify_target_direct_store_tag",
    "ascify_target_adapter_owner_type",
    "ascify_target_storage_type",
    "ascify_target_compute_type",
    "ascify_target_store_is_affine",
    "ascify_target_data",
    "ascify_target_row_stride",
    "ascify_target_weight",
};

bool isRowwiseSimdReservedMacro(llvm::StringRef name) {
  return std::find_if(
             std::begin(RowwiseSimdReservedMacros),
             std::end(RowwiseSimdReservedMacros),
             [&](const char *reserved) { return name == reserved; }) !=
         std::end(RowwiseSimdReservedMacros);
}

bool isRowwiseSimdPublishedHeaderMacro(llvm::StringRef name) {
  return name == "ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_RECIPES_HPP_" ||
         name == "ASCIFY_TARGET_DAV_C310_ROWWISE_HYBRID_REGISTRY_V1_HPP_" ||
         name == "ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_SELECTORS_V1_HPP_" ||
         name == "ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_ABI_H_" ||
         name == "ASCIFY_DAV_C310_ROWWISE_SIMD_ABI_VERSION" ||
         name == "ASCIFY_ASCIFY_CUDA_COMPAT_HPP" ||
         name == "ASCIFY_ALIGN" || name == "ASCIFY_FORCEINLINE";
}

bool isRowwiseSimdProtectedDeclarationName(llvm::StringRef name) {
  static const char *const names[] = {
      "ascify",
      "SimdTryResult",
      "HybridTryResult",
      "NotHandled",
      "RowwiseSimdFacadeV1",
      "RowwiseHybridFacadeV1",
      "HybridRecipeDescriptor",
      "HybridStageDescriptor",
      "DispatchRegisteredHybrid",
      "TrySoftmaxSimd",
      "TryRmsNormSimd",
      "TryLayerNormSimd",
      "TrySoftmaxHybrid",
      "TryRmsNormHybrid",
      "TryLayerNormHybrid",
      "ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_RECIPES_HPP_",
      "ASCIFY_TARGET_DAV_C310_ROWWISE_HYBRID_REGISTRY_V1_HPP_",
      "ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_SELECTORS_V1_HPP_",
      "ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_ABI_H_",
      "ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_ABI_TYPES_READY",
      "ASCIFY_DAV_C310_ROWWISE_SIMD_ABI_VERSION",
      "ASCIFY_ASCIFY_CUDA_COMPAT_HPP",
      "ASCIFY_ALIGN",
      "ASCIFY_FORCEINLINE",
      "ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS",
      "ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR",
      "ascify_target_direct_load_tag",
      "ascify_target_direct_store_tag",
      "ascify_target_adapter_owner_type",
      "ascify_target_storage_type",
      "ascify_target_compute_type",
      "ascify_target_store_is_affine",
      "ascify_target_data",
      "ascify_target_row_stride",
      "ascify_target_weight",
      "ascify950_softmax_reg_recompute_launch_v1",
      "ascify950_rmsnorm_reg_cached_launch_v1",
      "ascify950_rmsnorm_reg_plain_rowbatch_launch_v1",
      "ascify950_layernorm_reg_cached_launch_v1",
  };
  return std::find_if(
             std::begin(names), std::end(names),
             [&](const char *protectedName) {
               return name == protectedName;
             }) != std::end(names);
}

struct RowwiseSimdConflictingDeclaration {
  clang::SourceLocation location;
  std::string name;

  explicit operator bool() const { return !name.empty(); }
};

RowwiseSimdConflictingDeclaration
findRowwiseSimdConflictingDeclaration(clang::ASTContext &context) {
  class Visitor : public clang::RecursiveASTVisitor<Visitor> {
  public:
    explicit Visitor(clang::SourceManager &sourceManager)
        : sourceManager(sourceManager) {}

    bool VisitFileScopeAsmDecl(clang::FileScopeAsmDecl *declaration) {
      if (conflict || declaration == nullptr)
        return !conflict;
      return rejectUserAsm(declaration->getLocation(), "GNU asm");
    }

    bool VisitGCCAsmStmt(clang::GCCAsmStmt *statement) {
      if (conflict || statement == nullptr)
        return !conflict;
      return rejectUserAsm(statement->getAsmLoc(), "GNU asm");
    }

    bool VisitMSAsmStmt(clang::MSAsmStmt *statement) {
      if (conflict || statement == nullptr)
        return !conflict;
      return rejectUserAsm(statement->getAsmLoc(), "Microsoft asm");
    }

    bool VisitNamespaceDecl(clang::NamespaceDecl *declaration) {
      if (conflict || declaration == nullptr || declaration->isImplicit())
        return !conflict;
      const std::string qualified =
          declaration->getQualifiedNameAsString();
      const llvm::StringRef qualifiedRef(qualified);
      if (qualifiedRef == "ascify::target::dav_c310" ||
          qualifiedRef.starts_with("ascify::target::dav_c310::")) {
        conflict = {
            declaration->getLocation(), qualified};
        return false;
      }
      return true;
    }

    bool VisitNamedDecl(clang::NamedDecl *declaration) {
      if (conflict || declaration == nullptr ||
          declaration->getIdentifier() == nullptr)
        return !conflict;
      const llvm::StringRef name = declaration->getName();
      if (const auto *assemblerLabel =
              declaration->getAttr<clang::AsmLabelAttr>()) {
        llvm::StringRef label = assemblerLabel->getLabel();
        if (!label.empty() && label.front() == '\1')
          label = label.drop_front();
        if (isRowwiseSimdProtectedDeclarationName(label)) {
          conflict = {
              declaration->getLocation(), label.str()};
          return false;
        }
      }
      if (isRowwiseSimdProtectedDeclarationName(name)) {
        conflict = {
            declaration->getLocation(),
            declaration->getQualifiedNameAsString()};
        return false;
      }
      if (name != "RowwiseSimdFacadeV1" &&
          name != "RowwiseHybridFacadeV1" &&
          name != "TrySoftmaxSimd" && name != "TryRmsNormSimd" &&
          name != "TryLayerNormSimd" &&
          name != "TrySoftmaxHybrid" && name != "TryRmsNormHybrid" &&
          name != "TryLayerNormHybrid")
        return true;
      const auto *owner = llvm::dyn_cast_or_null<clang::NamedDecl>(
          declaration->getDeclContext());
      if (owner == nullptr)
        return true;
      const std::string qualifiedOwner =
          owner->getQualifiedNameAsString();
      const llvm::StringRef qualifiedOwnerRef(qualifiedOwner);
      if (qualifiedOwnerRef == "ascify::target::dav_c310" ||
          qualifiedOwnerRef.starts_with("ascify::target::dav_c310::")) {
        conflict = {
            declaration->getLocation(),
            declaration->getQualifiedNameAsString()};
        return false;
      }
      return true;
    }

  private:
    bool rejectUserAsm(
        clang::SourceLocation location, llvm::StringRef spelling) {
      const clang::SourceLocation expansion =
          sourceManager.getExpansionLoc(location);
      if (expansion.isInvalid() ||
          sourceManager.isInSystemHeader(expansion))
        return true;
      conflict = {location, spelling.str()};
      return false;
    }

  public:

    RowwiseSimdConflictingDeclaration conflict;
    clang::SourceManager &sourceManager;
  } visitor(context.getSourceManager());

  visitor.TraverseDecl(context.getTranslationUnitDecl());
  return visitor.conflict;
}

struct RawRowwiseSimdConflict {
  std::string macro;
  std::string declaration;
};

RawRowwiseSimdConflict rawFileRowwiseSimdConflict(
    clang::CompilerInstance &compiler, clang::FileID file,
    bool inspectAsmTokens = false) {
  clang::SourceManager &sourceManager =
      compiler.getSourceManager();
  clang::Preprocessor &preprocessor =
      compiler.getPreprocessor();
  bool invalid = false;
  const llvm::StringRef buffer =
      sourceManager.getBufferData(file, &invalid);
  if (invalid)
    return {};
  clang::Lexer lexer(
      sourceManager.getLocForStartOfFile(file),
      preprocessor.getLangOpts(), buffer.begin(), buffer.begin(),
      buffer.end());

  enum class DirectiveState { None, Hash, Define };
  DirectiveState state = DirectiveState::None;
  RawRowwiseSimdConflict conflict;
  clang::Token token;
  const auto tokenName = [](const clang::Token &candidate) {
    if (candidate.is(clang::tok::raw_identifier))
      return candidate.getRawIdentifier();
    return candidate.getIdentifierInfo() == nullptr
               ? llvm::StringRef()
               : candidate.getIdentifierInfo()->getName();
  };
  lexer.LexFromRawLexer(token);
  while (token.isNot(clang::tok::eof)) {
    if (token.isAtStartOfLine())
      state = DirectiveState::None;
    if (state == DirectiveState::None &&
        token.is(clang::tok::hash) &&
        token.isAtStartOfLine()) {
      state = DirectiveState::Hash;
    } else if (state == DirectiveState::Hash &&
               tokenName(token) == "define") {
      state = DirectiveState::Define;
    } else if (state == DirectiveState::Define &&
               !tokenName(token).empty()) {
      const llvm::StringRef name = tokenName(token);
      if (isRowwiseSimdReservedMacro(name))
        conflict.macro = name.str();
      state = DirectiveState::None;
    } else if (state != DirectiveState::None) {
      state = DirectiveState::None;
    }
    const llvm::StringRef name = tokenName(token);
    if (inspectAsmTokens && conflict.declaration.empty() &&
        (name == "asm" || name == "__asm" || name == "__asm__"))
      conflict.declaration = "GNU asm";
    if (conflict.declaration.empty() &&
        !name.empty() && isRowwiseSimdProtectedDeclarationName(name)) {
      conflict.declaration = name.str();
    }
    if (!conflict.macro.empty() && !conflict.declaration.empty())
      break;
    lexer.LexFromRawLexer(token);
  }
  return conflict;
}

std::string findConventionalMainOuterGuard(
    clang::CompilerInstance &compiler) {
  clang::SourceManager &sourceManager = compiler.getSourceManager();
  bool invalid = false;
  const llvm::StringRef buffer = sourceManager.getBufferData(
      sourceManager.getMainFileID(), &invalid);
  if (invalid)
    return {};

  clang::Lexer lexer(
      sourceManager.getLocForStartOfFile(sourceManager.getMainFileID()),
      compiler.getPreprocessor().getLangOpts(), buffer.begin(),
      buffer.begin(), buffer.end());
  std::vector<std::string> spellings;
  std::vector<bool> startsOfLine;
  clang::Token token;
  lexer.LexFromRawLexer(token);
  while (token.isNot(clang::tok::eof) && spellings.size() < 6) {
    if (token.is(clang::tok::comment)) {
      lexer.LexFromRawLexer(token);
      continue;
    }
    bool spellingInvalid = false;
    const std::string spelling = clang::Lexer::getSpelling(
        token, sourceManager, compiler.getPreprocessor().getLangOpts(),
        &spellingInvalid);
    if (spellingInvalid)
      return {};
    spellings.push_back(spelling);
    startsOfLine.push_back(token.isAtStartOfLine());
    lexer.LexFromRawLexer(token);
  }
  if (spellings.size() != 6 || spellings[0] != "#" ||
      spellings[1] != "ifndef" || spellings[2].empty() ||
      spellings[3] != "#" || spellings[4] != "define" ||
      spellings[5] != spellings[2] || !startsOfLine[0] ||
      !startsOfLine[3]) {
    return {};
  }
  return spellings[2];
}

} // namespace

void AscifyAction::RewriteString(StringRef s, clang::SourceLocation start) {
  auto &SM = getCompilerInstance().getSourceManager();
  size_t begin = 0;
  while ((begin = s.find("cu", begin)) != StringRef::npos) {
    const size_t end = s.find_first_of(" ", begin + 4);
    StringRef name = s.slice(begin, end);
    const auto found = CUDA_RENAMES_MAP().find(name);
    if (found != CUDA_RENAMES_MAP().end()) {
      StringRef repName = found->second.dppName;
      dppCounter counter = {s_string_literal, ConvTypes::CONV_LITERAL, ApiTypes::API_RUNTIME, found->second.supportDegree};
      Statistics::current().incrementCounter(counter, name.str());
      if (!Statistics::isUnsupported(counter)) {
        clang::SourceLocation sl = start.getLocWithOffset(begin + 1);
        ct::Replacement Rep(SM, sl, name.size(), repName.str());
        clang::FullSourceLoc fullSL(sl, SM);
        insertReplacement(Rep, fullSL);
      }
    }
    if (end == StringRef::npos) break;
    begin = end + 1;
  }
}

// TODO
clang::SourceLocation AscifyAction::GetSubstrLocation(const std::string &str, const clang::SourceRange &sr) {
  clang::SourceLocation sl(sr.getBegin());
  return sl;
}

void AscifyAction::FindAndReplace(StringRef name,
                                  clang::SourceLocation sl,
                                  const std::map<StringRef, dppCounter> &repMap,
                                  bool bReplace) {
  const auto found = repMap.find(name);
  if (found == repMap.end()) {
    // So it's an identifier, but not CUDA? Boring.
    return;
  }
  Statistics::current().incrementCounter(found->second, name.str());
  clang::DiagnosticsEngine &DE = getCompilerInstance().getDiagnostics();

  // Warn about the deprecated identifier in CUDA but hipify it.
  if (Statistics::isCudaDeprecated(found->second)) {
    const auto ID = DE.getCustomDiagID(clang::DiagnosticsEngine::Warning, "'%0' is deprecated in CUDA.");
    DE.Report(sl, ID) << found->first;
  }

  // TODO: Similar to hipify, add statistics analysis

    // Warn about the unsupported identifier.
  if (Statistics::isUnsupported(found->second)) {
    std::string sWarn = sDPP;
    const auto ID = DE.getCustomDiagID(clang::DiagnosticsEngine::Warning, "'%0' is unsupported in '%1'.");
    DE.Report(sl, ID) << found->first << sWarn;
    return;
  }
  if (!bReplace) {
    return;
  }
  StringRef repName = found->second.dppName;
  auto &SM = getCompilerInstance().getSourceManager();
  ct::Replacement Rep(SM, sl, name.size(), repName.str());
  clang::FullSourceLoc fullSL(sl, SM);
  insertReplacement(Rep, fullSL);
}

namespace {

clang::SourceRange getReadRange(clang::SourceManager &SM, const clang::SourceRange &exprRange) {
  clang::SourceLocation begin = exprRange.getBegin();
  clang::SourceLocation end = exprRange.getEnd();
  bool beginSafe = !SM.isMacroBodyExpansion(begin) || clang::Lexer::isAtStartOfMacroExpansion(begin, SM, clang::LangOptions{});
  bool endSafe = !SM.isMacroBodyExpansion(end) || clang::Lexer::isAtEndOfMacroExpansion(end, SM, clang::LangOptions{});
  if (beginSafe && endSafe) {
    return {SM.getFileLoc(begin), SM.getFileLoc(end)};
  } else {
    return {SM.getSpellingLoc(begin), SM.getSpellingLoc(end)};
  }
}

clang::SourceRange getWriteRange(clang::SourceManager &SM, const clang::SourceRange &exprRange) {
  clang::SourceLocation begin = exprRange.getBegin();
  clang::SourceLocation end = exprRange.getEnd();
  // If the range is contained within a macro, update the macro definition.
  // Otherwise, use the file location and hope for the best.
  if (!SM.isMacroBodyExpansion(begin) || !SM.isMacroBodyExpansion(end)) {
    return {SM.getFileLoc(begin), SM.getFileLoc(end)};
  }
  return {SM.getSpellingLoc(begin), SM.getSpellingLoc(end)};
}

StringRef readSourceText(clang::SourceManager &SM, const clang::SourceRange &exprRange) {
  return clang::Lexer::getSourceText(clang::CharSourceRange::getTokenRange(getReadRange(SM, exprRange)), SM, clang::LangOptions(), nullptr);
}

static const clang::Token *previousSignificantToken(
    const std::deque<clang::Token> &tokens) {
  if (tokens.size() < 2)
    return nullptr;
  for (std::size_t i = tokens.size() - 1; i > 0;) {
    --i;
    if (tokens[i].getKind() != 0)
      return &tokens[i];
  }
  return nullptr;
}

static bool isUnqualifiedFloatMathName(llvm::StringRef name) {
  return name == "exp" || name == "log" || name == "sqrt" || name == "rsqrt";
}

static bool requiresCudaCompatHeader(llvm::StringRef name) {
  return name == "__global__" ||
         name == "cudaMalloc" ||
         name == "cudaFree" ||
         name == "cudaMemcpy" ||
         name == "cudaMemcpyAsync" ||
         name == "cudaMemset" ||
         name == "cudaMemsetAsync" ||
         name == "cudaMallocHost" ||
         name == "cudaFreeHost" ||
         name == "cudaGetDevice" ||
         name == "cudaSetDevice" ||
         name == "cudaGetDeviceCount" ||
         name == "cudaDeviceSynchronize" ||
         name == "cudaDeviceReset" ||
         name == "cudaGetLastError" ||
         name == "cudaGetErrorString" ||
         name == "cudaDeviceGetAttribute" ||
         name == "cudaStreamCreate" ||
         name == "cudaStreamCreateWithFlags" ||
         name == "cudaStreamDefault" ||
         name == "cudaStreamNonBlocking" ||
         name == "cudaStreamDestroy" ||
         name == "cudaStreamSynchronize" ||
         name == "cudaEventCreate" ||
         name == "cudaEventDestroy" ||
         name == "cudaEventRecord" ||
         name == "cudaEventSynchronize" ||
         name == "cudaEventElapsedTime" ||
         name == "cudaDevAttrWarpSize" ||
         name == "cudaDevAttrMaxThreadsPerMultiProcessor" ||
         name == "cudaDevAttrMaxThreadsPerBlock" ||
         name == "cudaDevAttrMaxSharedMemoryPerBlock" ||
         name == "cudaDevAttrMaxSharedMemoryPerBlockOptin" ||
         name == "cudaPeekAtLastError" ||
         name == "cudaOccupancyMaxActiveBlocksPerMultiprocessor" ||
         name == "__shfl_sync" ||
         name == "__shfl_up_sync" ||
         name == "__shfl_down_sync" ||
         name == "__shfl_xor_sync" ||
         name == "__syncthreads" ||
         name == "__syncthreads_and" ||
         name == "__syncthreads_or" ||
         name == "__syncthreads_count" ||
         name == "__syncwarp" ||
         name == "__threadfence" ||
         name == "__threadfence_block" ||
         name == "__threadfence_system" ||
         name == "__expf" ||
         name == "__logf" ||
         name == "__log2f" ||
         name == "__log10f" ||
         name == "__fdividef" ||
         name == "__frsqrt_rn" ||
         name == "__fsqrt_rn" ||
         name == "exp" ||
         name == "log" ||
         name == "sqrt" ||
         name == "rsqrt" ||
         name == "__forceinline__" ||
         name == "__align__" ||
         name == "nv_bfloat16" ||
         name == "nv_bfloat16_2" ||
         name == "cudaFuncAttributes" ||
         name == "cudaFuncAttribute" ||
         name == "cudaFuncAttributeMaxDynamicSharedMemorySize" ||
         name == "cudaFuncGetAttributes" ||
         name == "cudaFuncSetAttribute";
}

static const clang::Expr *stripParenAndImplicitCasts(const clang::Expr *expr) {
  if (expr == nullptr)
    return nullptr;
  if (const auto *defaultArgument =
          llvm::dyn_cast<clang::CXXDefaultArgExpr>(expr))
    expr = defaultArgument->getExpr();
  return expr->IgnoreParenImpCasts();
}

static const clang::VarDecl *referencedVariable(const clang::Expr *expr) {
  expr = stripParenAndImplicitCasts(expr);
  const auto *reference = llvm::dyn_cast_or_null<clang::DeclRefExpr>(expr);
  return reference == nullptr
             ? nullptr
             : llvm::dyn_cast<clang::VarDecl>(reference->getDecl());
}

static bool evaluatesToNonNegative(const clang::Expr *expr,
                                   uint64_t expected,
                                   clang::ASTContext &context) {
  if (expr == nullptr)
    return false;
  expr = stripParenAndImplicitCasts(expr);
  clang::Expr::EvalResult result;
  if (!expr->EvaluateAsInt(result, context) || !result.Val.isInt())
    return false;
  const llvm::APSInt &value = result.Val.getInt();
  if ((value.isSigned() && value.isNegative()) ||
      value.getBitWidth() > 64)
    return false;
  return value.getZExtValue() == expected;
}

static bool evaluatesToFullWarpMask(const clang::Expr *expr,
                                    clang::ASTContext &context) {
  if (expr == nullptr || !expr->getType()->isIntegerType() ||
      context.getTypeSize(expr->getType()) != 32)
    return false;

  expr = stripParenAndImplicitCasts(expr);
  clang::Expr::EvalResult result;
  if (!expr->EvaluateAsInt(result, context) || !result.Val.isInt())
    return false;
  const llvm::APSInt &value = result.Val.getInt();
  const llvm::APInt normalized = llvm::APInt(value).zextOrTrunc(32);
  return normalized == llvm::APInt(32, UINT32_MAX);
}

static bool isSupportedWarpAddType(clang::QualType type,
                                   clang::ASTContext &context) {
  if (type.isNull() || type->isReferenceType() ||
      type.isVolatileQualified())
    return false;

  type = type.getCanonicalType().getUnqualifiedType();
  const auto *builtin = type->getAs<clang::BuiltinType>();
  if (builtin == nullptr || context.getTypeSize(type) != 32)
    return false;
  return builtin->getKind() == clang::BuiltinType::Float ||
         builtin->getKind() == clang::BuiltinType::Int ||
         builtin->getKind() == clang::BuiltinType::UInt;
}

static const clang::Stmt *singleLoopUpdate(const clang::Stmt *body) {
  const auto *compound = llvm::dyn_cast_or_null<clang::CompoundStmt>(body);
  if (compound == nullptr)
    return body;
  if (compound->size() != 1)
    return nullptr;
  return *compound->body_begin();
}

static const clang::Stmt *directFunctionBodyStatement(
    const clang::ForStmt *loop,
    const clang::FunctionDecl *function) {
  const auto *body =
      llvm::dyn_cast_or_null<clang::CompoundStmt>(function->getBody());
  if (body == nullptr)
    return nullptr;
  for (const clang::Stmt *statement : body->body()) {
    const clang::Stmt *unwrapped = statement;
    while (const auto *attributed =
               llvm::dyn_cast<clang::AttributedStmt>(unwrapped))
      unwrapped = attributed->getSubStmt();
    if (unwrapped == loop)
      return statement;
  }
  return nullptr;
}

static bool isOnlyUnrollPragmasAndWhitespace(llvm::StringRef source) {
  bool sawUnrollPragma = false;
  while (!source.empty()) {
    const std::pair<llvm::StringRef, llvm::StringRef> line =
        source.split('\n');
    source = line.second;
    const llvm::StringRef trimmed = line.first.trim();
    if (trimmed.empty())
      continue;
    if (trimmed != "#pragma unroll")
      return false;
    sawUnrollPragma = true;
  }
  return sawUnrollPragma;
}

// When Clang wraps a loop in AttributedStmt for `#pragma unroll`, own the
// complete physical pragma line as well as the loop.  Starting at column one
// prevents leaving a dangling `#pragma` fragment if an LLVM version reports
// the attribute location at a token after '#'.
static clang::SourceLocation canonicalWarpReductionReplacementBegin(
    const clang::Stmt *ownedStatement,
    const clang::ForStmt *loop,
    clang::SourceManager &sourceManager,
    const clang::LangOptions &langOptions) {
  const clang::SourceLocation loopBegin =
      sourceManager.getFileLoc(loop->getBeginLoc());
  if (ownedStatement == loop)
    return loopBegin;

  const clang::SourceLocation ownedBegin =
      sourceManager.getFileLoc(ownedStatement->getBeginLoc());
  if (ownedBegin.isInvalid() || loopBegin.isInvalid())
    return clang::SourceLocation();
  const std::pair<clang::FileID, unsigned> decomposedOwned =
      sourceManager.getDecomposedLoc(ownedBegin);
  const std::pair<clang::FileID, unsigned> decomposedLoop =
      sourceManager.getDecomposedLoc(loopBegin);
  if (decomposedOwned.first != decomposedLoop.first ||
      decomposedOwned.second >= decomposedLoop.second)
    return clang::SourceLocation();

  const unsigned line = sourceManager.getSpellingLineNumber(ownedBegin);
  const clang::SourceLocation lineBegin =
      sourceManager.translateLineCol(decomposedOwned.first, line, 1);
  if (lineBegin.isInvalid())
    return clang::SourceLocation();
  const llvm::StringRef prefix = clang::Lexer::getSourceText(
      clang::CharSourceRange::getCharRange(lineBegin, loopBegin),
      sourceManager, langOptions);
  if (!isOnlyUnrollPragmasAndWhitespace(prefix))
    return clang::SourceLocation();
  return lineBegin;
}

static bool matchCanonicalWarpAddReduction(
    const clang::ForStmt *loop,
    clang::ASTContext &context,
    clang::SourceManager &sourceManager,
    const clang::VarDecl *&accumulator) {
  accumulator = nullptr;
  if (loop == nullptr || loop->getInit() == nullptr ||
      loop->getCond() == nullptr || loop->getInc() == nullptr ||
      loop->getBody() == nullptr)
    return false;

  const clang::SourceLocation begin = loop->getBeginLoc();
  const clang::SourceLocation end = loop->getEndLoc();
  if (begin.isInvalid() || end.isInvalid() || begin.isMacroID() ||
      end.isMacroID() || !sourceManager.isWrittenInMainFile(begin) ||
      !sourceManager.isWrittenInMainFile(end))
    return false;

  const auto *declaration =
      llvm::dyn_cast<clang::DeclStmt>(loop->getInit());
  if (declaration == nullptr || !declaration->isSingleDecl())
    return false;
  const auto *offset =
      llvm::dyn_cast<clang::VarDecl>(declaration->getSingleDecl());
  if (offset == nullptr || !offset->hasInit() ||
      offset->getStorageDuration() != clang::SD_Automatic ||
      !offset->getType()->isIntegerType() ||
      offset->getType().isVolatileQualified() ||
      context.getTypeSize(offset->getType()) != 32 ||
      !evaluatesToNonNegative(offset->getInit(), 16, context))
    return false;

  const auto *condition = llvm::dyn_cast<clang::BinaryOperator>(
      stripParenAndImplicitCasts(loop->getCond()));
  if (condition == nullptr || condition->getOpcode() != clang::BO_GT ||
      referencedVariable(condition->getLHS()) != offset ||
      !evaluatesToNonNegative(condition->getRHS(), 0, context))
    return false;

  const auto *increment = llvm::dyn_cast<clang::CompoundAssignOperator>(
      stripParenAndImplicitCasts(loop->getInc()));
  if (increment == nullptr || referencedVariable(increment->getLHS()) != offset)
    return false;
  if (increment->getOpcode() == clang::BO_DivAssign) {
    if (!evaluatesToNonNegative(increment->getRHS(), 2, context))
      return false;
  } else if (increment->getOpcode() == clang::BO_ShrAssign) {
    if (!evaluatesToNonNegative(increment->getRHS(), 1, context))
      return false;
  } else {
    return false;
  }

  const clang::Stmt *updateStatement = singleLoopUpdate(loop->getBody());
  const auto *updateExpression =
      llvm::dyn_cast_or_null<clang::Expr>(updateStatement);
  if (updateExpression == nullptr)
    return false;
  updateExpression = stripParenAndImplicitCasts(updateExpression);

  const clang::CallExpr *shuffle = nullptr;
  if (const auto *compoundAdd =
          llvm::dyn_cast<clang::CompoundAssignOperator>(updateExpression)) {
    if (compoundAdd->getOpcode() != clang::BO_AddAssign)
      return false;
    accumulator = referencedVariable(compoundAdd->getLHS());
    shuffle = llvm::dyn_cast_or_null<clang::CallExpr>(
        stripParenAndImplicitCasts(compoundAdd->getRHS()));
  } else if (const auto *assignment =
                 llvm::dyn_cast<clang::BinaryOperator>(updateExpression)) {
    if (assignment->getOpcode() != clang::BO_Assign)
      return false;
    accumulator = referencedVariable(assignment->getLHS());
    const auto *addition = llvm::dyn_cast_or_null<clang::BinaryOperator>(
        stripParenAndImplicitCasts(assignment->getRHS()));
    if (addition == nullptr || addition->getOpcode() != clang::BO_Add)
      return false;
    const clang::Expr *shuffleExpression = nullptr;
    if (referencedVariable(addition->getLHS()) == accumulator)
      shuffleExpression = addition->getRHS();
    else if (referencedVariable(addition->getRHS()) == accumulator)
      shuffleExpression = addition->getLHS();
    else
      return false;
    shuffle = llvm::dyn_cast_or_null<clang::CallExpr>(
        stripParenAndImplicitCasts(shuffleExpression));
  } else {
    return false;
  }

  if (accumulator == nullptr || accumulator == offset || shuffle == nullptr ||
      !isSupportedWarpAddType(accumulator->getType(), context))
    return false;

  const auto *function =
      llvm::dyn_cast<clang::FunctionDecl>(accumulator->getDeclContext());
  if (function == nullptr ||
      (!function->hasAttr<clang::CUDADeviceAttr>() &&
      !function->hasAttr<clang::CUDAGlobalAttr>()) ||
      function->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate ||
      function->getDeclContext()->isDependentContext())
    return false;

  const clang::Stmt *ownedStatement =
      directFunctionBodyStatement(loop, function);
  if (ownedStatement == nullptr)
    return false;

  // The transformation removes the entire for-init declaration.  Prove the
  // accumulator is a different declaration that precedes, and therefore
  // remains visible at, the replacement site.
  const clang::SourceLocation accumulatorLoc =
      sourceManager.getFileLoc(accumulator->getLocation());
  const clang::SourceLocation ownedBegin =
      sourceManager.getFileLoc(ownedStatement->getBeginLoc());
  if (accumulator->getLocation().isInvalid() ||
      accumulator->getLocation().isMacroID() ||
      accumulatorLoc.isInvalid() || ownedBegin.isInvalid())
    return false;
  const std::pair<clang::FileID, unsigned> decomposedAccumulator =
      sourceManager.getDecomposedLoc(accumulatorLoc);
  const std::pair<clang::FileID, unsigned> decomposedOwned =
      sourceManager.getDecomposedLoc(ownedBegin);
  if (decomposedAccumulator.first != decomposedOwned.first ||
      decomposedAccumulator.second >= decomposedOwned.second)
    return false;

  const clang::FunctionDecl *callee = shuffle->getDirectCallee();
  if (callee == nullptr ||
      callee->getQualifiedNameAsString() != "__shfl_xor_sync" ||
      callee->getLocation().isInvalid() ||
      !sourceManager.isInSystemHeader(callee->getLocation()) ||
      shuffle->getNumArgs() < 3 || shuffle->getNumArgs() > 4 ||
      !evaluatesToFullWarpMask(shuffle->getArg(0), context) ||
      referencedVariable(shuffle->getArg(1)) != accumulator ||
      referencedVariable(shuffle->getArg(2)) != offset)
    return false;

  const clang::Expr *width = nullptr;
  if (shuffle->getNumArgs() == 4) {
    width = shuffle->getArg(3);
  } else {
    if (callee->getNumParams() != 4 ||
        !callee->getParamDecl(3)->hasDefaultArg())
      return false;
    width = callee->getParamDecl(3)->getDefaultArg();
  }
  if (!evaluatesToNonNegative(width, 32, context))
    return false;
  return true;
}

enum class CanonicalReducerKind {
  None,
  Sum,
  Max,
  Min,
};

static const clang::ParmVarDecl *referencedParameter(
    const clang::Expr *expr) {
  expr = stripParenAndImplicitCasts(expr);
  const auto *reference =
      llvm::dyn_cast_or_null<clang::DeclRefExpr>(expr);
  return reference == nullptr
             ? nullptr
             : llvm::dyn_cast<clang::ParmVarDecl>(reference->getDecl());
}

static clang::QualType canonicalReducerValueType(clang::QualType type) {
  if (type.isNull())
    return type;
  if (type->isReferenceType())
    type = type.getNonReferenceType();
  return type.getCanonicalType().getUnqualifiedType();
}

static bool referencesBothParameters(
    const clang::Expr *lhs,
    const clang::Expr *rhs,
    const clang::ParmVarDecl *first,
    const clang::ParmVarDecl *second) {
  const clang::ParmVarDecl *lhsParameter = referencedParameter(lhs);
  const clang::ParmVarDecl *rhsParameter = referencedParameter(rhs);
  return (lhsParameter == first && rhsParameter == second) ||
         (lhsParameter == second && rhsParameter == first);
}

static CanonicalReducerKind classifyCanonicalReducerReturn(
    const clang::Expr *expression,
    const clang::ParmVarDecl *first,
    const clang::ParmVarDecl *second) {
  expression = stripParenAndImplicitCasts(expression);
  if (const auto *addition =
          llvm::dyn_cast_or_null<clang::BinaryOperator>(expression)) {
    if (addition->getOpcode() == clang::BO_Add &&
        referencesBothParameters(
            addition->getLHS(), addition->getRHS(), first, second))
      return CanonicalReducerKind::Sum;
    return CanonicalReducerKind::None;
  }

  const auto *conditional =
      llvm::dyn_cast_or_null<clang::ConditionalOperator>(expression);
  if (conditional == nullptr)
    return CanonicalReducerKind::None;
  const auto *comparison = llvm::dyn_cast<clang::BinaryOperator>(
      stripParenAndImplicitCasts(conditional->getCond()));
  if (comparison == nullptr ||
      (comparison->getOpcode() != clang::BO_GT &&
       comparison->getOpcode() != clang::BO_LT) ||
      !referencesBothParameters(
          comparison->getLHS(), comparison->getRHS(), first, second))
    return CanonicalReducerKind::None;

  const clang::ParmVarDecl *conditionLhs =
      referencedParameter(comparison->getLHS());
  const clang::ParmVarDecl *conditionRhs =
      referencedParameter(comparison->getRHS());
  const clang::ParmVarDecl *trueValue =
      referencedParameter(conditional->getTrueExpr());
  const clang::ParmVarDecl *falseValue =
      referencedParameter(conditional->getFalseExpr());
  if (trueValue == nullptr || falseValue == nullptr)
    return CanonicalReducerKind::None;

  const bool selectsConditionLhs =
      trueValue == conditionLhs && falseValue == conditionRhs;
  const bool selectsConditionRhs =
      trueValue == conditionRhs && falseValue == conditionLhs;
  if (!selectsConditionLhs && !selectsConditionRhs)
    return CanonicalReducerKind::None;

  if (comparison->getOpcode() == clang::BO_GT)
    return selectsConditionLhs ? CanonicalReducerKind::Max
                               : CanonicalReducerKind::Min;
  return selectsConditionLhs ? CanonicalReducerKind::Min
                             : CanonicalReducerKind::Max;
}

static CanonicalReducerKind classifyCanonicalBinaryReducer(
    const clang::CXXRecordDecl *record,
    clang::ASTContext &context,
    clang::SourceManager &sourceManager) {
  if (record == nullptr || !record->isThisDeclarationADefinition() ||
      record->isImplicit() || record->isUnion() ||
      record->getIdentifier() == nullptr ||
      !record->getDeclContext()->isFileContext() ||
      record->getNumBases() != 0 ||
      record->getLocation().isInvalid() ||
      record->getLocation().isMacroID() ||
      !sourceManager.isWrittenInMainFile(record->getLocation()))
    return CanonicalReducerKind::None;

  const clang::ClassTemplateDecl *classTemplate =
      record->getDescribedClassTemplate();
  if (classTemplate == nullptr ||
      classTemplate->getTemplateParameters()->size() != 1)
    return CanonicalReducerKind::None;
  const auto *valueTypeParameter =
      llvm::dyn_cast<clang::TemplateTypeParmDecl>(
          classTemplate->getTemplateParameters()->getParam(0));
  if (valueTypeParameter == nullptr ||
      valueTypeParameter->isParameterPack() ||
      valueTypeParameter->getIdentifier() == nullptr)
    return CanonicalReducerKind::None;

  if (!record->lookup(
          &context.Idents.get("ascify_reduction_tag")).empty() ||
      !record->lookup(
          &context.Idents.get("ascify_reduction_value_type")).empty() ||
      !record->lookup(
          &context.Idents.get("ascify_reduction_owner_type")).empty())
    return CanonicalReducerKind::None;

  const clang::CXXMethodDecl *callOperator = nullptr;
  for (const clang::CXXMethodDecl *method : record->methods()) {
    if (method->isImplicit() ||
        method->getOverloadedOperator() != clang::OO_Call)
      continue;
    if (callOperator != nullptr)
      return CanonicalReducerKind::None;
    callOperator = method;
  }
  if (callOperator == nullptr || callOperator->isDeleted() ||
      callOperator->isVariadic() || !callOperator->isConst() ||
      callOperator->getNumParams() != 2 ||
      callOperator->getTemplatedKind() !=
          clang::FunctionDecl::TK_NonTemplate ||
      !callOperator->hasAttr<clang::CUDADeviceAttr>() ||
      !callOperator->hasBody())
    return CanonicalReducerKind::None;

  const clang::ParmVarDecl *first = callOperator->getParamDecl(0);
  const clang::ParmVarDecl *second = callOperator->getParamDecl(1);
  clang::QualType firstType =
      canonicalReducerValueType(first->getType());
  clang::QualType secondType =
      canonicalReducerValueType(second->getType());
  clang::QualType returnType =
      canonicalReducerValueType(callOperator->getReturnType());
  const clang::QualType templateValueType =
      context.getTypeDeclType(valueTypeParameter)
          .getCanonicalType().getUnqualifiedType();
  if (firstType.isNull() || secondType.isNull() || returnType.isNull() ||
      templateValueType.isNull() ||
      first->getType().getNonReferenceType().isVolatileQualified() ||
      second->getType().getNonReferenceType().isVolatileQualified() ||
      callOperator->getReturnType()->isReferenceType() ||
      !context.hasSameType(firstType, secondType) ||
      !context.hasSameType(firstType, returnType) ||
      !context.hasSameType(firstType, templateValueType))
    return CanonicalReducerKind::None;

  const auto *body =
      llvm::dyn_cast<clang::CompoundStmt>(callOperator->getBody());
  if (body == nullptr || body->size() != 1 ||
      body->getBeginLoc().isInvalid() || body->getEndLoc().isInvalid() ||
      body->getBeginLoc().isMacroID() || body->getEndLoc().isMacroID())
    return CanonicalReducerKind::None;
  const auto *returnStatement =
      llvm::dyn_cast<clang::ReturnStmt>(*body->body_begin());
  if (returnStatement == nullptr || returnStatement->getRetValue() == nullptr)
    return CanonicalReducerKind::None;

  return classifyCanonicalReducerReturn(
      returnStatement->getRetValue(), first, second);
}

static llvm::StringRef canonicalReducerTagName(CanonicalReducerKind kind) {
  switch (kind) {
  case CanonicalReducerKind::Sum:
    return "Sum";
  case CanonicalReducerKind::Max:
    return "Max";
  case CanonicalReducerKind::Min:
    return "Min";
  case CanonicalReducerKind::None:
    break;
  }
  return "";
}

enum class GlobalAtomicScalarKind {
  Unsupported,
  Int32,
  UInt32,
};

struct GlobalAtomicSpec {
  const char *cudaName;
  const char *wrapperName;
  unsigned argumentCount;
  bool unsignedOnly;
};

static const GlobalAtomicSpec GlobalAtomicSpecs[] = {
    {"atomicAdd", "atomic_add_global", 2, false},
    {"atomicSub", "atomic_sub_global", 2, false},
    {"atomicExch", "atomic_exch_global", 2, false},
    {"atomicMax", "atomic_max_global", 2, false},
    {"atomicMin", "atomic_min_global", 2, false},
    {"atomicInc", "atomic_inc_global", 2, true},
    {"atomicDec", "atomic_dec_global", 2, true},
    {"atomicCAS", "atomic_cas_global", 3, false},
    {"atomicAnd", "atomic_and_global", 2, false},
    {"atomicOr", "atomic_or_global", 2, false},
    {"atomicXor", "atomic_xor_global", 2, false},
};

static const GlobalAtomicSpec *findGlobalAtomicSpec(
    llvm::StringRef name) {
  for (const GlobalAtomicSpec &spec : GlobalAtomicSpecs) {
    if (name == spec.cudaName)
      return &spec;
  }
  return nullptr;
}

static GlobalAtomicScalarKind classifyGlobalAtomicScalar(
    clang::QualType type,
    clang::ASTContext &context) {
  if (type.isNull() || type->isReferenceType() ||
      type.isConstQualified() || type.isVolatileQualified())
    return GlobalAtomicScalarKind::Unsupported;
  type = type.getCanonicalType().getUnqualifiedType();
  if (!type->isIntegerType() || context.getTypeSize(type) != 32)
    return GlobalAtomicScalarKind::Unsupported;
  if (type->isSignedIntegerType())
    return GlobalAtomicScalarKind::Int32;
  if (type->isUnsignedIntegerType())
    return GlobalAtomicScalarKind::UInt32;
  return GlobalAtomicScalarKind::Unsupported;
}

static GlobalAtomicScalarKind classifyGlobalAtomicPointer(
    clang::QualType type,
    clang::ASTContext &context) {
  if (type.isNull())
    return GlobalAtomicScalarKind::Unsupported;
  type = type.getCanonicalType();
  if (!type->isPointerType())
    return GlobalAtomicScalarKind::Unsupported;
  return classifyGlobalAtomicScalar(type->getPointeeType(), context);
}

static const clang::FunctionDecl *enclosingNonLambdaFunction(
    const clang::Stmt *statement,
    clang::ASTContext &context) {
  const clang::Stmt *current = statement;
  std::set<const clang::Stmt *> active;
  while (current != nullptr && active.insert(current).second) {
    const auto parents = context.getParents(*current);
    const clang::Stmt *next = nullptr;
    for (const clang::DynTypedNode &parent : parents) {
      if (parent.get<clang::LambdaExpr>() != nullptr)
        return nullptr;
      if (const auto *function = parent.get<clang::FunctionDecl>())
        return function;
      if (next == nullptr)
        next = parent.get<clang::Stmt>();
    }
    current = next;
  }
  return nullptr;
}

// This is deliberately a provenance proof, not a pointer-type guess.  Only a
// direct parameter of the current CUDA global function establishes global
// memory.  Local aliases, globals, fields, shared variables, helper parameters,
// and conditional pointer selection all fail closed.
static bool isAddressRootedAtCurrentGlobalParameter(
    const clang::Expr *expression,
    const clang::FunctionDecl *kernel,
    clang::ASTContext &context) {
  if (expression == nullptr || kernel == nullptr)
    return false;
  expression = stripParenAndImplicitCasts(expression);

  if (const auto *reference =
          llvm::dyn_cast_or_null<clang::DeclRefExpr>(expression)) {
    const auto *parameter =
        llvm::dyn_cast<clang::ParmVarDecl>(reference->getDecl());
    const auto *owner =
        parameter == nullptr
            ? nullptr
            : llvm::dyn_cast<clang::FunctionDecl>(
                  parameter->getDeclContext());
    return parameter != nullptr && owner != nullptr &&
           owner->getCanonicalDecl() == kernel->getCanonicalDecl() &&
           classifyGlobalAtomicPointer(parameter->getType(), context) !=
               GlobalAtomicScalarKind::Unsupported;
  }

  if (const auto *cast =
          llvm::dyn_cast_or_null<clang::CastExpr>(expression)) {
    if (cast->getCastKind() != clang::CK_BitCast &&
        cast->getCastKind() != clang::CK_NoOp)
      return false;
    if (classifyGlobalAtomicPointer(cast->getType(), context) ==
            GlobalAtomicScalarKind::Unsupported ||
        classifyGlobalAtomicPointer(cast->getSubExpr()->getType(), context) ==
            GlobalAtomicScalarKind::Unsupported)
      return false;
    return isAddressRootedAtCurrentGlobalParameter(
        cast->getSubExpr(), kernel, context);
  }

  if (const auto *subscript =
          llvm::dyn_cast_or_null<clang::ArraySubscriptExpr>(expression))
    return isAddressRootedAtCurrentGlobalParameter(
        subscript->getBase(), kernel, context);

  if (const auto *unary =
          llvm::dyn_cast_or_null<clang::UnaryOperator>(expression)) {
    if (unary->getOpcode() != clang::UO_AddrOf &&
        unary->getOpcode() != clang::UO_Deref)
      return false;
    return isAddressRootedAtCurrentGlobalParameter(
        unary->getSubExpr(), kernel, context);
  }

  const auto *binary =
      llvm::dyn_cast_or_null<clang::BinaryOperator>(expression);
  if (binary == nullptr ||
      (binary->getOpcode() != clang::BO_Add &&
       binary->getOpcode() != clang::BO_Sub))
    return false;
  const bool lhsPointer =
      classifyGlobalAtomicPointer(binary->getLHS()->getType(), context) !=
      GlobalAtomicScalarKind::Unsupported;
  const bool rhsPointer =
      classifyGlobalAtomicPointer(binary->getRHS()->getType(), context) !=
      GlobalAtomicScalarKind::Unsupported;
  if (binary->getOpcode() == clang::BO_Sub) {
    return lhsPointer && !rhsPointer &&
           binary->getRHS()->getType()->isIntegerType() &&
           isAddressRootedAtCurrentGlobalParameter(
               binary->getLHS(), kernel, context);
  }
  if (lhsPointer == rhsPointer)
    return false;
  const clang::Expr *pointer =
      lhsPointer ? binary->getLHS() : binary->getRHS();
  const clang::Expr *offset =
      lhsPointer ? binary->getRHS() : binary->getLHS();
  return offset->getType()->isIntegerType() &&
         isAddressRootedAtCurrentGlobalParameter(
             pointer, kernel, context);
}

static bool hasExactGlobalAtomicSignature(
    const clang::FunctionDecl *callee,
    const GlobalAtomicSpec &spec,
    GlobalAtomicScalarKind kind,
    clang::ASTContext &context) {
  if (callee == nullptr || callee->isVariadic() ||
      callee->getNumParams() != spec.argumentCount ||
      classifyGlobalAtomicPointer(
          callee->getParamDecl(0)->getType(), context) != kind ||
      classifyGlobalAtomicScalar(callee->getReturnType(), context) != kind)
    return false;
  for (unsigned index = 1; index < spec.argumentCount; ++index) {
    if (classifyGlobalAtomicScalar(
            callee->getParamDecl(index)->getType(), context) != kind)
      return false;
  }
  return true;
}

/**
  * Get a string representation of the expression `arg`, unless it's a defaulting function
  * call argument, in which case get a 0. Used for building argument lists to kernel calls.
  */
std::string stringifyZeroDefaultedArg(clang::SourceManager &SM, const clang::Expr *arg) {
  if (clang::isa<clang::CXXDefaultArgExpr>(arg)) return "0";
  else return std::string(readSourceText(SM, arg->getSourceRange()));
}

} // anonymous namespace

/**
  * Look at, and consider altering, a given token.
  *
  * If it's not a CUDA identifier, nothing happens.
  * If it's an unsupported CUDA identifier, a warning is emitted.
  * Otherwise, the source file is updated with the corresponding ascification.
  *
  * Returns true when the raw lexer was advanced past rewritten text.
  */
bool AscifyAction::isInSemanticRewriteRange(clang::SourceLocation loc) {
  if (loc.isInvalid())
    return false;
  auto &SM = getCompilerInstance().getSourceManager();
  const clang::SourceLocation fileLoc = SM.getFileLoc(loc);
  if (fileLoc.isInvalid())
    return false;
  const std::pair<clang::FileID, unsigned> decomposed =
      SM.getDecomposedLoc(fileLoc);
  for (const SemanticRewriteRange &range : SemanticRewriteRanges) {
    if (range.file == decomposed.first &&
        decomposed.second >= range.beginOffset &&
        decomposed.second < range.endOffset)
      return true;
  }
  return false;
}

bool AscifyAction::isInNvidiaSampleHelperMacroRange(
    clang::SourceLocation loc) {
  if (loc.isInvalid())
    return false;
  auto &sourceManager = getCompilerInstance().getSourceManager();
  const clang::SourceLocation expansion =
      sourceManager.getExpansionLoc(loc);
  const clang::SourceLocation fileLoc = sourceManager.getFileLoc(expansion);
  if (fileLoc.isInvalid())
    return false;
  const std::pair<clang::FileID, unsigned> decomposed =
      sourceManager.getDecomposedLoc(fileLoc);
  for (const SemanticRewriteRange &range :
       nvidiaSampleHelperMacroRanges) {
    if (range.file == decomposed.first &&
        decomposed.second >= range.beginOffset &&
        decomposed.second < range.endOffset)
      return true;
  }
  return false;
}

bool AscifyAction::hasUnsupportedNvidiaSampleHelperDeclarationUse() {
  class Visitor : public clang::RecursiveASTVisitor<Visitor> {
  public:
    Visitor(clang::SourceManager &sourceManager,
            const std::vector<SemanticRewriteRange> &supportedRanges,
            std::vector<NvidiaSampleHelperMacroCandidate> &candidates)
        : sourceManager(sourceManager), supportedRanges(supportedRanges),
          candidates(candidates) {}

    bool VisitCallExpr(clang::CallExpr *expression) {
      if (expression == nullptr)
        return true;
      const clang::FunctionDecl *callee = expression->getDirectCallee();
      if (callee == nullptr || callee->getIdentifier() == nullptr ||
          callee->getName() != "check" ||
          !locationComesFromRecognizedNvidiaSampleHelper(
              sourceManager, callee->getLocation()))
        return true;
      NvidiaSampleHelperMacroCandidate *candidate =
          candidateAt(expression->getExprLoc(),
                      NvidiaSampleHelperMacroKind::CheckCudaErrors);
      if (candidate == nullptr || expression->getNumArgs() == 0)
        return true;
      candidate->statusDomainProven = isAdmittedCudaRuntimeStatusCall(
          sourceManager, expression->getArg(0));
      return true;
    }

    bool VisitDeclRefExpr(clang::DeclRefExpr *expression) {
      return expression == nullptr
                 ? true
                 : inspect(expression->getExprLoc(),
                           expression->getDecl());
    }

    bool VisitMemberExpr(clang::MemberExpr *expression) {
      return expression == nullptr
                 ? true
                 : inspect(expression->getMemberLoc(),
                           expression->getMemberDecl());
    }

    bool VisitCXXConstructExpr(clang::CXXConstructExpr *expression) {
      return expression == nullptr
                 ? true
                 : inspect(expression->getExprLoc(),
                           expression->getConstructor());
    }

    bool VisitTypedefTypeLoc(clang::TypedefTypeLoc location) {
      const clang::TypedefType *type = location.getTypePtr();
      return inspect(location.getNameLoc(),
                     type == nullptr ? nullptr : type->getDecl());
    }

    bool VisitRecordTypeLoc(clang::RecordTypeLoc location) {
      return inspect(location.getNameLoc(), location.getDecl());
    }

    bool VisitEnumTypeLoc(clang::EnumTypeLoc location) {
      return inspect(location.getNameLoc(), location.getDecl());
    }

    bool VisitUsingShadowDecl(clang::UsingShadowDecl *declaration) {
      return declaration == nullptr
                 ? true
                 : inspect(declaration->getLocation(),
                           declaration->getTargetDecl());
    }

    bool hasConflict() const { return conflict; }
    llvm::StringRef conflictingName() const { return name; }

  private:
    NvidiaSampleHelperMacroCandidate *candidateAt(
        clang::SourceLocation location,
        NvidiaSampleHelperMacroKind kind) {
      const clang::SourceLocation expansion =
          sourceManager.getExpansionLoc(location);
      const clang::SourceLocation fileLoc =
          sourceManager.getFileLoc(expansion);
      if (fileLoc.isInvalid())
        return nullptr;
      const std::pair<clang::FileID, unsigned> decomposed =
          sourceManager.getDecomposedLoc(fileLoc);
      for (NvidiaSampleHelperMacroCandidate &candidate : candidates) {
        const SemanticRewriteRange &range = candidate.invocationRange;
        if (candidate.kind == kind && range.file == decomposed.first &&
            decomposed.second >= range.beginOffset &&
            decomposed.second < range.endOffset)
          return &candidate;
      }
      return nullptr;
    }

    bool isSupportedMacroExpansion(clang::SourceLocation location) const {
      const clang::SourceLocation expansion =
          sourceManager.getExpansionLoc(location);
      const clang::SourceLocation fileLoc =
          sourceManager.getFileLoc(expansion);
      if (fileLoc.isInvalid())
        return false;
      const std::pair<clang::FileID, unsigned> decomposed =
          sourceManager.getDecomposedLoc(fileLoc);
      for (const SemanticRewriteRange &range : supportedRanges) {
        if (range.file == decomposed.first &&
            decomposed.second >= range.beginOffset &&
            decomposed.second < range.endOffset)
          return true;
      }
      return false;
    }

    bool inspect(clang::SourceLocation useLocation,
                 const clang::NamedDecl *referenced) {
      if (conflict || referenced == nullptr)
        return !conflict;
      const clang::SourceLocation expansion =
          sourceManager.getExpansionLoc(useLocation);
      if (expansion.isInvalid() ||
          !locationComesFromRecognizedNvidiaSampleHelper(
              sourceManager, referenced->getLocation()) ||
          isSupportedMacroExpansion(useLocation) ||
          (!sourceManager.isWrittenInMainFile(expansion) &&
           locationComesFromRecognizedNvidiaSampleHelper(
               sourceManager, useLocation)))
        return true;
      conflict = true;
      name = referenced->getQualifiedNameAsString();
      return false;
    }

    clang::SourceManager &sourceManager;
    const std::vector<SemanticRewriteRange> &supportedRanges;
    std::vector<NvidiaSampleHelperMacroCandidate> &candidates;
    bool conflict = false;
    std::string name;
  } visitor(getCompilerInstance().getSourceManager(),
            nvidiaSampleHelperMacroRanges,
            nvidiaSampleHelperMacroCandidates);

  visitor.TraverseDecl(
      getCompilerInstance().getASTContext().getTranslationUnitDecl());
  if (visitor.hasConflict()) {
    llvm::errs() << "Ascify NVIDIA sample-helper closure: residual helper "
                 << "declaration '" << visitor.conflictingName()
                 << "' keeps helper_cuda.h\n";
  }
  for (const NvidiaSampleHelperMacroCandidate &candidate :
       nvidiaSampleHelperMacroCandidates) {
    if (candidate.kind ==
            NvidiaSampleHelperMacroKind::CheckCudaErrors &&
        !candidate.statusDomainProven) {
      nvidiaSampleHelperUnsupportedMacroUse = true;
      llvm::errs()
          << "Ascify NVIDIA sample-helper closure: checkCudaErrors status "
          << "domain not proven; macro and include kept\n";
    }
  }
  return visitor.hasConflict();
}

bool AscifyAction::RewriteToken(clang::Lexer &, clang::Token &tok) {
  auditRawNvidiaSampleHelperToken(tok);
  // AST semantic transformations own their complete source span.  The raw
  // identifier pass runs later and must not add overlapping token replacements
  // inside that span.
  if (isInSemanticRewriteRange(tok.getLocation()))
    return false;

  if (!AscifyAMAP) {
    clang::SourceRange sr(tok.getLocation());
    for (const auto &skipped : SkippedSourceRanges) {
      if (skipped.fullyContains(sr))
        return false;
    }
  }

  if (tok.is(clang::tok::string_literal)) {
    llvm::StringRef s(tok.getLiteralData(), tok.getLength());
    RewriteString(unquoteStr(s), tok.getLocation());
    return false;
  }
  if (!tok.isAnyIdentifier())
    return false;

  llvm::StringRef name = tok.getRawIdentifier();

  // Do not turn std::exp (or another explicitly-qualified function) into an
  // invalid std::expf spelling. Unqualified CUDA device math remains mapped.
  if (isUnqualifiedFloatMathName(name)) {
    const clang::Token *previous = previousSignificantToken(rawTokenWindow);
    if (previous != nullptr && previous->is(clang::tok::coloncolon))
      return false;
  }

  if (requiresCudaCompatHeader(name))
    needsCudaCompatHeader = true;

  FindAndReplace(name, tok.getLocation(), CUDA_RENAMES_MAP());
  return false;
}

bool AscifyAction::Exclude(const dppCounter &hipToken) {
  return false;
}

void AscifyAction::InclusionDirective(clang::SourceLocation hash_loc,
                                      const clang::Token&,
                                      StringRef file_name,
                                      bool is_angled,
                                      clang::CharSourceRange filename_range,
                                      StringRef resolved_file_name, StringRef,
                                      StringRef, const clang::Module*) {
  std::string frontendCompatibilityError;
  if (!ascify::ValidateFrontendCompatibilityInclude(
          frontendCompatibility, file_name.str(), resolved_file_name.str(),
          frontendCompatibilityError)) {
    clang::DiagnosticsEngine &DE = getCompilerInstance().getDiagnostics();
    const auto ID = DE.getCustomDiagID(
        clang::DiagnosticsEngine::Error, "%0");
    DE.Report(filename_range.getBegin(), ID) << frontendCompatibilityError;
    return;
  }
  outs() << "File included: " << file_name << "\n";
  auto &SM = getCompilerInstance().getSourceManager();
  const bool recognizedNvidiaHelper =
      isRecognizedNvidiaSampleHelperCuda(resolved_file_name);
  if (!SM.isWrittenInMainFile(hash_loc)) {
    if (recognizedNvidiaHelper) {
      nvidiaSampleHelperUnsupportedMacroUse = true;
      llvm::errs()
          << "Ascify NVIDIA sample-helper closure: recognized helper was "
          << "included transitively; all helper edits kept\n";
    }
    return;
  }
  if (recognizedNvidiaHelper) {
    const clang::SourceLocation filenameBegin = filename_range.getBegin();
    const clang::SourceLocation filenameEnd = filename_range.getEnd();
    const clang::SourceLocation hashFile = SM.getFileLoc(hash_loc);
    const clang::SourceLocation beginFile = SM.getFileLoc(filenameBegin);
    const clang::SourceLocation endFile = SM.getFileLoc(filenameEnd);
    bool invalidSpelling = false;
    const llvm::StringRef spelling = clang::Lexer::getSourceText(
        filename_range, SM, getCompilerInstance().getLangOpts(),
        &invalidSpelling);
    const std::string filename = file_name.str();
    const std::string quoted = "\"" + filename + "\"";
    const std::string angled = "<" + filename + ">";
    const std::string quotedWithoutEnd = "\"" + filename;
    const std::string angledWithoutEnd = "<" + filename;
    const bool directSpelling =
        !invalidSpelling &&
        (spelling == file_name || spelling == quoted ||
         spelling == angled || spelling == quotedWithoutEnd ||
         spelling == angledWithoutEnd);
    const bool directRange =
        hash_loc.isFileID() && filenameBegin.isFileID() &&
        filenameEnd.isFileID() && hashFile.isValid() &&
        beginFile.isValid() && endFile.isValid() &&
        SM.isWrittenInMainFile(beginFile) &&
        SM.isWrittenInMainFile(endFile) &&
        SM.getFileID(hashFile) == SM.getFileID(beginFile) &&
        SM.getFileID(beginFile) == SM.getFileID(endFile);
    if (file_name == "helper_cuda.h" && directRange && directSpelling) {
      nvidiaSampleHelperIncludes.push_back({hash_loc, filenameEnd});
      recognizedNvidiaSampleHelperPaths.insert(resolved_file_name.str());
    } else {
      // A TU may contain both a removable direct include and another include
      // that reaches the same recognized helper through a macro. The latter
      // cannot be edited by a direct source range, so it invalidates the
      // entire helper transaction instead of being ignored in cardinality.
      nvidiaSampleHelperUnsupportedMacroUse = true;
      llvm::errs()
          << "Ascify NVIDIA sample-helper closure: helper include is not a "
          << "direct main-file literal; include and calls kept\n";
    }
  }
  if (file_name == "ascify/ascify_cuda_compat.hpp")
    hasCudaCompatHeader = true;
  if (file_name == ascify::DavC310TargetRecipe::TargetHeader)
    hasDavC310TargetHeader = true;
  if (file_name == ascify::DavC310TargetRecipe::SimdTargetHeader)
    hasDavC310SimdTargetHeader = true;
  if (file_name == "cub/cub.cuh" ||
      file_name == "acl_cub/aclcub.hpp") {
    hasCubCompatHeader = true;
    if (firstCubCompatHeaderLoc.isInvalid())
      firstCubCompatHeaderLoc = hash_loc;
  }
  if (!firstHeader) {
    firstHeader = true;
    firstHeaderLoc = hash_loc;
  }
  const auto found = CUDA_INCLUDE_MAP.find(file_name);
  if (found != CUDA_INCLUDE_MAP.end()) {
    bool exclude = Exclude(found->second);
    Statistics::current().incrementCounter(found->second, file_name.str());
    clang::SourceLocation sl = filename_range.getBegin();

    if (Statistics::isUnsupported(found->second)) {
      clang::DiagnosticsEngine &DE = getCompilerInstance().getDiagnostics();
      std::string sWarn = sDPP;
      const auto ID = DE.getCustomDiagID(
          clang::DiagnosticsEngine::Warning,
          "'%0' is unsupported header in '%1'.");
      DE.Report(sl, ID) << found->first << sWarn;
      return;
    }
    clang::StringRef newInclude;
    // Keep the same include type that the user gave.
    if (!exclude) {
      clang::SmallString<128> includeBuffer;
      llvm::StringRef name = found->second.dppName;
      if (is_angled)
        newInclude = llvm::Twine("<" + name + ">").toStringRef(includeBuffer);
      else
        newInclude =
            llvm::Twine("\"" + name + "\"").toStringRef(includeBuffer);
    } else {
      // hashLoc is location of the '#', thus replacing the whole include
      // directive by empty newInclude starting with '#'.
      sl = hash_loc;
    }
    const char *B = SM.getCharacterData(sl);
    const char *E = SM.getCharacterData(filename_range.getEnd());
    ct::Replacement Rep(SM, sl, E - B, newInclude.str());
    const clang::SourceLocation replacementEnd =
        sl.getLocWithOffset(static_cast<int>(E - B));
    // The raw identifier pass also sees tokens inside include spellings (for
    // example both `cub` tokens in <cub/cub.cuh>).  Make the preprocessor's
    // include rewrite the sole owner of that span.
    insertSemanticReplacement(
        Rep, clang::FullSourceLoc{sl, SM}, sl, replacementEnd);
    return;
  }

  if (localHeaderContext == nullptr)
    return;

  const clang::SourceLocation fileLoc =
      SM.getFileLoc(filename_range.getBegin());
  const bool isLiteral = !filename_range.getBegin().isMacroID();
  const unsigned sourceOffset = fileLoc.isValid() ? SM.getFileOffset(fileLoc)
                                                   : 0;
  const std::string resolvedPath = resolved_file_name.str();
  const LocalHeaderIncludeDecision decision = localHeaderContext->observe(
      sourceOffset, file_name.str(), resolvedPath, is_angled, isLiteral);
  if (decision.fatal) {
    clang::DiagnosticsEngine &DE = getCompilerInstance().getDiagnostics();
    const auto ID = DE.getCustomDiagID(clang::DiagnosticsEngine::Error,
                                       "local-header closure failed: %0");
    DE.Report(hash_loc, ID) << decision.diagnostic;
    return;
  }
  if (!decision.redirect)
    return;

  clang::SourceLocation sl = filename_range.getBegin();
  const char *B = SM.getCharacterData(sl);
  const char *E = SM.getCharacterData(filename_range.getEnd());
  ct::Replacement Rep(SM, sl, E - B, decision.emittedSpelling);
  const clang::SourceLocation replacementEnd =
      sl.getLocWithOffset(static_cast<int>(E - B));
  insertSemanticReplacement(
      Rep, clang::FullSourceLoc{sl, SM}, sl, replacementEnd);
}

void AscifyAction::PragmaDirective(clang::SourceLocation Loc, clang::PragmaIntroducerKind Introducer) {

}

bool AscifyAction::cudaLaunchKernel(const mat::MatchFinder::MatchResult &Result) {
  return true;
}

bool AscifyAction::lowerCudaGlobalScalarDoubleParam(
    const mat::MatchFinder::MatchResult &Result) {
  const auto *param =
      Result.Nodes.getNodeAs<clang::ParmVarDecl>(sCudaGlobalScalarDoubleParam);
  if (param == nullptr)
    return false;

  const auto *function =
      llvm::dyn_cast<clang::FunctionDecl>(param->getDeclContext());
  if (function == nullptr || !function->hasAttr<clang::CUDAGlobalAttr>())
    return false;

  const clang::TypeSourceInfo *typeSource = param->getTypeSourceInfo();
  if (typeSource == nullptr)
    return false;

  const clang::BuiltinTypeLoc builtinTypeLoc =
      typeSource->getTypeLoc().getUnqualifiedLoc().getAs<clang::BuiltinTypeLoc>();
  if (!builtinTypeLoc ||
      builtinTypeLoc.getTypePtr()->getKind() != clang::BuiltinType::Double)
    return false;

  auto &SM = getCompilerInstance().getSourceManager();
  const clang::LangOptions &LO = getCompilerInstance().getLangOpts();
  const clang::SourceLocation doubleLoc = builtinTypeLoc.getBuiltinLoc();
  if (doubleLoc.isInvalid() || doubleLoc.isMacroID() ||
      !SM.isWrittenInMainFile(doubleLoc))
    return false;

  const llvm::StringRef spelling = clang::Lexer::getSourceText(
      clang::CharSourceRange::getTokenRange(
          clang::SourceRange(doubleLoc, doubleLoc)),
      SM, LO);
  if (spelling != "double")
    return false;

  const unsigned offset = SM.getFileOffset(doubleLoc);
  if (!loweredDeviceDoubleParamOffsets.insert(offset).second)
    return true;

  static const dppCounter counter = {
      "float", CONV_DEVICE_TYPE, API_RUNTIME, 0, FULL};
  Statistics::current().incrementCounter(
      counter, "CUDA __global__ by-value scalar double parameter");

  clang::DiagnosticsEngine &DE = getCompilerInstance().getDiagnostics();
  const auto ID = DE.getCustomDiagID(
      clang::DiagnosticsEngine::Warning,
      "lowering by-value scalar double parameter '%0' of CUDA __global__ "
      "function '%1' to float for Ascend SIMT; use "
      "'-no-lower-device-double-params' to disable");
  DE.Report(doubleLoc, ID)
      << param->getName() << function->getQualifiedNameAsString();

  ct::Replacement Rep(SM, doubleLoc, spelling.size(), "float");
  const clang::SourceLocation doubleEnd =
      clang::Lexer::getLocForEndOfToken(doubleLoc, 0, SM, LO);
  if (!insertSemanticReplacement(
          Rep, clang::FullSourceLoc(doubleLoc, SM), doubleLoc, doubleEnd)) {
    loweredDeviceDoubleParamOffsets.erase(offset);
    return false;
  }
  return true;
}

bool AscifyAction::rewriteCudaDefaultDim3(
    const mat::MatchFinder::MatchResult &Result) {
  const auto *variable =
      Result.Nodes.getNodeAs<clang::VarDecl>(sCudaDefaultDim3);
  if (variable == nullptr || Result.SourceManager == nullptr ||
      variable->isImplicit() || llvm::isa<clang::ParmVarDecl>(variable) ||
      variable->getIdentifier() == nullptr || !variable->isLocalVarDecl() ||
      variable->hasExternalStorage())
    return false;

  // This lowering repairs host launch-shape declarations only. Global,
  // device, kernel, shared, and extern declarations retain their source
  // semantics and fail closed if the target cannot compile them.
  const auto *function =
      llvm::dyn_cast<clang::FunctionDecl>(variable->getDeclContext());
  if (function == nullptr ||
      function->hasAttr<clang::CUDADeviceAttr>() ||
      function->hasAttr<clang::CUDAGlobalAttr>())
    return false;

  auto &sourceManager = *Result.SourceManager;
  const clang::CXXRecordDecl *record =
      variable->getType().getUnqualifiedType()->getAsCXXRecordDecl();
  if (record == nullptr || record->getQualifiedNameAsString() != "dim3" ||
      record->getLocation().isInvalid() ||
      !sourceManager.isInSystemHeader(record->getLocation()))
    return false;

  const clang::SourceLocation nameLocation = variable->getLocation();
  if (nameLocation.isInvalid() || nameLocation.isMacroID() ||
      !sourceManager.isWrittenInMainFile(nameLocation))
    return false;

  const clang::LangOptions &langOptions =
      getCompilerInstance().getLangOpts();
  const clang::SourceLocation nameEnd =
      clang::Lexer::getLocForEndOfToken(
          nameLocation, 0, sourceManager, langOptions);
  if (nameEnd.isInvalid())
    return false;

  // A CUDA default-initialized declaration has no source initializer. Match
  // that syntax directly instead of depending on whether a Clang release
  // materializes the implicit CXXConstructExpr in the AST. Explicit (), {},
  // assignment, macro, attribute, array, and user-defined dim3 forms remain
  // untouched.
  clang::Token nextToken;
  if (clang::Lexer::getRawToken(
          nameEnd, nextToken, sourceManager, langOptions,
          /*IgnoreWhiteSpace=*/true) ||
      (nextToken.getKind() != clang::tok::comma &&
       nextToken.getKind() != clang::tok::semi))
    return false;

  const llvm::StringRef sourceName = clang::Lexer::getSourceText(
      clang::CharSourceRange::getTokenRange(
          clang::SourceRange(nameLocation, nameLocation)),
      sourceManager, langOptions);
  if (sourceName != variable->getName())
    return false;

  const unsigned offset = sourceManager.getFileOffset(nameLocation);
  if (!rewrittenCudaDefaultDim3Offsets.insert(offset).second)
    return true;

  static const dppCounter counter = {
      "dim3", CONV_TYPE, API_RUNTIME, 0, FULL};
  Statistics::current().incrementCounter(
      counter, "CUDA host-local default dim3 construction");

  const std::string replacement =
      sourceName.str() + " = dim3(1U, 1U, 1U)";
  ct::Replacement Rep(
      sourceManager, nameLocation, sourceName.size(), replacement);
  if (!insertSemanticReplacement(
          Rep, clang::FullSourceLoc(nameLocation, sourceManager),
          nameLocation, nameEnd)) {
    rewrittenCudaDefaultDim3Offsets.erase(offset);
    return false;
  }
  return true;
}

bool AscifyAction::rewriteProvenGlobalAtomicCall(
    const mat::MatchFinder::MatchResult &Result) {
  const auto *call =
      Result.Nodes.getNodeAs<clang::CallExpr>(sProvenGlobalAtomicCall);
  if (call == nullptr || Result.Context == nullptr ||
      Result.SourceManager == nullptr)
    return false;

  const clang::FunctionDecl *callee = call->getDirectCallee();
  if (callee == nullptr || callee->getIdentifier() == nullptr)
    return false;
  const GlobalAtomicSpec *spec =
      findGlobalAtomicSpec(callee->getName());
  if (spec == nullptr || call->getNumArgs() != spec->argumentCount ||
      callee->getQualifiedNameAsString() != spec->cudaName ||
      !callee->hasAttr<clang::CUDADeviceAttr>())
    return false;

  auto &context = *Result.Context;
  auto &sourceManager = *Result.SourceManager;
  if (callee->getLocation().isInvalid() ||
      !sourceManager.isInSystemHeader(callee->getLocation()))
    return false;
  const clang::FunctionDecl *kernel =
      enclosingNonLambdaFunction(call, context);
  if (kernel == nullptr || !kernel->hasAttr<clang::CUDAGlobalAttr>() ||
      call->getArg(0) == nullptr)
    return false;

  const GlobalAtomicScalarKind kind = classifyGlobalAtomicPointer(
      call->getArg(0)->getType(), context);
  if (kind == GlobalAtomicScalarKind::Unsupported ||
      (spec->unsignedOnly && kind != GlobalAtomicScalarKind::UInt32) ||
      !hasExactGlobalAtomicSignature(callee, *spec, kind, context))
    return false;
  for (unsigned index = 1; index < spec->argumentCount; ++index) {
    if (classifyGlobalAtomicScalar(
            call->getArg(index)->getType(), context) != kind)
      return false;
  }
  if (!isAddressRootedAtCurrentGlobalParameter(
          call->getArg(0), kernel, context))
    return false;

  const clang::Expr *calleeExpression =
      stripParenAndImplicitCasts(call->getCallee());
  const auto *reference =
      llvm::dyn_cast_or_null<clang::DeclRefExpr>(calleeExpression);
  if (reference == nullptr || reference->getDecl()->getCanonicalDecl() !=
                                  callee->getCanonicalDecl())
    return false;
  const clang::SourceLocation identifierLocation =
      reference->getLocation();
  if (identifierLocation.isInvalid() || identifierLocation.isMacroID() ||
      !sourceManager.isWrittenInMainFile(identifierLocation))
    return false;

  const clang::LangOptions &languageOptions =
      getCompilerInstance().getLangOpts();
  const llvm::StringRef spelling = clang::Lexer::getSourceText(
      clang::CharSourceRange::getTokenRange(
          clang::SourceRange(identifierLocation, identifierLocation)),
      sourceManager, languageOptions);
  if (spelling != spec->cudaName)
    return false;
  const unsigned offset = sourceManager.getFileOffset(identifierLocation);
  if (!rewrittenGlobalAtomicOffsets.insert(offset).second)
    return true;

  const clang::SourceLocation identifierEnd =
      clang::Lexer::getLocForEndOfToken(
          identifierLocation, 0, sourceManager, languageOptions);
  if (identifierEnd.isInvalid()) {
    rewrittenGlobalAtomicOffsets.erase(offset);
    return false;
  }
  const std::string replacement =
      std::string("ascify::") + spec->wrapperName;
  ct::Replacement edit(
      sourceManager, identifierLocation, spelling.size(), replacement);
  if (!insertSemanticReplacement(
          edit,
          clang::FullSourceLoc(identifierLocation, sourceManager),
          identifierLocation,
          identifierEnd)) {
    rewrittenGlobalAtomicOffsets.erase(offset);
    return false;
  }

  static const dppCounter counter = {
      "ascify global atomic wrapper",
      CONV_DEVICE_FUNC, API_RUNTIME, 0, FULL};
  Statistics::current().incrementCounter(
      counter, std::string("AST-proven ") + spec->cudaName);
  needsCudaCompatHeader = true;
  return true;
}

bool AscifyAction::rewriteCanonicalWarpAddReduction(
    const mat::MatchFinder::MatchResult &Result) {
  const auto *loop =
      Result.Nodes.getNodeAs<clang::ForStmt>(sCanonicalWarpAddReduction);
  if (loop == nullptr || Result.Context == nullptr ||
      Result.SourceManager == nullptr)
    return false;

  const clang::VarDecl *accumulator = nullptr;
  if (!matchCanonicalWarpAddReduction(
          loop, *Result.Context, *Result.SourceManager, accumulator) ||
      accumulator == nullptr || accumulator->getName().empty())
    return false;

  auto &SM = *Result.SourceManager;
  const clang::LangOptions &LO = getCompilerInstance().getLangOpts();
  const auto *function =
      llvm::dyn_cast<clang::FunctionDecl>(accumulator->getDeclContext());
  if (function == nullptr)
    return false;
  const clang::Stmt *ownedStatement =
      directFunctionBodyStatement(loop, function);
  if (ownedStatement == nullptr)
    return false;
  const clang::SourceLocation begin =
      canonicalWarpReductionReplacementBegin(
          ownedStatement, loop, SM, LO);
  const clang::SourceLocation endToken =
      SM.getFileLoc(ownedStatement->getEndLoc());
  const clang::SourceLocation end =
      clang::Lexer::getLocForEndOfToken(endToken, 0, SM, LO);
  if (begin.isInvalid() || end.isInvalid())
    return false;

  const unsigned offset = SM.getFileOffset(begin);
  if (!rewrittenWarpAddReductionOffsets.insert(offset).second)
    return true;

  const unsigned endOffset = SM.getFileOffset(end);
  if (endOffset <= offset) {
    rewrittenWarpAddReductionOffsets.erase(offset);
    return false;
  }

  const std::string name = accumulator->getNameAsString();
  std::string indentation;
  if (begin != SM.getFileLoc(loop->getBeginLoc())) {
    const unsigned loopColumn =
        SM.getSpellingColumnNumber(loop->getBeginLoc());
    if (loopColumn > 1)
      indentation.assign(loopColumn - 1, ' ');
  }
  const std::string replacementText =
      indentation + name + " = ascify::warp_reduce_add(" + name + ");";
  ct::Replacement Rep(SM, begin, endOffset - offset, replacementText);
  if (!insertSemanticReplacement(
          Rep, clang::FullSourceLoc(begin, SM), begin, end)) {
    rewrittenWarpAddReductionOffsets.erase(offset);
    return false;
  }

  static const dppCounter counter = {
      "ascify::warp_reduce_add", CONV_DEVICE_FUNC, API_RUNTIME, 0, FULL};
  Statistics::current().incrementCounter(
      counter, "canonical full-warp add reduction");
  needsCudaCompatHeader = true;
  return true;
}

bool AscifyAction::tagCanonicalBinaryReducer(
    const mat::MatchFinder::MatchResult &Result) {
  const auto *record =
      Result.Nodes.getNodeAs<clang::CXXRecordDecl>(sCanonicalBinaryReducer);
  if (!hasCubCompatHeader || record == nullptr || Result.Context == nullptr ||
      Result.SourceManager == nullptr)
    return false;

  const CanonicalReducerKind kind = classifyCanonicalBinaryReducer(
      record, *Result.Context, *Result.SourceManager);
  const llvm::StringRef tag = canonicalReducerTagName(kind);
  if (tag.empty())
    return false;

  auto &SM = *Result.SourceManager;
  const clang::SourceLocation includeLoc =
      SM.getFileLoc(firstCubCompatHeaderLoc);
  const clang::SourceLocation recordLoc =
      SM.getFileLoc(record->getBeginLoc());
  const clang::SourceLocation closeBrace =
      SM.getFileLoc(record->getBraceRange().getEnd());
  if (includeLoc.isInvalid() || recordLoc.isInvalid() ||
      closeBrace.isInvalid() ||
      SM.getFileID(includeLoc) != SM.getFileID(recordLoc) ||
      SM.getFileOffset(includeLoc) >= SM.getFileOffset(recordLoc) ||
      !SM.isWrittenInMainFile(closeBrace))
    return false;
  const unsigned offset = SM.getFileOffset(closeBrace);
  if (!taggedCanonicalReducerOffsets.insert(offset).second)
    return true;

  std::string annotation =
      "\n public:\n"
      "  using ascify_reduction_tag = ::aclcub::";
  annotation += tag.str();
  annotation += ";\n"
                "  using ascify_reduction_value_type = ";
  annotation += record->getDescribedClassTemplate()
                    ->getTemplateParameters()
                    ->getParam(0)
                    ->getNameAsString();
  annotation += ";\n"
                "  using ascify_reduction_owner_type = ";
  annotation += record->getNameAsString();
  annotation += ";\n";
  ct::Replacement Rep(SM, closeBrace, 0, annotation);
  if (!insertReplacement(
          Rep, clang::FullSourceLoc(closeBrace, SM))) {
    taggedCanonicalReducerOffsets.erase(offset);
    return false;
  }

  static const dppCounter counter = {
      "aclcub semantic reduction tag",
      CONV_DEVICE_FUNC, API_CUB, 0, FULL};
  Statistics::current().incrementCounter(
      counter, "canonical pure binary reduction functor");
  return true;
}

bool AscifyAction::cudaDeviceFuncCall(const mat::MatchFinder::MatchResult &Result) {
  return true;
}

bool AscifyAction::cubNamespacePrefix(const mat::MatchFinder::MatchResult &Result) {
  return true;
}

bool AscifyAction::cubUsingNamespaceDecl(const mat::MatchFinder::MatchResult &Result) {
  return false;
}

bool AscifyAction::cubFunctionTemplateDecl(const mat::MatchFinder::MatchResult &Result) {
  return false;
}

bool AscifyAction::cudaHostFuncCall(const mat::MatchFinder::MatchResult &Result) {
  return false;
}

bool AscifyAction::cudaOverloadedHostFuncCall(const mat::MatchFinder::MatchResult& Result) {
  return false;
}

bool AscifyAction::half2Member(const mat::MatchFinder::MatchResult &Result) {
  return false;
}

bool AscifyAction::dataTypeSelection(const mat::MatchFinder::MatchResult &Result) {
  return false;
}

bool AscifyAction::insertReplacement(const ct::Replacement &rep,
                                     const clang::FullSourceLoc &fullSL) {
  std::string errorMessage;
  if (!llcompat::insertReplacement(*replacements, rep, &errorMessage)) {
    clang::DiagnosticsEngine &DE = getCompilerInstance().getDiagnostics();
    const auto ID = DE.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "cannot apply Ascify source replacement: %0");
    DE.Report(fullSL, ID) << errorMessage;
    return false;
  }
  if (PrintStats || PrintStatsCSV) {
    rep.getLength();
    Statistics::current().lineTouched(fullSL.getExpansionLineNumber());
    Statistics::current().bytesChanged(rep.getLength());
  }
  return true;
}

bool AscifyAction::insertSemanticReplacement(
    const ct::Replacement &rep,
    const clang::FullSourceLoc &fullSL,
    clang::SourceLocation begin,
    clang::SourceLocation end) {
  auto &SM = getCompilerInstance().getSourceManager();
  begin = SM.getFileLoc(begin);
  end = SM.getFileLoc(end);
  if (begin.isInvalid() || end.isInvalid())
    return false;
  const std::pair<clang::FileID, unsigned> decomposedBegin =
      SM.getDecomposedLoc(begin);
  const std::pair<clang::FileID, unsigned> decomposedEnd =
      SM.getDecomposedLoc(end);
  if (decomposedBegin.first != decomposedEnd.first ||
      decomposedEnd.second <= decomposedBegin.second)
    return false;
  for (const SemanticRewriteRange &range : SemanticRewriteRanges) {
    if (range.file == decomposedBegin.first &&
        decomposedBegin.second < range.endOffset &&
        range.beginOffset < decomposedEnd.second) {
      clang::DiagnosticsEngine &DE = getCompilerInstance().getDiagnostics();
      const auto ID = DE.getCustomDiagID(
          clang::DiagnosticsEngine::Warning,
          "skipping overlapping Ascify semantic rewrite");
      DE.Report(fullSL, ID);
      return false;
    }
  }
  if (!insertReplacement(rep, fullSL))
    return false;
  SemanticRewriteRanges.push_back(
      {decomposedBegin.first, decomposedBegin.second, decomposedEnd.second});
  return true;
}
std::unique_ptr<clang::ASTConsumer> AscifyAction::CreateASTConsumer(clang::CompilerInstance &CI, StringRef) {
  Finder.reset(new mat::MatchFinder);
  davC310TargetRecipe.reset();
  // Replace the <<<...>>> language extension with a hip kernel launch
  Finder->addMatcher(mat::cudaKernelCallExpr(mat::isExpansionInMainFile()).bind(sCudaLaunchKernel), this);
  if (!NoLowerDeviceDoubleParams) {
    Finder->addMatcher(
        mat::parmVarDecl(mat::isExpansionInMainFile())
            .bind(sCudaGlobalScalarDoubleParam),
        this);
  }
  Finder->addMatcher(
      mat::varDecl(mat::isExpansionInMainFile())
          .bind(sCudaDefaultDim3),
      this);
  // Keep this matcher independent of API spelling and perform the CUDA API,
  // scalar type, enclosing-kernel, and address-provenance proof in the
  // callback. A failed proof makes no edit, so shared/local/helper pointers
  // cannot acquire a forged global-memory address-space cast.
  Finder->addMatcher(
      mat::callExpr(mat::callee(mat::functionDecl()),
                    mat::isExpansionInMainFile())
          .bind(sProvenGlobalAtomicCall),
      this);
  if (TargetPolicy == "dav-c310-vec" && SimtMathMode == "fast") {
    // Keep the matcher broad and perform the semantic proof in the callback.
    // This prevents source-name coupling and makes every failed proof a
    // no-op that falls back to the translated shuffle loop.
    Finder->addMatcher(
        mat::forStmt(mat::isExpansionInMainFile())
            .bind(sCanonicalWarpAddReduction),
        this);
    Finder->addMatcher(
        mat::cxxRecordDecl(mat::isDefinition(),
                           mat::isExpansionInMainFile())
            .bind(sCanonicalBinaryReducer),
        this);
    davC310TargetRecipe.registerMatchers(*Finder, this);
  }
  return Finder->newASTConsumer();
}

void AscifyAction::Ifndef(clang::SourceLocation Loc, const clang::Token &MacroNameTok, const clang::MacroDefinition &MD) {
  auditExternalNvidiaSampleHelperPreprocessorUse(
      Loc, MacroNameTok, "#ifndef");
}

void AscifyAction::Ifdef(clang::SourceLocation Loc,
                         const clang::Token &MacroNameTok,
                         const clang::MacroDefinition &) {
  auditExternalNvidiaSampleHelperPreprocessorUse(
      Loc, MacroNameTok, "#ifdef");
}

void AscifyAction::Defined(const clang::Token &MacroNameTok,
                           const clang::MacroDefinition &,
                           clang::SourceRange Range) {
  auditExternalNvidiaSampleHelperPreprocessorUse(
      Range.getBegin(), MacroNameTok, "defined");
}

void AscifyAction::MacroUndefined(
    const clang::Token &MacroNameTok,
    const clang::MacroDefinition &) {
  auditExternalNvidiaSampleHelperPreprocessorUse(
      MacroNameTok.getLocation(), MacroNameTok, "#undef");
}

void AscifyAction::MacroDefined(const clang::Token &MacroNameTok) {
  if (MacroNameTok.getIdentifierInfo() == nullptr)
    return;
  const llvm::StringRef name =
      MacroNameTok.getIdentifierInfo()->getName();
  if (name == "ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS" ||
      name == "ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR") {
    clang::SourceManager &sourceManager =
        getCompilerInstance().getSourceManager();
    if (!locationComesFromAscifyCudaCompat(
            sourceManager, MacroNameTok.getLocation())) {
      nvidiaSampleHelperOutputMacroEverDefined = true;
      nvidiaSampleHelperUnsupportedMacroUse = true;
      llvm::errs()
          << "Ascify NVIDIA sample-helper closure: reserved output macro '"
          << name << "' was already defined outside Ascify compat; all "
          << "helper edits kept\n";
    }
  }
  if (TargetRecipe != ascify::DavC310TargetRecipe::SimdRecipeName)
    return;
  if (isRowwiseSimdReservedMacro(name))
    rowwiseSimdMacrosEverDefined.insert(name.str());
}

void AscifyAction::MacroExpands(const clang::Token &MacroNameTok,
                                const clang::MacroDefinition &MD,
                                clang::SourceRange Range) {
  if (MacroNameTok.getIdentifierInfo() == nullptr)
    return;
  clang::SourceManager &sourceManager =
      getCompilerInstance().getSourceManager();
  const clang::MacroInfo *macroInfo = MD.getMacroInfo();
  if (macroInfo == nullptr ||
      !locationComesFromRecognizedNvidiaSampleHelper(
          sourceManager, macroInfo->getDefinitionLoc()))
    return;
  const clang::SourceLocation macroLocation =
      sourceManager.getExpansionLoc(MacroNameTok.getLocation());
  if (macroLocation.isInvalid()) {
    nvidiaSampleHelperUnsupportedMacroUse = true;
    return;
  }
  if (!sourceManager.isWrittenInMainFile(macroLocation)) {
    // Ignore implementation-internal expansions from the recognized helper,
    // but never remove the helper when an arbitrary local/user header relies
    // on one of its macros: this prototype does not rewrite that header.
    if (!locationComesFromRecognizedNvidiaSampleHelper(
            sourceManager, macroLocation)) {
      nvidiaSampleHelperUnsupportedMacroUse = true;
      llvm::errs()
          << "Ascify NVIDIA sample-helper closure: helper macro expansion "
          << "outside the main file keeps all helper edits\n";
    }
    return;
  }

  const llvm::StringRef name =
      MacroNameTok.getIdentifierInfo()->getName();
  if (name != "checkCudaErrors" && name != "getLastCudaError") {
    nvidiaSampleHelperUnsupportedMacroUse = true;
    return;
  }
  // Only rewrite a macro name token that is spelled directly in the main
  // file. For example, in `#define CHECK checkCudaErrors`, the nested helper
  // expansion points back to the source token `CHECK`; replacing that offset
  // with the longer helper name would corrupt the invocation. Macro-ID and
  // source-spelling checks make aliases and forwarding fail closed.
  const clang::SourceLocation macroNameLocation =
      MacroNameTok.getLocation();
  bool invalidSourceToken = false;
  const llvm::StringRef sourceToken = clang::Lexer::getSourceText(
      clang::CharSourceRange::getTokenRange(macroNameLocation,
                                             macroNameLocation),
      sourceManager, getCompilerInstance().getLangOpts(),
      &invalidSourceToken);
  if (macroNameLocation.isMacroID() || invalidSourceToken ||
      sourceToken != name) {
    nvidiaSampleHelperUnsupportedMacroUse = true;
    llvm::errs()
        << "Ascify NVIDIA sample-helper closure: macro '" << name
        << "' is not a direct source-token invocation; macro and include "
        << "kept\n";
    return;
  }
  if (!macroInfo->isFunctionLike() || macroInfo->getNumParams() != 1) {
    nvidiaSampleHelperUnsupportedMacroUse = true;
    nvidiaSampleHelperRewriteFailed = true;
    return;
  }
  if (!activeNvidiaSampleHelperMacroBodyMatches(
          name, *macroInfo, getCompilerInstance().getPreprocessor())) {
    nvidiaSampleHelperUnsupportedMacroUse = true;
    llvm::errs()
        << "Ascify NVIDIA sample-helper closure: active macro body for '"
        << name << "' is not admitted; macro and include kept\n";
    return;
  }

  const clang::SourceLocation rangeBegin =
      sourceManager.getExpansionLoc(Range.getBegin());
  clang::SourceLocation rangeEnd =
      sourceManager.getExpansionLoc(Range.getEnd());
  rangeEnd = clang::Lexer::getLocForEndOfToken(
      rangeEnd, 0, sourceManager,
      getCompilerInstance().getLangOpts());
  if (rangeBegin.isValid() && rangeEnd.isValid()) {
    const std::pair<clang::FileID, unsigned> begin =
        sourceManager.getDecomposedLoc(
            sourceManager.getFileLoc(rangeBegin));
    const std::pair<clang::FileID, unsigned> end =
        sourceManager.getDecomposedLoc(
            sourceManager.getFileLoc(rangeEnd));
    if (begin.first == end.first && end.second > begin.second) {
      const SemanticRewriteRange invocation =
          {begin.first, begin.second, end.second};
      nvidiaSampleHelperMacroRanges.push_back(invocation);
      nvidiaSampleHelperMacroCandidates.push_back(
          {name == "checkCudaErrors"
               ? NvidiaSampleHelperMacroKind::CheckCudaErrors
               : NvidiaSampleHelperMacroKind::GetLastCudaError,
           macroLocation, invocation,
           name == "getLastCudaError", false});
      nvidiaSampleHelperNormalMacroOffsets.insert(
          sourceManager.getFileOffset(
              sourceManager.getFileLoc(macroLocation)));
    } else {
      nvidiaSampleHelperRewriteFailed = true;
    }
  } else {
    nvidiaSampleHelperRewriteFailed = true;
  }

}

void AscifyAction::rewriteProvenNvidiaSampleHelperMacros() {
  // Replacements are accumulated without rollback support. Restrict closure
  // to one direct recognized include so include removal is atomic: duplicate
  // includes keep both directives and every original helper macro call.
  if (nvidiaSampleHelperIncludes.size() != 1) {
    nvidiaSampleHelperUnsupportedMacroUse = true;
    llvm::errs()
        << "Ascify NVIDIA sample-helper closure: expected exactly one "
        << "recognized helper_cuda.h include; all helper edits kept\n";
    return;
  }
  // Replacements are not rollback-capable. Complete every provenance,
  // preprocessor, status-domain, declaration, and range proof before the
  // first helper edit. One unsafe helper use therefore keeps every helper
  // macro invocation and the include in their original form.
  if (nvidiaSampleHelperUnsupportedMacroUse ||
      nvidiaSampleHelperUnsupportedDeclarationUse ||
      nvidiaSampleHelperRewriteFailed ||
      nvidiaSampleHelperOutputMacroEverDefined) {
    llvm::errs()
        << "Ascify NVIDIA sample-helper closure: global proof failed before "
        << "replacement; all helper edits kept\n";
    return;
  }
  for (const NvidiaSampleHelperMacroCandidate &candidate :
       nvidiaSampleHelperMacroCandidates) {
    if (!candidate.statusDomainProven) {
      nvidiaSampleHelperRewriteFailed = true;
      llvm::errs()
          << "Ascify NVIDIA sample-helper closure: candidate proof was "
          << "incomplete; all helper edits kept\n";
      return;
    }
  }
  // The raw-token audit runs later in ExecuteAction. Do not touch the global
  // replacement set here. EndSourceFileAction commits macro and include edits
  // together only after that audit and every other rewrite have completed.
  nvidiaSampleHelperReadyToCommit = true;
}

void AscifyAction::auditRawNvidiaSampleHelperToken(
    const clang::Token &token) {
  if (nvidiaSampleHelperIncludes.empty() || !token.isAnyIdentifier())
    return;
  const llvm::StringRef name = token.getRawIdentifier();
  if (name != "checkCudaErrors" && name != "getLastCudaError")
    return;
  clang::SourceManager &sourceManager =
      getCompilerInstance().getSourceManager();
  const clang::SourceLocation fileLoc =
      sourceManager.getFileLoc(token.getLocation());
  if (fileLoc.isInvalid() || !sourceManager.isWrittenInMainFile(fileLoc))
    return;
  const unsigned offset = sourceManager.getFileOffset(fileLoc);
  if (nvidiaSampleHelperNormalMacroOffsets.count(offset) != 0)
    return;
  if (!nvidiaSampleHelperUnsupportedMacroUse) {
    llvm::errs()
        << "Ascify NVIDIA sample-helper closure: residual raw PP use of '"
        << name << "' keeps helper_cuda.h\n";
  }
  nvidiaSampleHelperUnsupportedMacroUse = true;
}

void AscifyAction::auditExternalNvidiaSampleHelperPreprocessorUse(
    clang::SourceLocation location,
    const clang::Token &macroNameToken,
    llvm::StringRef directive) {
  if (macroNameToken.getIdentifierInfo() == nullptr)
    return;
  const llvm::StringRef name =
      macroNameToken.getIdentifierInfo()->getName();
  if (name != "checkCudaErrors" && name != "getLastCudaError")
    return;
  clang::SourceManager &sourceManager =
      getCompilerInstance().getSourceManager();
  const clang::SourceLocation expansion =
      sourceManager.getExpansionLoc(location);
  if (expansion.isInvalid() ||
      sourceManager.isWrittenInMainFile(expansion) ||
      locationComesFromRecognizedNvidiaSampleHelper(
          sourceManager, expansion))
    return;
  nvidiaSampleHelperUnsupportedMacroUse = true;
  llvm::errs()
      << "Ascify NVIDIA sample-helper closure: external " << directive
      << " use of '" << name << "' keeps all helper edits\n";
}

void AscifyAction::finalizeNvidiaSampleHelperClosure() {
  if (nvidiaSampleHelperIncludes.empty())
    return;
  const bool safeToRemove =
      nvidiaSampleHelperIncludes.size() == 1 &&
      nvidiaSampleHelperReadyToCommit &&
      nvidiaSampleHelperRawAuditCompleted &&
      !nvidiaSampleHelperUnsupportedMacroUse &&
      !nvidiaSampleHelperUnsupportedDeclarationUse &&
      !nvidiaSampleHelperRewriteFailed &&
      !nvidiaSampleHelperOutputMacroEverDefined;
  if (!safeToRemove) {
    llvm::errs()
        << "Ascify NVIDIA sample-helper closure: include kept"
        << ", check_rewrites=" << nvidiaSampleCheckCudaErrorsRewrites
        << ", get_last_rewrites="
        << nvidiaSampleGetLastCudaErrorRewrites
        << ", unsupported_macro="
        << (nvidiaSampleHelperUnsupportedMacroUse ? 1 : 0)
        << ", unsupported_declaration="
        << (nvidiaSampleHelperUnsupportedDeclarationUse ? 1 : 0)
        << ", rewrite_failed="
        << (nvidiaSampleHelperRewriteFailed ? 1 : 0)
        << ", ready="
        << (nvidiaSampleHelperReadyToCommit ? 1 : 0)
        << ", raw_audit_completed="
        << (nvidiaSampleHelperRawAuditCompleted ? 1 : 0) << "\n";
    return;
  }

  clang::SourceManager &sourceManager =
      getCompilerInstance().getSourceManager();
  ct::Replacements staged(*replacements);
  std::vector<SemanticRewriteRange> stagedSemanticRanges;
  std::vector<ct::Replacement> committedMacroReplacements;
  std::vector<clang::SourceLocation> committedMacroLocations;
  std::string stagingError;
  auto overlaps = [](const SemanticRewriteRange &lhs,
                     const SemanticRewriteRange &rhs) {
    return lhs.file == rhs.file && lhs.beginOffset < rhs.endOffset &&
           rhs.beginOffset < lhs.endOffset;
  };
  for (const NvidiaSampleHelperMacroCandidate &candidate :
       nvidiaSampleHelperMacroCandidates) {
    for (const SemanticRewriteRange &existing : SemanticRewriteRanges) {
      if (overlaps(candidate.invocationRange, existing)) {
        stagingError = "overlap with an existing semantic rewrite";
        break;
      }
    }
    for (const SemanticRewriteRange &pending : stagedSemanticRanges) {
      if (overlaps(candidate.invocationRange, pending)) {
        stagingError = "overlap between helper macro rewrites";
        break;
      }
    }
    if (!stagingError.empty())
      break;
    const llvm::StringRef originalName =
        candidate.kind == NvidiaSampleHelperMacroKind::CheckCudaErrors
            ? "checkCudaErrors"
            : "getLastCudaError";
    const llvm::StringRef replacementName =
        candidate.kind == NvidiaSampleHelperMacroKind::CheckCudaErrors
            ? "ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS"
            : "ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR";
    ct::Replacement replacement(sourceManager, candidate.nameLocation,
                                originalName.size(),
                                replacementName.str());
    if (!llcompat::insertReplacement(staged, replacement, &stagingError))
      break;
    committedMacroReplacements.push_back(replacement);
    committedMacroLocations.push_back(candidate.nameLocation);
    stagedSemanticRanges.push_back(candidate.invocationRange);
  }

  bool compatInsertedByClosure = hasCudaCompatHeader;
  const NvidiaSampleHelperInclude &include =
      nvidiaSampleHelperIncludes.front();
  const char *begin = sourceManager.getCharacterData(include.hashLocation);
  const char *end = sourceManager.getCharacterData(include.filenameEnd);
  if (stagingError.empty() &&
      (begin == nullptr || end == nullptr || end < begin)) {
    stagingError = "invalid direct include range";
  }
  std::string includeReplacementText;
  const bool helperNeedsCompat = !nvidiaSampleHelperMacroCandidates.empty();
  if (stagingError.empty() &&
      (needsCudaCompatHeader || helperNeedsCompat) &&
      !compatInsertedByClosure) {
    includeReplacementText =
        "#include <ascify/ascify_cuda_compat.hpp>";
    compatInsertedByClosure = true;
  }
  std::unique_ptr<ct::Replacement> includeReplacement;
  if (stagingError.empty()) {
    includeReplacement.reset(new ct::Replacement(
        sourceManager, include.hashLocation,
        static_cast<unsigned>(end - begin), includeReplacementText));
    if (!llcompat::insertReplacement(
            staged, *includeReplacement, &stagingError)) {
      // Keep stagingError for the fail-closed diagnostic below.
    }
  }
  if (!stagingError.empty()) {
    nvidiaSampleHelperRewriteFailed = true;
    llvm::errs()
        << "Ascify NVIDIA sample-helper closure: staged transaction failed; "
        << "all helper edits kept: " << stagingError << "\n";
    return;
  }

  // No shared state was changed before this point. Commit the complete set in
  // one assignment, then publish ranges, counters, and header state.
  *replacements = std::move(staged);
  SemanticRewriteRanges.insert(SemanticRewriteRanges.end(),
                               stagedSemanticRanges.begin(),
                               stagedSemanticRanges.end());
  for (NvidiaSampleHelperMacroCandidate &candidate :
       nvidiaSampleHelperMacroCandidates) {
    candidate.rewritten = true;
    if (candidate.kind == NvidiaSampleHelperMacroKind::CheckCudaErrors)
      ++nvidiaSampleCheckCudaErrorsRewrites;
    else
      ++nvidiaSampleGetLastCudaErrorRewrites;
  }
  needsCudaCompatHeader = needsCudaCompatHeader || helperNeedsCompat;
  if (compatInsertedByClosure)
    hasCudaCompatHeader = true;
  if (PrintStats || PrintStatsCSV) {
    for (size_t index = 0; index < committedMacroReplacements.size();
         ++index) {
      Statistics::current().lineTouched(
          clang::FullSourceLoc(committedMacroLocations[index], sourceManager)
              .getExpansionLineNumber());
      Statistics::current().bytesChanged(
          committedMacroReplacements[index].getLength());
    }
    Statistics::current().lineTouched(
        clang::FullSourceLoc(include.hashLocation, sourceManager)
            .getExpansionLineNumber());
    Statistics::current().bytesChanged(includeReplacement->getLength());
  }
  llvm::errs()
      << "Ascify NVIDIA sample-helper closure: include removed"
      << ", includes=1"
      << ", check_rewrites=" << nvidiaSampleCheckCudaErrorsRewrites
      << ", get_last_rewrites="
      << nvidiaSampleGetLastCudaErrorRewrites << "\n";
}

void AscifyAction::EndSourceFileAction() {
  finalizeNvidiaSampleHelperClosure();
  std::string includes;
  if (needsCudaCompatHeader && !hasCudaCompatHeader)
    includes += "#include <ascify/ascify_cuda_compat.hpp>\n";
  const bool useSimdTargetRecipe =
      TargetRecipe == ascify::DavC310TargetRecipe::SimdRecipeName;
  const bool hasSelectedTargetHeader =
      useSimdTargetRecipe ? hasDavC310SimdTargetHeader
                          : hasDavC310TargetHeader;
  if (needsDavC310TargetHeader && !hasSelectedTargetHeader) {
    includes += "#include <";
    includes += useSimdTargetRecipe
                    ? ascify::DavC310TargetRecipe::SimdTargetHeader
                    : ascify::DavC310TargetRecipe::TargetHeader;
    includes += ">\n";
  }
  if (includes.empty())
    return;

  if (useSimdTargetRecipe) {
    std::string shielded;
    std::vector<const char *> headerShieldNames;
    for (const char *name : RowwiseSimdReservedMacros) {
      if (isRowwiseSimdPublishedHeaderMacro(name))
        continue;
      headerShieldNames.push_back(name);
      shielded += "#pragma push_macro(\"";
      shielded += name;
      shielded += "\")\n#undef ";
      shielded += name;
      shielded += "\n";
    }
    shielded += includes;
    for (auto name = headerShieldNames.rbegin();
         name != headerShieldNames.rend(); ++name) {
      shielded += "#pragma pop_macro(\"";
      shielded += *name;
      shielded += "\")\n";
    }
    includes = std::move(shielded);
    const std::string mainOuterGuardName =
        findConventionalMainOuterGuard(getCompilerInstance());
    if (!mainOuterGuardName.empty()) {
      includes =
          "#pragma push_macro(\"" + mainOuterGuardName + "\")\n#undef " +
          mainOuterGuardName + "\n" + includes +
          "#pragma pop_macro(\"" + mainOuterGuardName + "\")\n";
    }
  }

  auto &SM = getCompilerInstance().getSourceManager();
  const clang::FileID mainFile = SM.getMainFileID();
  const clang::SourceLocation insertLoc =
      useSimdTargetRecipe
          ? SM.getLocForStartOfFile(mainFile)
          : (firstHeader ? firstHeaderLoc
                         : SM.getLocForStartOfFile(mainFile));
  ct::Replacement Rep(SM, insertLoc, 0, includes);
  insertReplacement(Rep, clang::FullSourceLoc(insertLoc, SM));
}

namespace {

/**
  * A silly little class to proxy PPCallbacks back to the AscifyAction class.
  */
class PPCallbackProxy : public clang::PPCallbacks {
  AscifyAction &ascifyAction;

public:
  explicit PPCallbackProxy(AscifyAction &action): ascifyAction(action) {}
  void InclusionDirective(clang::SourceLocation hash_loc, const clang::Token &include_token,
                          StringRef file_name, bool is_angled, clang::CharSourceRange filename_range,
#if LLVM_VERSION_MAJOR < 15
                          const clang::FileEntry *file,
#elif LLVM_VERSION_MAJOR == 15
                          Optional<clang::FileEntryRef> file,
#else
                          clang::OptionalFileEntryRef file,
#endif
                          StringRef search_path, StringRef relative_path,
#if LLVM_VERSION_MAJOR < 19
                          const clang::Module *SuggestedModule
#else
                          const clang::Module *SuggestedModule,
                          bool ModuleImported
#endif
#if LLVM_VERSION_MAJOR > 6
                        , clang::SrcMgr::CharacteristicKind FileType
#endif
                         ) override {
#if LLVM_VERSION_MAJOR < 15
    const StringRef resolvedFileName =
        file == nullptr ? StringRef() : file->getName();
#else
    // LLVM 15+ carries the selected path on FileEntryRef.  In LLVM 23 the
    // underlying FileEntry no longer exposes getName(), so keep the resolved
    // FileEntryRef as the source of truth instead of guessing include search.
    const StringRef resolvedFileName =
        file ? file->getName() : StringRef();
#endif
    ascifyAction.InclusionDirective(
        hash_loc, include_token, file_name, is_angled, filename_range,
        resolvedFileName, search_path, relative_path, SuggestedModule);
  }

  void PragmaDirective(clang::SourceLocation Loc, clang::PragmaIntroducerKind Introducer) override {
    ascifyAction.PragmaDirective(Loc, Introducer);
  }

  void Ifndef(clang::SourceLocation Loc, const clang::Token &MacroNameTok, const clang::MacroDefinition &MD) override {
    ascifyAction.Ifndef(Loc, MacroNameTok, MD);
  }

  void Ifdef(clang::SourceLocation Loc,
             const clang::Token &MacroNameTok,
             const clang::MacroDefinition &MD) override {
    ascifyAction.Ifdef(Loc, MacroNameTok, MD);
  }

  void Defined(const clang::Token &MacroNameTok,
               const clang::MacroDefinition &MD,
               clang::SourceRange Range) override {
    ascifyAction.Defined(MacroNameTok, MD, Range);
  }

  void MacroUndefined(
      const clang::Token &MacroNameTok,
      const clang::MacroDefinition &MD,
      const clang::MacroDirective *) override {
    ascifyAction.MacroUndefined(MacroNameTok, MD);
  }

  void MacroDefined(
      const clang::Token &MacroNameTok,
      const clang::MacroDirective *) override {
    ascifyAction.MacroDefined(MacroNameTok);
  }

  void MacroExpands(const clang::Token &MacroNameTok,
                    const clang::MacroDefinition &MD,
                    clang::SourceRange Range,
                    const clang::MacroArgs *) override {
    ascifyAction.MacroExpands(MacroNameTok, MD, Range);
  }

  void SourceRangeSkipped(clang::SourceRange Range, clang::SourceLocation EndifLoc) override {
    ascifyAction.AddSkippedSourceRange(Range);
  }
};
}

bool AscifyAction::BeginInvocation(clang::CompilerInstance &CI) {
  clang::DiagnosticsEngine &DE = CI.getDiagnostics();
  if (TargetPolicy != "portable" && TargetPolicy != "dav-c310-vec") {
    const auto ID = DE.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "unsupported --target-policy value '%0'; expected 'portable' or "
        "'dav-c310-vec'");
    DE.Report(ID) << TargetPolicy;
    return false;
  }
  if (SimtMathMode != "precise" && SimtMathMode != "fast") {
    const auto ID = DE.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "unsupported --simt-math value '%0'; expected 'precise' or 'fast'");
    DE.Report(ID) << SimtMathMode;
    return false;
  }
  if (TargetRecipe != "none" &&
      TargetRecipe != ascify::DavC310TargetRecipe::SimdRecipeName) {
    const auto ID = DE.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "unsupported --target-recipe value '%0'; expected 'none' or "
        "'dav-3510-rowwise-simd-v1'");
    DE.Report(ID) << TargetRecipe;
    return false;
  }
  if (TargetRecipe == ascify::DavC310TargetRecipe::SimdRecipeName &&
      (TargetPolicy != "dav-c310-vec" || SimtMathMode != "fast")) {
    const auto ID = DE.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "--target-recipe=dav-3510-rowwise-simd-v1 requires "
        "--target-policy=dav-c310-vec and --simt-math=fast");
    DE.Report(ID);
    return false;
  }
  llcompat::RetainExcludedConditionalBlocks(CI);
  return true;
}

void AscifyAction::ExecuteAction() {
  clang::Preprocessor &PP = getCompilerInstance().getPreprocessor();
  // Register yourself as the preprocessor callback, by proxy.
  PP.addPPCallbacks(std::unique_ptr<PPCallbackProxy>(new PPCallbackProxy(*this)));
#if LLVM_VERSION_MAJOR > 3
  Statistics::cudaVersionUsedByClang = Statistics::convertCudaToolkitVersion(clang::ToCudaVersion(PP.getTargetInfo().getSDKVersion()));
  llvm::errs() << " !!!!!!! CUDA SDK version detected: " << int(clang::ToCudaVersion(PP.getTargetInfo().getSDKVersion())) << "\n";
#endif
  // Now we're done futzing with the lexer, have the subclass proceeed with Sema and AST matching.
  clang::ASTFrontendAction::ExecuteAction();
  auto &SM = getCompilerInstance().getSourceManager();
  if (!nvidiaSampleHelperIncludes.empty()) {
    nvidiaSampleHelperUnsupportedDeclarationUse =
        hasUnsupportedNvidiaSampleHelperDeclarationUse();
    rewriteProvenNvidiaSampleHelperMacros();
  }
  if (TargetPolicy == "dav-c310-vec" && SimtMathMode == "fast") {
    if (TargetRecipe == ascify::DavC310TargetRecipe::SimdRecipeName) {
      const RawRowwiseSimdConflict rawMainConflict =
          rawFileRowwiseSimdConflict(
              getCompilerInstance(), SM.getMainFileID(), false);
      std::string conflictingMacro = rawMainConflict.macro;
      if (rowwiseSimdRawConflictingDeclaration.empty())
        rowwiseSimdRawConflictingDeclaration =
            rawMainConflict.declaration;
      for (const char *name : RowwiseSimdReservedMacros) {
        if (!conflictingMacro.empty())
          break;
        const clang::IdentifierInfo *identifier =
            PP.getIdentifierInfo(name);
        if (rowwiseSimdMacrosEverDefined.count(name) != 0 ||
            (identifier != nullptr && PP.isMacroDefined(identifier))) {
          conflictingMacro = name;
          break;
        }
      }
      if (!conflictingMacro.empty()) {
        const auto diagnostic =
            getCompilerInstance().getDiagnostics().getCustomDiagID(
                clang::DiagnosticsEngine::Error,
                "--target-recipe=dav-3510-rowwise-simd-v1 rejects reserved "
                "macro '%0'");
        getCompilerInstance().getDiagnostics().Report(diagnostic)
            << conflictingMacro;
        return;
      }
      if (!rowwiseSimdRawConflictingDeclaration.empty()) {
        const auto diagnostic =
            getCompilerInstance().getDiagnostics().getCustomDiagID(
                clang::DiagnosticsEngine::Error,
                "--target-recipe=dav-3510-rowwise-simd-v1 rejects raw "
                "input token '%0' reserved by the row-wise SIMD recipe");
        getCompilerInstance().getDiagnostics().Report(diagnostic)
            << rowwiseSimdRawConflictingDeclaration;
        return;
      }
      const RowwiseSimdConflictingDeclaration conflictingDeclaration =
          findRowwiseSimdConflictingDeclaration(
              getCompilerInstance().getASTContext());
      if (conflictingDeclaration) {
        const auto diagnostic =
            getCompilerInstance().getDiagnostics().getCustomDiagID(
                clang::DiagnosticsEngine::Error,
                "--target-recipe=dav-3510-rowwise-simd-v1 rejects "
                "input declaration or asm '%0' reserved by the row-wise "
                "SIMD recipe");
        getCompilerInstance().getDiagnostics()
            .Report(conflictingDeclaration.location, diagnostic)
            << conflictingDeclaration.name;
        return;
      }
    }
    const ascify::DavC310TargetRecipe::FinalizedEdits recipeEdits =
        davC310TargetRecipe.finalize(
            getCompilerInstance().getASTContext(), SM, PP.getLangOpts(),
            TargetRecipe == ascify::DavC310TargetRecipe::SimdRecipeName);
    unsigned insertedRecipeEdits = 0;
    for (const ascify::DavC310TargetRecipe::Edit &edit :
         recipeEdits.edits) {
      ct::Replacement replacement(SM, edit.location, 0, edit.text);
      if (insertReplacement(
              replacement, clang::FullSourceLoc(edit.location, SM)))
        ++insertedRecipeEdits;
    }
    needsDavC310TargetHeader =
        recipeEdits.needsTargetHeader && insertedRecipeEdits != 0;
    if (insertedRecipeEdits != 0) {
      llvm::errs()
          << "Ascify dav-c310 recipe: adapters(load="
          << recipeEdits.directLoadAdapters << ",store="
          << recipeEdits.directStoreAdapters
          << "), direct_wrappers(softmax="
          << recipeEdits.softmaxDirectWrappers << ",rmsnorm="
          << recipeEdits.rmsNormDirectWrappers << ",layernorm="
          << recipeEdits.layerNormDirectWrappers << ")\n";
    }
  }
  // Start lexing the specified input file.
  llcompat::Memory_Buffer FromFile = llcompat::getMemoryBuffer(SM);
  clang::Lexer RawLex(SM.getMainFileID(), FromFile, SM, PP.getLangOpts());
  RawLex.SetKeepWhitespaceMode(true);
  // Perform a token-level rewrite of CUDA identifiers to hip ones. The raw-mode lexer gives us enough
  // information to tell the difference between identifiers, string literals, and "other stuff". It also
  // ignores preprocessor directives, so this transformation will operate inside preprocessor-deleted code.
  rawTokenWindow.clear();
  clang::Token RawTok;
  RawLex.LexFromRawLexer(RawTok);
  while (RawTok.isNot(clang::tok::eof)) {
    while (rawTokenWindow.size() >= kRawTokenWindowCap)
      rawTokenWindow.pop_front();
    rawTokenWindow.push_back(RawTok);
    if (RewriteToken(RawLex, RawTok))
      continue;
    RawLex.LexFromRawLexer(RawTok);
  }
  // Every helper edit remains staged until the raw pass reaches EOF. Any
  // earlier target-recipe return leaves this false, so EndSourceFileAction
  // cannot commit a closure that skipped raw preprocessor auditing.
  nvidiaSampleHelperRawAuditCompleted = true;
}

void AscifyAction::AddSkippedSourceRange(clang::SourceRange Range) {
  SkippedSourceRanges.push_back(Range);
  if (TargetRecipe != ascify::DavC310TargetRecipe::SimdRecipeName)
    return;
  clang::SourceManager &sourceManager =
      getCompilerInstance().getSourceManager();
  const clang::SourceLocation begin =
      sourceManager.getFileLoc(Range.getBegin());
  if (begin.isInvalid() || sourceManager.isInSystemHeader(begin))
    return;
  const clang::FileID file = sourceManager.getFileID(begin);
  if (file.isInvalid() ||
      !rowwiseSimdRawScannedFiles.insert(file.getHashValue()).second)
    return;
  const RawRowwiseSimdConflict conflict =
      rawFileRowwiseSimdConflict(getCompilerInstance(), file, true);
  if (!conflict.macro.empty())
    rowwiseSimdMacrosEverDefined.insert(conflict.macro);
  if (rowwiseSimdRawConflictingDeclaration.empty())
    rowwiseSimdRawConflictingDeclaration = conflict.declaration;
}

void AscifyAction::run(const mat::MatchFinder::MatchResult &Result) {
  if (davC310TargetRecipe.collect(Result))
    return;
  if (Result.Nodes.getNodeAs<clang::CUDAKernelCallExpr>(sCudaLaunchKernel) != nullptr) {
    (void)cudaLaunchKernel(Result);
    return;
  }
  if (Result.Nodes.getNodeAs<clang::ParmVarDecl>(
          sCudaGlobalScalarDoubleParam) != nullptr) {
    (void)lowerCudaGlobalScalarDoubleParam(Result);
    return;
  }
  if (Result.Nodes.getNodeAs<clang::VarDecl>(sCudaDefaultDim3) != nullptr) {
    (void)rewriteCudaDefaultDim3(Result);
    return;
  }
  if (Result.Nodes.getNodeAs<clang::CallExpr>(
          sProvenGlobalAtomicCall) != nullptr) {
    (void)rewriteProvenGlobalAtomicCall(Result);
    return;
  }
  if (Result.Nodes.getNodeAs<clang::ForStmt>(
          sCanonicalWarpAddReduction) != nullptr) {
    (void)rewriteCanonicalWarpAddReduction(Result);
    return;
  }
  if (Result.Nodes.getNodeAs<clang::CXXRecordDecl>(
          sCanonicalBinaryReducer) != nullptr) {
    (void)tagCanonicalBinaryReducer(Result);
    return;
  }
}
