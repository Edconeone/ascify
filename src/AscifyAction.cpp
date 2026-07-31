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
#include <string>
#include "AscifyAction.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
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
#if LLVM_VERSION_MAJOR < 17
#include "clang/Basic/TargetInfo.h"
#endif
#include "LLVMCompat.h"
#include "ArgParse.h"
#include "ImplicitCudaHeaders.h"
#include "CUDA2DPP.h"
#include "StringUtils.h"

using namespace ascify;

const std::string sDPP = "DPP";
const std::string s_string_literal = "[string literal]";
// Matchers' names
const StringRef sCudaLaunchKernel = "cudaLaunchKernel";
const StringRef sCudaGlobalScalarDoubleParam = "cudaGlobalScalarDoubleParam";
const StringRef sCanonicalWarpAddReduction = "canonicalWarpAddReduction";
const StringRef sCanonicalBinaryReducer = "canonicalBinaryReducer";

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

static bool isDeclSpecifierLike(const clang::Token &t) {
  if (t.is(clang::tok::star))
    return true;
  if (t.isAnyIdentifier())
    return true;
  switch (t.getKind()) {
  case clang::tok::kw_struct:
  case clang::tok::kw_class:
  case clang::tok::kw_union:
  case clang::tok::kw_enum:
  case clang::tok::kw_const:
  case clang::tok::kw_volatile:
  case clang::tok::kw_restrict:
  case clang::tok::kw_unsigned:
  case clang::tok::kw_signed:
  case clang::tok::kw_void:
  case clang::tok::kw_bool:
  case clang::tok::kw_char:
  case clang::tok::kw_short:
  case clang::tok::kw_int:
  case clang::tok::kw_long:
  case clang::tok::kw_float:
  case clang::tok::kw_double:
  case clang::tok::kw_wchar_t:
  case clang::tok::kw_char16_t:
  case clang::tok::kw_char32_t:
  case clang::tok::kw_auto:
    return true;
  default:
    return false;
  }
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
  return name == "cudaDeviceGetAttribute" ||
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

int getNextNonZeroIdx(const std::deque<clang::Token> &rawTokenWindow, int &idx) {
  if (idx == 0) return 0;
  idx--;
  while(idx > 0 && rawTokenWindow[idx].getKind() == 0) {
    idx--;
  }
  return idx;
}


/// We should support two cases:
/// 1. ( ... ) malloc
/// -> currentIdx = getIdx("(")
/// 2. malloc
/// -> currentIdx = getIdx("malloc")
static bool matchParenCastBeforeMalloc(const std::deque<clang::Token> &win, int mallocIdx, int &currentIdx) {
  if (mallocIdx <= 0)
    return false;
  int castCloseIdx = getNextNonZeroIdx(win, mallocIdx);
  // if it is case 2, return true
  if (!win[castCloseIdx].is(clang::tok::r_paren)) {
    // check if the token is equal sign
    if (!win[castCloseIdx].is(clang::tok::equal)) {
      return false;
    }
    currentIdx = mallocIdx;
    return true;
  }
  int depth = 1;
  std::size_t j = castCloseIdx;
  while (j > 0) {
    --j;
    if (win[j].is(clang::tok::r_paren))
      depth++;
    else if (win[j].is(clang::tok::l_paren)) {
      if(j == 0) {
        return false;
      }
      depth--;
      if (depth == 0) {
        currentIdx = j;
        return true;
      }
    }
  }
  return false;
}

static unsigned ascifySourceOffset(const clang::SourceManager &SM, clang::SourceLocation loc) {
  return SM.getDecomposedLoc(SM.getFileLoc(loc)).second;
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

static bool scanMallocCallAndSemi(const char *p, llvm::StringRef &argOut, const char *&afterSemiOut) {
  while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
    ++p;
  if (*p != '(')
    return false;
  ++p;
  const char *argBegin = p;
  int depth = 1;
  while (*p && depth > 0) {
    if (*p == '"') {
      ++p;
      while (*p && *p != '"') {
        if (*p == '\\' && p[1])
          ++p;
        ++p;
      }
      if (*p == '"')
        ++p;
      continue;
    }
    if (*p == '\'') {
      ++p;
      while (*p && *p != '\'') {
        if (*p == '\\' && p[1])
          ++p;
        ++p;
      }
      if (*p == '\'')
        ++p;
      continue;
    }
    if (*p == '/' && p[1] == '/') {
      while (*p && *p != '\n')
        ++p;
      continue;
    }
    if (*p == '/' && p[1] == '*') {
      p += 2;
      while (*p && !(*p == '*' && p[1] == '/'))
        ++p;
      if (*p == '*' && p[1] == '/')
        p += 2;
      continue;
    }
    if (*p == '(')
      ++depth;
    else if (*p == ')')
      --depth;
    ++p;
  }
  if (depth != 0)
    return false;
  const char *argEnd = p - 1;
  argOut = llvm::StringRef(argBegin, argEnd - argBegin);
  while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
    ++p;
  if (*p != ';')
    return false;
  ++p;
  afterSemiOut = p;
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

bool AscifyAction::tryRewriteMallocDeviceAllocDecl(clang::Lexer &lex, clang::Token &tok) {
  if (!tok.isAnyIdentifier() || tok.getRawIdentifier() != "malloc")
    return false;
  if (rawTokenWindow.empty()) {
    llvm::errs() << "error: rawTokenWindow is empty\n";
    return false;
  }
  const std::size_t mi = rawTokenWindow.size() - 1;
  if (!(rawTokenWindow[mi].getLocation() == tok.getLocation())) {
    llvm::errs() << "error: mi location mismatch\n";
    return false;
  }

  int currentIdx = mi;
  if (!matchParenCastBeforeMalloc(rawTokenWindow, mi, currentIdx)) {
    llvm::errs() << "error: matchParenCastBeforeMalloc failed\n";
    return false;
  }
  if (currentIdx < 2) {
    llvm::errs() << "error: currentIdx < 2\n";
    return false;
  }
  int equalIdx = getNextNonZeroIdx(rawTokenWindow, currentIdx);
  if (!rawTokenWindow[equalIdx].is(clang::tok::equal)) {
    llvm::errs() << "error: equal sign not found\n";
    return false;
  }
  const std::size_t varIdx = getNextNonZeroIdx(rawTokenWindow, currentIdx);
  if (!rawTokenWindow[varIdx].isAnyIdentifier()) {
    llvm::errs() << "error: variable not found\n";
    return false;
  }

  std::size_t declStartIdx = varIdx;
  for (int t = varIdx; t > 0;) {
    --t;
    if (!isDeclSpecifierLike(rawTokenWindow[t]))
      break;
    declStartIdx = t;
  }

  clang::CompilerInstance &CI = getCompilerInstance();
  clang::SourceManager &SM = CI.getSourceManager();
  const clang::LangOptions &LO = CI.getLangOpts();
  clang::SourceLocation declStartLoc = rawTokenWindow[declStartIdx].getLocation();
  if (!SM.isWrittenInMainFile(declStartLoc) || !SM.isWrittenInMainFile(tok.getLocation()))
    return false;

  if (!AscifyAMAP) {
    clang::SourceRange sr(declStartLoc);
    for (const auto &skipped : SkippedSourceRanges) {
      if (skipped.fullyContains(sr))
        return false;
    }
  }

  const clang::SourceLocation mallocEnd =
      clang::Lexer::getLocForEndOfToken(tok.getLocation(), 0, SM, LO);
  const char *scan = SM.getCharacterData(mallocEnd);
  llvm::StringRef mallocArg;
  const char *afterSemi = nullptr;
  if (!scanMallocCallAndSemi(scan, mallocArg, afterSemi))
    return false;

  const clang::FileID mainFID = SM.getMainFileID();
  const char *mainBuf = SM.getCharacterData(SM.getLocForStartOfFile(mainFID));
  const char *declCh = SM.getCharacterData(declStartLoc);
  if (declCh < mainBuf || afterSemi < declCh)
    return false;

  const unsigned endOff = static_cast<unsigned>(afterSemi - mainBuf);
  clang::SourceLocation varLoc = rawTokenWindow[varIdx].getLocation();
  const clang::SourceLocation varEnd =
      clang::Lexer::getLocForEndOfToken(varLoc, 0, SM, LO);
  llvm::StringRef declPart =
      clang::Lexer::getSourceText(clang::CharSourceRange::getCharRange(declStartLoc, varEnd), SM, LO);

  llvm::SmallString<512> newText;
  newText += declPart;
  newText += ";\naclrtMallocHost(";
  newText += rawTokenWindow[varIdx].getRawIdentifier();
  newText += ", ";
  newText += mallocArg.trim();
  newText += ");";

  // Only replace when the span [decl, ';') fully covers this `malloc` spelling so the
  // initializer (including malloc) is removed and not left behind beside aclrtMalloc.
  const clang::SourceLocation mallocPast =
      clang::Lexer::getLocForEndOfToken(tok.getLocation(), 0, SM, LO);
  const char *mallocFirst = SM.getCharacterData(tok.getLocation());
  const char *mallocAfter = SM.getCharacterData(mallocPast);
  if (mallocFirst < declCh || mallocAfter > afterSemi)
    return false;
  if (llvm::StringRef(mallocFirst, static_cast<std::size_t>(mallocAfter - mallocFirst)) != "malloc")
    return false;

  const unsigned repLen = static_cast<unsigned>(afterSemi - declCh);
  ct::Replacement Rep(SM, declStartLoc, repLen, newText.str());
  insertReplacement(Rep, clang::FullSourceLoc(declStartLoc, SM));

  while (!tok.is(clang::tok::eof)) {
    if (ascifySourceOffset(SM, tok.getLocation()) >= endOff)
      break;
    lex.LexFromRawLexer(tok);
  }
  return true;
}

/**
  * Look at, and consider altering, a given token.
  *
  * If it's not a CUDA identifier, nothing happens.
  * If it's an unsupported CUDA identifier, a warning is emitted.
  * Otherwise, the source file is updated with the corresponding ascification.
  *
  * Returns true when the raw lexer was advanced past rewritten text (see malloc rewrite).
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

bool AscifyAction::RewriteToken(clang::Lexer &lex, clang::Token &tok) {
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
  if (name == "malloc" && tryRewriteMallocDeviceAllocDecl(lex, tok))
    return true;

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
                                      const clang::FileEntry*, StringRef,
                                      StringRef, const clang::Module*) {
                                        outs() << "File included: " << file_name << "\n";
  auto &SM = getCompilerInstance().getSourceManager();
  if (!SM.isWrittenInMainFile(hash_loc)) return;
  if (file_name == "ascify/ascify_cuda_compat.hpp")
    hasCudaCompatHeader = true;
  if (file_name == ascify::DavC310TargetRecipe::TargetHeader)
    hasDavC310TargetHeader = true;
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
  if (found == CUDA_INCLUDE_MAP.end()) return;
  bool exclude = Exclude(found->second);
  Statistics::current().incrementCounter(found->second, file_name.str());
  clang::SourceLocation sl = filename_range.getBegin();

  if (Statistics::isUnsupported(found->second)) {
    clang::DiagnosticsEngine &DE = getCompilerInstance().getDiagnostics();
    std::string sWarn = sDPP;
    const auto ID = DE.getCustomDiagID(clang::DiagnosticsEngine::Warning, "'%0' is unsupported header in '%1'.");
    DE.Report(sl, ID) << found->first << sWarn;
    return;
  }
  clang::StringRef newInclude;
  // Keep the same include type that the user gave.
  if (!exclude) {
    clang::SmallString<128> includeBuffer;
    llvm::StringRef name = found->second.dppName;
    if (is_angled) newInclude = llvm::Twine("<" + name+ ">").toStringRef(includeBuffer);
    else           newInclude = llvm::Twine("\"" + name + "\"").toStringRef(includeBuffer);
  } else {
    // hashLoc is location of the '#', thus replacing the whole include directive by empty newInclude starting with '#'.
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
}

void AscifyAction::EndSourceFileAction() {
  std::string includes;
  if (needsCudaCompatHeader && !hasCudaCompatHeader)
    includes += "#include <ascify/ascify_cuda_compat.hpp>\n";
  if (needsDavC310TargetHeader && !hasDavC310TargetHeader) {
    includes += "#include <";
    includes += ascify::DavC310TargetRecipe::TargetHeader;
    includes += ">\n";
  }
  if (includes.empty())
    return;

  auto &SM = getCompilerInstance().getSourceManager();
  const clang::FileID mainFile = SM.getMainFileID();
  const clang::SourceLocation insertLoc =
      firstHeader ? firstHeaderLoc : SM.getLocForStartOfFile(mainFile);
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
    auto f = file;
#else
    auto f = &file->getFileEntry();
#endif
    ascifyAction.InclusionDirective(hash_loc, include_token, file_name, is_angled, filename_range, f, search_path, relative_path, SuggestedModule);
  }

  void PragmaDirective(clang::SourceLocation Loc, clang::PragmaIntroducerKind Introducer) override {
    ascifyAction.PragmaDirective(Loc, Introducer);
  }

  void Ifndef(clang::SourceLocation Loc, const clang::Token &MacroNameTok, const clang::MacroDefinition &MD) override {
    ascifyAction.Ifndef(Loc, MacroNameTok, MD);
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
  if (TargetPolicy == "dav-c310-vec" && SimtMathMode == "fast") {
    const ascify::DavC310TargetRecipe::FinalizedEdits recipeEdits =
        davC310TargetRecipe.finalize(
            getCompilerInstance().getASTContext(), SM, PP.getLangOpts());
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
          << recipeEdits.rmsNormDirectWrappers << ")\n";
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
}

void AscifyAction::AddSkippedSourceRange(clang::SourceRange Range) {
  SkippedSourceRanges.push_back(Range);
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
