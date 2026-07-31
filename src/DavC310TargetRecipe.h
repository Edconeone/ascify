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

#include <string>
#include <vector>

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/SourceLocation.h"

namespace clang {
class ASTContext;
class CXXRecordDecl;
class FunctionDecl;
class LangOptions;
class SourceManager;
} // namespace clang

namespace ascify {

// Collects AST facts during MatchFinder traversal, then emits conservative
// insert-only edits after the complete translation unit is available.  The
// caller owns applying the edits and inserting TargetHeader when requested.
class DavC310TargetRecipe {
public:
  struct Edit {
    enum class Kind {
      AdapterMarker,
      SoftmaxDirectWrapperPrologue,
      RmsNormDirectWrapperPrologue,
    };

    clang::SourceLocation location;
    std::string text;
    Kind kind;
  };

  struct FinalizedEdits {
    std::vector<Edit> edits;
    bool needsTargetHeader = false;
    unsigned directLoadAdapters = 0;
    unsigned directStoreAdapters = 0;
    unsigned softmaxDirectWrappers = 0;
    unsigned rmsNormDirectWrappers = 0;
  };

  static constexpr const char *TargetHeader =
      "ascify/target/dav_c310/rowwise_norm_recipes.hpp";
  static constexpr const char *FunctionBinding =
      "davC310TargetRecipeFunction";
  static constexpr const char *RecordBinding =
      "davC310TargetRecipeRecord";

  void reset();

  // AscifyAction should call this only under
  // dav-c310-vec + fast.  The matchers intentionally stay broad; all
  // algorithm and adapter proofs live in finalize().
  void registerMatchers(
      clang::ast_matchers::MatchFinder &finder,
      clang::ast_matchers::MatchFinder::MatchCallback *callback) const;

  // Returns true when this result belonged to the recipe.
  bool collect(
      const clang::ast_matchers::MatchFinder::MatchResult &result);

  FinalizedEdits finalize(clang::ASTContext &context,
                          clang::SourceManager &sourceManager,
                          const clang::LangOptions &langOptions);

private:
  std::vector<const clang::FunctionDecl *> functions;
  std::vector<const clang::CXXRecordDecl *> records;
};

} // namespace ascify
