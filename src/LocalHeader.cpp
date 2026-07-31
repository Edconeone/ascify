#include "LocalHeader.h"

#include <sstream>
#include <regex>
#include <set>
#include <vector>
#include <system_error>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "LLVMCompat.h"
#include "ArgParse.h"

using namespace clang;
using namespace clang::tooling;
using namespace llvm;
using namespace std;

static std::string normalizeSmallStringPath(SmallString<256> &p) {
  llvm::sys::path::remove_dots(p, true);

  SmallString<256> realBuf;
  std::error_code ec = llvm::sys::fs::real_path(p, realBuf);
  if (!ec) {
    return std::string(realBuf.str());
  }

  return std::string(p.str());
}

static bool pathExists(const std::string &p) {
  SmallString<256> in(p.begin(), p.end());

  SmallString<256> realBuf;
  std::error_code ec = llvm::sys::fs::real_path(in, realBuf);
  if (!ec) return true;

  SmallString<256> norm = in;
  llvm::sys::path::remove_dots(norm, true);
  return llvm::sys::fs::exists(norm);
}

namespace {
  static const std::regex LocalIncludeRe(
      R"(^\s*#\s*include\s*\"([^\"\n]+)\"\s*(?://.*)?$)", std::regex::ECMAScript);

  bool readFile(const std::string &path, std::string &out) {
    auto MBOrErr = llvm::MemoryBuffer::getFile(path);
    if (!MBOrErr) return false;
    out = MBOrErr->get()->getBuffer().str();
    return true;
  }

  bool resolveLocalIncludeInternal(const std::string &mainSourceAbsPath,
                                  const std::string &includeTok,
                                  std::string &outAbs) {
    auto tryCandidate = [&](const SmallString<256> &path) {
      SmallString<256> candidate(path);
      sys::path::append(candidate, includeTok);
      sys::path::remove_dots(candidate, true);
      if (!pathExists(std::string(candidate.str()))) return false;
      outAbs = normalizeSmallStringPath(candidate);
      return true;
    };

    // Match the compiler's quoted-include order first: the including file's
    // directory, followed by explicit -I roots.
    SmallString<256> base(mainSourceAbsPath);
    sys::path::remove_filename(base);
    if (tryCandidate(base)) return true;
    for (const std::string &includeDir : IncludeDirs) {
      SmallString<256> root(includeDir);
      if (tryCandidate(root)) return true;
    }

    // Project-qualified includes such as
    // "oneflow/core/cuda/layer_norm.cuh" are common in CUDA headers. When the
    // command line did not expose the project -I root to the option parser,
    // walk the including file's ancestors and use the nearest matching root.
    // This remains deterministic and never searches outside the ancestor
    // chain.
    SmallString<256> ancestor(base);
    while (!ancestor.empty()) {
      SmallString<256> parent(ancestor);
      sys::path::remove_filename(parent);
      if (parent == ancestor) break;
      ancestor = parent;
      if (tryCandidate(ancestor)) return true;
    }
    return false;
  }
} 

bool resolveLocalInclude(const std::string &mainSourceAbsPath,
                         const std::string &includeToken,
                         std::string &outAbsPath) {
  return resolveLocalIncludeInternal(mainSourceAbsPath, includeToken, outAbsPath);
}

bool collectLocalQuotedIncludes(const std::string &mainSourceAbsPath,
                                std::vector<std::string> &outHeaders) {
  std::string content;
  if (!readFile(mainSourceAbsPath, content)) {
    errs() << "\n" << sAscify << sError << "Cannot read source file: " << mainSourceAbsPath << "\n";
    return false;
  }

  std::set<std::string> uniq;
  std::smatch m;
  std::istringstream iss(content);
  std::string line;
  while (std::getline(iss, line)) {
    if (std::regex_match(line, m, LocalIncludeRe)) {
      std::string rel = m[1].str();
      std::string abs;
      if (resolveLocalIncludeInternal(mainSourceAbsPath, rel, abs)){
        uniq.insert(abs);
      } else {
        errs() << sAscify << sWarning
               << "Missing local header referenced: \"" << rel
               << "\" in " << mainSourceAbsPath << "\n";
      }
    }
  }
  outHeaders.assign(uniq.begin(), uniq.end());
  return true;
}

bool ascifyLocalHeaders(const std::string &mainSourceAbsPath,
                             const ct::CompilationDatabase *compDB,
                             ct::CommonOptionsParser *OptionsParserPtr,
                             const char *ascify_exe,
                             bool recursive) {

  std::vector<std::string> initial;
  if (!collectLocalQuotedIncludes(mainSourceAbsPath, initial)) {
    return false;
  }
  
  if (initial.empty()) {
    outs() << sAscify << "No local headers detected in " << mainSourceAbsPath << "\n";
    return true;
  }

  std::vector<std::string> work(initial.begin(), initial.end());
  std::set<std::string> processed;

  while (!work.empty()) {
    std::string hdr = work.back();
    work.pop_back();
    if (processed.count(hdr)) {
      errs() << sAscify << sWarning << "Duplicate local header reference ignored: " << hdr << "\n";
      continue;
    }
    processed.insert(hdr);

    std::string original;
    if (!readFile(hdr, original)) {
      errs() << sAscify << sError << "Cannot read header: " << hdr << "\n";
      continue;
    }

    std::string dppOut = hdr + ".dpp";
    bool ok = ascifySingleSource(hdr, dppOut, compDB, OptionsParserPtr,
                                  ascify_exe, mainSourceAbsPath, false);

    if (!ok) {
      errs() << sAscify << sError << "Ascify failed for header: " << hdr << "\n";
      return false;
    }

    if (recursive) {
      std::smatch m;
      std::istringstream iss(original);
      std::string line;
      while (std::getline(iss, line)) {
        if (std::regex_match(line, m, LocalIncludeRe)) {
          std::string rel = m[1].str();
          std::string abs;
          if (resolveLocalIncludeInternal(hdr, rel, abs) &&
              !processed.count(abs))
            work.push_back(abs);
        }
      }
    }
  }

  return true;
}
