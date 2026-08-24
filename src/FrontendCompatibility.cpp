#include "FrontendCompatibility.h"

#include <algorithm>
#include <filesystem>
#include <set>
#include <system_error>

#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/Refactoring.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace {

namespace fs = std::filesystem;

constexpr char kAdmissionHeader[] = R"ASCIFY(#ifndef ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_COOPERATIVE_GROUPS_H_
#define ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_COOPERATIVE_GROUPS_H_

#if defined(__CUDACC__) || defined(__CUDA__)
#define ASCIFY_FRONTEND_COMPAT_DEVICE_ __device__
#else
#define ASCIFY_FRONTEND_COMPAT_DEVICE_
#endif

namespace cooperative_groups {

class thread_block {
 public:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ void sync() const;
};

ASCIFY_FRONTEND_COMPAT_DEVICE_ thread_block this_thread_block();

ASCIFY_FRONTEND_COMPAT_DEVICE_ inline void sync(const thread_block& group) {
  group.sync();
}

}  // namespace cooperative_groups

#undef ASCIFY_FRONTEND_COMPAT_DEVICE_

#endif  // ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_COOPERATIVE_GROUPS_H_
)ASCIFY";

constexpr char kReductionPoison[] =
    "#error \"Ascify frontend compatibility ascify-admitted-v1 does not "
    "admit cooperative_groups/reduce.h\"\n";

constexpr char kProfileManifest[] =
    "schema=ascify.frontend-compat-profile.v1\n"
    "profile=ascify-admitted-v1\n"
    "file=cooperative_groups.h;bytes=702;sha256="
    "2f494aad929396ac870a469c58b783da183d99de2a965fb22e190bae91414657\n"
    "file=cooperative_groups/reduce.h;bytes=101;sha256="
    "75adbe65aeb5c2acfd63c9896376e67d270198e566080b9260a219ab99e2de8a\n";

static_assert(sizeof(kAdmissionHeader) - 1 == 702,
              "admission header identity drifted");
static_assert(sizeof(kReductionPoison) - 1 == 101,
              "reduction poison identity drifted");
static_assert(sizeof(kProfileManifest) - 1 == 291,
              "frontend profile manifest identity drifted");

struct RequiredProfileFile {
  const char* relativePath;
  std::uintmax_t bytes;
  const char* exactContent;
};

constexpr RequiredProfileFile kRequiredProfileFiles[] = {
    {"profile.manifest", 291, kProfileManifest},
    {"cooperative_groups.h", 702, kAdmissionHeader},
    {"cooperative_groups/reduce.h", 101, kReductionPoison},
};

std::string pathString(const fs::path& path) {
  return path.lexically_normal().generic_string();
}

bool pathIsWithin(const fs::path& root, const fs::path& candidate) {
  const fs::path relative = candidate.lexically_relative(root);
  if (relative.empty() || relative.is_absolute())
    return false;
  for (const fs::path& component : relative) {
    if (component == "..")
      return false;
  }
  return true;
}

bool readFile(const fs::path& path, std::string& content) {
  auto buffer = llvm::MemoryBuffer::getFile(pathString(path));
  if (!buffer)
    return false;
  content = buffer.get()->getBuffer().str();
  return true;
}

bool regularFileWithoutSymlink(const fs::path& path) {
  std::error_code error;
  const fs::file_status linkStatus = fs::symlink_status(path, error);
  return !error && !fs::is_symlink(linkStatus) &&
         fs::is_regular_file(linkStatus);
}

bool ValidateProfileRoot(const fs::path& candidate,
                         std::string& resolved,
                         std::string& error) {
  std::error_code filesystemError;
  if (!fs::exists(candidate, filesystemError) || filesystemError)
    return false;

  const fs::path canonicalRoot = fs::canonical(candidate, filesystemError);
  if (filesystemError || !fs::is_directory(canonicalRoot, filesystemError) ||
      filesystemError) {
    error = "frontend compatibility profile is not a readable directory: " +
            pathString(candidate);
    return false;
  }

  std::set<std::string> expectedFiles;
  for (const RequiredProfileFile& required : kRequiredProfileFiles) {
    expectedFiles.insert(required.relativePath);
    const fs::path lexicalPath = canonicalRoot / required.relativePath;
    if (!regularFileWithoutSymlink(lexicalPath)) {
      error = "frontend compatibility profile is missing required regular "
              "file '" + std::string(required.relativePath) + "'";
      return false;
    }
    const fs::path canonicalFile =
        fs::canonical(lexicalPath, filesystemError);
    if (filesystemError ||
        canonicalFile != (canonicalRoot / required.relativePath)) {
      error = "frontend compatibility profile file escapes or aliases its "
              "verified root: " + std::string(required.relativePath);
      return false;
    }
    std::string content;
    if (!readFile(canonicalFile, content) ||
        content.size() != required.bytes) {
      error = "frontend compatibility profile file has an unexpected size: " +
              std::string(required.relativePath);
      return false;
    }
    if (content != required.exactContent) {
      if (std::string(required.relativePath) == "profile.manifest") {
        error = "frontend compatibility profile manifest/version is not "
                "recognized";
      } else {
        error = "frontend compatibility profile exact content mismatch for '" +
                std::string(required.relativePath) + "'";
      }
      return false;
    }
  }

  std::set<std::string> observedFiles;
  std::set<std::string> observedDirectories;
  for (fs::recursive_directory_iterator iterator(canonicalRoot,
                                                  filesystemError), end;
       !filesystemError && iterator != end; iterator.increment(filesystemError)) {
    const fs::file_status linkStatus =
        fs::symlink_status(iterator->path(), filesystemError);
    if (filesystemError)
      break;
    if (fs::is_symlink(linkStatus)) {
      error = "frontend compatibility profile contains a symlink: " +
              pathString(iterator->path());
      return false;
    }
    const fs::path relative =
        fs::relative(iterator->path(), canonicalRoot, filesystemError);
    if (filesystemError)
      break;
    if (fs::is_regular_file(linkStatus)) {
      observedFiles.insert(pathString(relative));
    } else if (fs::is_directory(linkStatus)) {
      observedDirectories.insert(pathString(relative));
    } else {
      error = "frontend compatibility profile contains an unsupported entry";
      return false;
    }
  }
  if (filesystemError) {
    error = "cannot audit frontend compatibility profile layout: " +
            filesystemError.message();
    return false;
  }
  if (observedFiles != expectedFiles) {
    error = "frontend compatibility profile layout does not match its "
            "closed manifest";
    return false;
  }
  if (observedDirectories != std::set<std::string>{"cooperative_groups"}) {
    error = "frontend compatibility profile directory layout does not match "
            "its closed manifest";
    return false;
  }

  resolved = pathString(canonicalRoot);
  return true;
}

bool ResolveFrontendCompatibilityRoot(
    const std::string& profile,
    const char* ascifyExecutable,
    std::string& resolved,
    std::string& error) {
  static int executableAnchor;
  const std::string executable = llvm::sys::fs::getMainExecutable(
      ascifyExecutable, &executableAnchor);
  llvm::SmallString<256> installed(
      llvm::sys::path::parent_path(executable));
  llvm::sys::path::append(installed, "..");
  llvm::sys::path::append(installed,
                          ASCIFY_FRONTEND_COMPAT_INSTALL_RELPATH);
  llvm::sys::path::append(installed, profile);
  llvm::sys::path::remove_dots(installed, true);
  std::error_code filesystemError;
  const bool installedExists =
      fs::exists(fs::path(installed.str().str()), filesystemError);
  if (filesystemError) {
    error = "cannot inspect installed frontend compatibility profile: " +
            filesystemError.message();
    return false;
  }
  if (installedExists) {
    return ValidateProfileRoot(
        fs::path(installed.str().str()), resolved, error);
  }

  const fs::path canonicalExecutable =
      fs::canonical(fs::path(executable), filesystemError);
  if (filesystemError) {
    error = "cannot canonicalize ascify executable while resolving frontend "
            "compatibility data: " + filesystemError.message();
    return false;
  }
  const fs::path canonicalBuildRoot =
      fs::canonical(fs::path(ASCIFY_FRONTEND_COMPAT_BUILD_DIR),
                    filesystemError);
  if (filesystemError ||
      !pathIsWithin(canonicalBuildRoot, canonicalExecutable)) {
    error = "installed frontend compatibility profile is missing for '" +
            profile + "'";
    return false;
  }

  llvm::SmallString<256> source(ASCIFY_FRONTEND_COMPAT_SOURCE_DIR);
  llvm::sys::path::append(source, profile);
  if (ValidateProfileRoot(fs::path(source.str().str()), resolved, error))
    return true;
  if (error.empty()) {
    error = "cannot locate installed or source-tree frontend compatibility "
            "profile '" + profile + "'";
  }
  return false;
}

enum class CooperativeGroupsIncludeKind {
  Other,
  ExactAdmissionHeader,
  Unadmitted,
};

CooperativeGroupsIncludeKind classifyCooperativeGroupsInclude(
    const std::string& spelling) {
  std::string normalized = spelling;
  std::replace(normalized.begin(), normalized.end(), '\\', '/');
  if (normalized == "cooperative_groups.h")
    return CooperativeGroupsIncludeKind::ExactAdmissionHeader;
  const fs::path path(normalized);
  if (path.filename() == "cooperative_groups.h" ||
      normalized == "cooperative_groups" ||
      normalized.rfind("cooperative_groups/", 0) == 0 ||
      normalized.find("/cooperative_groups/") != std::string::npos) {
    return CooperativeGroupsIncludeKind::Unadmitted;
  }
  return CooperativeGroupsIncludeKind::Other;
}

}  // namespace

namespace ascify {

bool ConfigureFrontendCompatibility(
    clang::tooling::RefactoringTool& tool,
    const std::string& profile,
    const char* ascifyExecutable,
    FrontendCompatibilityConfig& config,
    std::string& error) {
  config = FrontendCompatibilityConfig{};
  if (profile == kNoFrontendCompatibility) {
    config.profile = kNoFrontendCompatibility;
    return true;
  }
  if (profile != kAdmittedFrontendCompatibilityV1) {
    error = "unsupported --frontend-compat value '" + profile +
            "'; expected 'none' or 'ascify-admitted-v1'";
    return false;
  }

  std::string root;
  if (!ResolveFrontendCompatibilityRoot(
          profile, ascifyExecutable, root, error)) {
    return false;
  }

  // Apply this adjuster last. Inserting at BEGIN makes this narrow product
  // header win over the CUDA installation without changing any other include.
  const std::string includeArgument = "-I" + root;
  tool.appendArgumentsAdjuster(clang::tooling::getInsertArgumentAdjuster(
      includeArgument.c_str(),
      clang::tooling::ArgumentInsertPosition::BEGIN));
  config.profile = profile;
  config.canonicalRoot = root;
  return true;
}

bool ValidateFrontendCompatibilityInclude(
    const FrontendCompatibilityConfig& config,
    const std::string& includeSpelling,
    const std::string& resolvedPath,
    std::string& error) {
  if (!config.enabled())
    return true;

  const CooperativeGroupsIncludeKind kind =
      classifyCooperativeGroupsInclude(includeSpelling);
  if (kind == CooperativeGroupsIncludeKind::Other)
    return true;
  if (kind == CooperativeGroupsIncludeKind::Unadmitted) {
    error = "frontend compatibility profile '" + config.profile +
            "' rejects unadmitted cooperative-groups header '" +
            includeSpelling + "'";
    return false;
  }
  if (resolvedPath.empty()) {
    error = "frontend compatibility profile '" + config.profile +
            "' could not prove the selected cooperative_groups.h source";
    return false;
  }

  std::error_code filesystemError;
  const fs::path actual = fs::canonical(resolvedPath, filesystemError);
  if (filesystemError) {
    error = "frontend compatibility profile '" + config.profile +
            "' cannot canonicalize selected cooperative_groups.h: " +
            filesystemError.message();
    return false;
  }
  const fs::path expected =
      fs::path(config.canonicalRoot) / "cooperative_groups.h";
  if (actual != expected) {
    error = "frontend compatibility profile '" + config.profile +
            "' requires cooperative_groups.h from its verified profile; got " +
            pathString(actual);
    return false;
  }
  return true;
}

}  // namespace ascify
