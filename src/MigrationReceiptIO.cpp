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

#include "MigrationReceiptIO.h"

#include "MigrationReceipt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

namespace ascify {

bool publishMigrationReceiptAtomically(const std::string &path,
                                       const MigrationReceipt &receipt,
                                       std::string &error) {
  if (path.empty()) {
    error = "migration receipt path is empty";
    return false;
  }

  llvm::SmallString<256> model(path);
  model += ".tmp-%%%%%%";
  llvm::SmallString<256> temporaryPath;
  int fileDescriptor = -1;
  std::error_code ec = llvm::sys::fs::createUniqueFile(
      model, fileDescriptor, temporaryPath);
  if (ec) {
    error = "cannot create migration receipt temporary file: " +
            ec.message();
    return false;
  }

  {
    llvm::raw_fd_ostream output(fileDescriptor, true);
    output << receipt.toJson();
    output.close();
    if (output.has_error()) {
      error = "cannot write migration receipt temporary file";
      llvm::sys::fs::remove(temporaryPath);
      return false;
    }
  }

  ec = llvm::sys::fs::rename(temporaryPath, path);
  if (ec) {
    error = "cannot atomically replace migration receipt: " + ec.message();
    llvm::sys::fs::remove(temporaryPath);
    return false;
  }
  return true;
}

} // namespace ascify
